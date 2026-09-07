# Pohang DAS ↔ Streamer Domain Translation

Unpaired image-to-image translation between **Pohang seafloor DAS shot gathers**
(domain A) and **conventional streamer shot gathers** (domain B).

The code is extracted from the `seismic_section_cyclegan` research repository;
only the Pohang-related part is kept here. The Smeaheia, Westsea and 3-D
coordinate/InfoGAN experiments from that repository are *not* included.

Three training variants are provided:

| variant | script | generators | translation loss |
|---|---|---|---|
| CycleGAN (two-way) | `train/train_pohang_shore.py` | `G_A2B`, `G_B2A` | identity + cycle consistency (+ SSIM) |
| One-way | `train/train_pohang_shore_oneway.py` | `G_A2B`, `G_B2A` | one-way variant of the above |
| CUT | `train/train_pohang_shore_cut.py` | `G_A2B` only | identity + PatchNCE (contrastive) |

The CycleGAN/one-way variants use the isotropic U-Net generator in
`network/pohang_shore_network.py`; the CUT variant uses the anisotropic
generator plus `PatchSampleF` / `PatchNCELoss` in
`network/pohang_shore_network_aniso.py`. Both discriminators are hinge-loss
PatchGAN-style convolutional critics.

## Layout

```
config/       yacs configs, one per training variant
process/      data preparation: log-envelope scaling, synthetic 3-D section building
module/       dataset + SSIM loss
network/      generators, discriminators, PatchNCE
train/        training entry points (DDP, AMP)
inference/    checkpoint loading + qualitative comparison plots
utils/        project paths, DDP helpers, weight init, signal processing
data/         inputs (not tracked)
checkpoint/   saved training state (not tracked)
```

## Data

`module/dataset_pohang_shore.py` expects two arrays of shape
`(n_shot, n_time, n_trace)` under `data/pohang_shore/`:

- `das_data_log.npy` — DAS gathers; traces `44:116` (72 channels) are used for training
- `str_data_log.npy` — streamer gathers

Both are log-envelope-scaled versions of the raw frequency-domain-filtered
gathers. Produce them from `das_data_freq.npy` / `str_data_freq.npy` with:

```bash
python -m process.logenv_process
```

The dataset clips to the 99th percentile, normalises by half the maximum
absolute value, crops randomly along time, and augments with random trace flips
and polarity reversal.

### Receiver-geometry QC — `process/check_receiver_geometry.py`

Reads the trace headers of `data/pohang_shore/{das,streamer}/data0*.segy`,
checks that the four files of each kind agree on their receiver coordinates,
and maps DAS against streamer receivers on a 1:1 metric scale (equal aspect,
north arrow, scale bar). `SHOW_SOURCES` adds a second panel with the shot
positions, which span ~2 km against the arrays' ~90 m.

```bash
python -m process.check_receiver_geometry
```

Geometry of the current data, as reported by the script:

| | channels | spacing | array length | records per file |
|---|---|---|---|---|
| DAS | 3175 | 0.257 m | 809 m | 135 / 137 / 139 / 140 |
| streamer | 24 | 3.12 m | 71.8 m | 135 / 137 / 139 / 140 |

Both kinds cover the same shot records. The two arrays are collinear: the 24
streamer channels sit on top of the southern end of the DAS fibre. The
streamer files agree on their receiver coordinates to 0.000 m.

The script also guards against a download failure seen once on the DAS files,
where every trace header past the first few dozen came through zeroed. It
reports header coverage per file, and falls back to bounding the channel count
from the trace counts when the headers are unusable — so a truncated or
corrupted transfer shows up as a coverage figure well below 100 % rather than
as silently missing receivers.

### Building the training arrays

`process/make_streamer_npy.py` turns the streamer SEG-Y files into
`data/pohang_shore/str_data_raw.npy`, shape `(551, 3500, 24)` float32. It
resamples 0.5 ms → 1 ms with an anti-alias filter (`scipy.signal.resample_poly`,
zero-phase) and crops to 3500 samples, putting the gathers on the same time grid
as the DAS files. Sample values are otherwise untouched — bandpass, envelope
scaling and normalisation belong to the later stages. `WRITE_META` also saves
the per-shot source and receiver coordinates for offset filtering.

`process/plot_shots.py` draws every STEP-th shot of such an array, tiled on one
page, memory-mapped so it also works on the multi-GB DAS array.

```bash
python -m process.make_streamer_npy
python -m process.plot_shots
```

### Synthetic 3-D sections — `process/make_pohang.py`

A separate, standalone script carried over from the same repository. It is *not*
part of the DAS ↔ streamer training path: it builds synthetic 3-D reflectivity
volumes from the interpreted Pohang Eastsea horizons, by densifying the
interpreted strata into sub-layers, laying Gaussian reflectors along each one and
convolving with a Ricker wavelet.

Inputs, under `data/pohang/`:

- `Eastsea_3D.npy` — the migrated 3-D volume `(nz, nx, ny)`
- `Eastsea_3D_layer.npy` — 5 interpreted horizons `(5, nx, ny)`

The `__main__` block cleans up the horizons (TV denoising, fixing crossings
between horizons 1–2), crops depth to `100:700`, writes the cropped volume and
corrected horizons to `data/pohang/sections/0.npy` and `data/pohang/layers/0.npy`,
then plots them and **exits**. The `Parallel(n_jobs=20)` fan-out below that
`exit()` — which writes 40 synthetic realisations to `data/synt_sparse/` — is
therefore dead unless you remove the `exit()` call. That is how the script was
left in the original repository; it was copied as-is.

Needs two extra dependencies (`scikit-image`, `joblib`) and pins `scipy<1.15`,
because `scipy.signal.ricker` was removed in SciPy 1.15.

```bash
python -m process.make_pohang
```

## Setup

```bash
pip install -r requirements.txt
```

Install a PyTorch build matching your CUDA version first — see the note at the
top of `requirements.txt`.

## Running

Scripts import from the repository root, so run them **as modules from the repo
root**, not by path:

```bash
python -m train.train_pohang_shore          # config/pohang_shore_das_str.yaml
python -m train.train_pohang_shore_oneway   # config/pohang_shore_das_str_oneway.yaml
python -m train.train_pohang_shore_cut      # config/pohang_shore_das_str_cut.yaml
```

```bash
python -m inference.test_pohang_shore        # edit `epoch` / `device` at the top
python -m inference.test_pohang_cut
python -m inference.test_pohang_cut_compare  # also inverts the log scaling
```

Training uses `torch.multiprocessing.spawn` + `DistributedDataParallel`; set the
GPU ids in the config's `GPUS` list. Checkpoints are written to
`checkpoint/<TAG>_<epoch>` every epoch and thinned down to every
`TRAIN.CHECK_EPOCH`-th one.

The inference scripts have the epoch number and CUDA device hard-coded near the
top — edit them before running.

## Configs

The original repository did not ship the `config/*.yaml` files, so the three
configs here were reconstructed from the keys the code reads. Architecture keys
(`A_CHANNELS`/`B_CHANNELS` = 1, generator/discriminator widths, norm/spectral/
attention switches) and the loss weights are **starting points, not the values
used in the original experiments** — retune them for your run.
