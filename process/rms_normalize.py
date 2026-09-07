"""RMS normalisation only - the last stage before the dataset.

    numpy/das_data_freq.npy  (551, 3500, 1058)  ->  das_data_norm.npy
    numpy/str_data_geom.npy  (551, 3500,   93)  ->  str_data_norm.npy

One scalar per domain, `TARGET_RMS / rms(input)`, applied to every shot.
Relative amplitudes between shots, traces and samples are untouched; only the
unit changes.  That is the whole stage.

No envelope gain.  process/logenv_process.py is the version that also applies
calculate_logscale; this one exists because the two domains have to be brought
onto a common amplitude scale for an unpaired translation - the DAS array has
an RMS around 1.6e+03 and the streamer around 1.1e-01, four orders of magnitude
apart, purely from how each instrument reports amplitude - and that is a
separate question from whether their dynamic range should be compressed.

Where the streamer input comes from
-----------------------------------
str_data_geom.npy, not str_data_freq.npy: whichever route produced `geom` -
the curvelet cascade or process/interpolate_kernel.py - has already applied its
low-pass on the 0.5 ms grid, where it doubles as the anti-alias filter for the
decimation to 1 ms.  A separate `freq` stage on the streamer side would be a
second pass with nothing left to remove, squaring the response near the corner
for no gain.  The DAS side does have one, since its low-cut and low-pass are
applied by process/freq_filter.py.

Parameters and measurements go next to each output as `<name>_meta.json` and
`<name>_meta.npz`, written whether or not the arrays are.

Edit the settings block below, then run from the repository root:

    python -m process.rms_normalize
"""

import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np

import config as C
from utils.data import create_memmap, get_project_root, npy_shape

# ---------------------------------------------------------------- settings --

# (tag, input, output).  The DAS side reads the band-passed array, the streamer
# side the geometry array - see the note above.
JOBS = (
    ("das", C.ARRAYS["das"]["freq"], C.ARRAYS["das"]["norm"]),
    ("str", C.ARRAYS["str"]["geom"], C.ARRAYS["str"]["norm"]),
)

# Draw a few shots and show them.  Writes nothing.
PREVIEW = True
PREVIEW_SHOTS = (0, 275, 550)
SAMPLES = None
CLIP_PERC = 99.0

# Normalise every shot and write the outputs.
WRITE_ALL = True

# Show the preview interactively, and optionally save it.
SHOW = False
SAVE_DIR = C.DATA_DIR

# Sidecar parameter/measurement files.
WRITE_META = True

# ----------------------------------------------------------------- derived --

TARGET_RMS = C.TARGET_RMS
RMS_SAMPLE_SHOTS = C.RMS_SAMPLE_SHOTS

# ---------------------------------------------------------------------------


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(get_project_root(), path)


def sample_indices(n_shots):
    if RMS_SAMPLE_SHOTS is None or RMS_SAMPLE_SHOTS >= n_shots:
        return np.arange(n_shots)
    return np.unique(np.linspace(0, n_shots - 1, RMS_SAMPLE_SHOTS).astype(int))


def measure_rms(data, idx):
    """Overall RMS over the sampled shots, plus the per-shot values.

    A sum of squares rather than a mean of means, so the result is the true RMS
    of the sampled block even though it is read shot by shot.
    """
    total, count = 0.0, 0
    per_shot = np.zeros(len(idx))
    for k, i in enumerate(idx):
        g = np.asarray(data[i], dtype=np.float64)
        ss = float(np.sum(g ** 2))
        total += ss
        count += g.size
        per_shot[k] = np.sqrt(ss / g.size)
    return float(np.sqrt(total / count)), per_shot


def preview(tag, data, norm_scale):
    n_shots = data.shape[0]
    bad = [s for s in PREVIEW_SHOTS if not 0 <= s < n_shots]
    if bad:
        raise SystemExit(f"preview shots out of range 0..{n_shots - 1}: {bad}")
    ns = data.shape[1] if SAMPLES is None else min(SAMPLES, data.shape[1])

    fig, axes = plt.subplots(1, len(PREVIEW_SHOTS),
                             figsize=(5.2 * len(PREVIEW_SHOTS), 6),
                             squeeze=False)
    for col, shot in enumerate(PREVIEW_SHOTS):
        g = np.asarray(data[shot], dtype=np.float32) * norm_scale
        clip = np.percentile(np.abs(g), CLIP_PERC)
        clip = clip if clip > 0 else 1.0
        ax = axes[0][col]
        ax.imshow(g[:ns], cmap="gray", vmin=-clip, vmax=clip, aspect="auto",
                  interpolation="nearest")
        ax.set_title(f"shot {shot}", fontsize=9)
        ax.set_xlabel("trace")
        if col == 0:
            ax.set_ylabel("sample")
        print(f"      shot {shot:>4}: rms {np.sqrt(np.mean(g.astype(np.float64) ** 2)):.4g}, "
              f"|max| {np.abs(g).max():.4g}")
    fig.suptitle(f"{tag}: RMS normalised to {TARGET_RMS:g} "
                 f"(panels clipped at {CLIP_PERC:g}%)")
    fig.tight_layout()

    if SAVE_DIR:
        out_dir = resolve(SAVE_DIR)
        os.makedirs(out_dir, exist_ok=True)
        p = os.path.join(out_dir, f"rms_normalize_{tag}.png")
        fig.savefig(p, dpi=130)
        print(f"    wrote {p}")
    if SHOW:
        plt.show()
    else:
        plt.close(fig)


def write_all(data, out_rel, norm_scale):
    out_path = resolve(out_rel)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if os.path.isfile(out_path):
        prev = npy_shape(out_path)
        print(f"    overwriting {out_rel} (was {prev})")
        del prev

    n_shots = data.shape[0]
    out = create_memmap(out_path, data.shape)
    print(f"    -> {out_rel}  {out.nbytes / 1e9:.2f} GB")

    rms_out = np.zeros(n_shots)
    lo, hi = np.inf, -np.inf
    t0 = time.time()
    for i in range(n_shots):
        y = np.asarray(data[i], dtype=np.float32) * norm_scale
        out[i] = y
        rms_out[i] = np.sqrt(np.mean(y.astype(np.float64) ** 2))
        lo = min(lo, float(y.min()))
        hi = max(hi, float(y.max()))
        if (i + 1) % 100 == 0 or i + 1 == n_shots:
            el = time.time() - t0
            print(f"      {i + 1}/{n_shots} shots  {el:.0f}s elapsed")
    out.flush()
    # The overall RMS is the quantity TARGET_RMS sets, and it comes out exact.
    # The mean of the per-shot values is always lower - Jensen - and by more
    # when the shots vary more; reported alongside so the two are not confused.
    overall = float(np.sqrt(np.mean(rms_out ** 2)))
    print(f"    amplitude range [{lo:.4g}, {hi:.4g}]")
    print(f"    overall rms {overall:.6f} (target {TARGET_RMS:g}), "
          f"mean of per-shot rms {rms_out.mean():.4f}, "
          f"per-shot {rms_out.min():.4f}..{rms_out.max():.4f}")
    return dict(amplitude_min=lo, amplitude_max=hi,
                value_range_max_abs=max(abs(lo), abs(hi)),
                rms_output_overall=overall), rms_out


def write_meta(out_rel, tag, params, info, arrays):
    stem = resolve(os.path.splitext(out_rel)[0])
    with open(stem + "_meta.json", "w", encoding="utf-8") as f:
        json.dump({"tag": tag, "parameters": params, "measured": info}, f,
                  indent=2, sort_keys=True)
    np.savez(stem + "_meta.npz", **arrays)
    print(f"    wrote {os.path.basename(stem)}_meta.json / _meta.npz")


def main():
    for tag, in_rel, out_rel in JOBS:
        in_path = resolve(in_rel)
        if not os.path.isfile(in_path):
            raise SystemExit(f"not found: {in_path}")

        data = np.load(in_path, mmap_mode="r")
        idx = sample_indices(data.shape[0])
        print(f"{tag}: {in_rel}  {data.shape}")

        rms_in, rms_in_per_shot = measure_rms(data, idx)
        norm_scale = TARGET_RMS / rms_in
        print(f"    rms {rms_in:.6g} over {len(idx)} shots -> {TARGET_RMS:g}  "
              f"(scale {norm_scale:.6g})")
        recorded = C.INPUT_RMS.get(tag)
        if recorded is not None:
            drift = rms_in / recorded - 1.0
            note = "matches" if abs(drift) < 1e-6 else "DRIFTED - update it"
            print(f"    config.INPUT_RMS[{tag!r}] = {recorded:.6g}  "
                  f"({100 * drift:+.4f} %, {note})")

        if PREVIEW:
            preview(tag, data, norm_scale)

        params = dict(
            input=in_rel.replace("\\", "/"),
            output=out_rel.replace("\\", "/"),
            shape=list(data.shape),
            target_rms=TARGET_RMS,
            rms_sample_shots=RMS_SAMPLE_SHOTS,
            envelope_gain="none",
            filter={name: C.FILTER_PARAMS[name]
                    for name in C.NORM_FILTER_CHAIN[tag]},
        )
        info = dict(rms_input=rms_in, norm_scale=norm_scale,
                    rms_shots_measured=len(idx),
                    config_input_rms=C.INPUT_RMS.get(tag),
                    config_norm_scale=C.NORM_SCALE.get(tag),
                    output_written=bool(WRITE_ALL))
        arrays = dict(rms_input_per_shot=rms_in_per_shot,
                      rms_input_shot_index=idx)

        if WRITE_ALL:
            out_info, rms_out = write_all(data, out_rel, norm_scale)
            info.update(rms_output_mean=float(rms_out.mean()),
                        rms_output_min=float(rms_out.min()),
                        rms_output_max=float(rms_out.max()), **out_info)
            arrays["rms_output_per_shot"] = rms_out
        else:
            print(f"    WRITE_ALL is off - {out_rel} not touched")

        if WRITE_META:
            write_meta(out_rel, tag, params, info, arrays)
        print()


if __name__ == "__main__":
    main()
