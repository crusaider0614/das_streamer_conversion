"""Band-limit the streamer array before the trace interpolation.

    numpy/str_data_raw.npy  (551, 8000, 24) @ 0.5 ms
    numpy/str_data_bl.npy   (551, 7000, 24) @ 0.5 ms

The `raw -> bl` stage.  Separate from `process/freq_filter.py`, which is the
`geom -> freq` stage at the far end of the pipeline: this one runs on the
SEG-Y's own 0.5 ms grid, on 24 channels, and it does crop the record, none of
which is true there.  Keeping them apart is also what keeps the two meanings of
"filtered" from colliding again - `_bl` is band-limited for the interpolation,
`_freq` is the band-pass applied to the finished geometry.

Why this runs before the interpolation
--------------------------------------
The recorded nodes sit 3.00 m apart at 14 of the 23 intervals and 3.25 m at the
other 9, so the spatial Nyquist is 0.1614 c/m at the mean and 0.1538 c/m at the
widest gap - and it is the widest gap that decides.  An event of apparent
velocity v folds over above f = v / (2 dx).  Above that the interpolation
cannot recover the true dip: it fits the fold-over instead, and the alias
replica survives in the output as a false event near k = 0.45 cycles/m.

The direct arrival is the steepest thing here, and cross-correlating adjacent
traces puts it at 3.1 samples of 0.5 ms per trace - 1950 m/s, not the 1500 m/s
a marine gather is usually assumed to hold.  That gives 300 Hz at the widest
gap, which is where LP_F_CUT now sits.  It was 220 Hz on the 1500 m/s
assumption, which the data does not support.

Aliasing is not a cliff, though.  Measured against the DAS - 0.75 m sampling
over the same aperture, so four times the Nyquist and able to see what the
streamer folds - between 4 and 10 % of the energy in each band already sits
past the streamer's Nyquist, rising gradually with frequency rather than
switching on at 300 Hz.  That is an upper bound, since the seafloor DAS also
records slow interface waves a towed streamer does not.

The crop
--------
f_filtering transforms the whole trace at once, so the filter is circular.
PAD_FRONT zeros absorb the wrap and CROP_TOP removes them again, leaving the
time origin unchanged.  CROP_BOTTOM additionally drops the last 1000 samples of
the record: 8000 + 2000 - 2000 - 1000 = 7000, which is what str_data_intp.npy
holds and therefore what a rerun has to reproduce.

Edit the settings block below, then run from the repository root:

    python -m process.band_limit
"""

import os
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config as C  # noqa: E402
from utils.data import create_memmap, get_project_root  # noqa: E402
from utils.process import f_filter, f_filtering  # noqa: E402
from utils import mute as MU  # noqa: E402

# ---------------------------------------------------------------- settings --

INPUT_NPY = C.ARRAYS["str"]["raw"]
OUTPUT_NPY = C.ARRAYS["str"]["bl"]

# Sample interval of the input, in microseconds.  str_data_raw.npy is on the
# SEG-Y's own grid, not the 1 ms one the rest of the pipeline uses.
DT_US = 500

# The filter comes from config.FILTER_PARAMS["str"] - the single place the
# streamer band is defined.  process/decimate_intp.py imports these names from
# here for its anti-alias pass, and process/rms_normalize.py records the same
# config entry in the sidecar, so all three agree by construction.  Change the
# corner in config.py, not here.
_F = C.FILTER_PARAMS["str_bl"]
LOWPASS = _F["lowpass"]
LP_F_CUT = _F["lp_f_cut"]
LP_ORDER = _F["lp_order"]
LP_DECAY = _F["lp_decay"]
HIGHPASS = _F["highpass"]
HP_F_CUT = _F["hp_f_cut"]
HP_ORDER = _F["hp_order"]
HP_DECAY = _F["hp_decay"]
ZERO_DC = _F["zero_dc"]

# 8000 + 2000 - 2000 - 1000 = 7000 output samples.  See the docstring.
PAD_FRONT = 2000
CROP_TOP = 2000
CROP_BOTTOM = 1000

# Trace spacing, for the alias-onset reference numbers.
DX_M = 3.1196
VELOCITIES = [1500.0, 2000.0]

# Mute ahead of the direct arrival on the way out.  Doing it here rather
# than later is the point: the interpolation never sees the pre-arrival
# noise, so it cannot spend coefficients on it or turn it into the
# coherent wavetrains it otherwise manufactures there.
MUTE_AFTER = True

REPORT_RESPONSE = True
PROBE_HZ = [5, 10, 50, 100, 150, 180, 200, 210, 220, 230, 240, 260, 300, 400]

# One shot before and after, plus the response.  None to skip.
FIGURE_PATH = os.path.join(C.DATA_DIR, "band_limit.png")
FIGURE_SHOT = 5
FIGURE_SAMPLES = 2000
CLIP_PERC = 99.0

# ---------------------------------------------------------------------------


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(get_project_root(), path)


def build_mask(nt, dt):
    mask = np.ones(nt)
    if LOWPASS:
        mask *= f_filter(nt, dt, LP_F_CUT, LP_ORDER, LP_DECAY, is_lowpass=True)
    if HIGHPASS:
        mask *= f_filter(nt, dt, HP_F_CUT, HP_ORDER, HP_DECAY, is_lowpass=False)
    return mask


def report(mask, nt, dt):
    f = np.abs(np.fft.fftfreq(nt, dt))
    half = nt // 2
    print("  amplitude response:")
    for p in PROBE_HZ:
        i = int(np.argmin(np.abs(f[:half] - p)))
        db = 20 * np.log10(max(mask[i], 1e-300))
        print(f"    {p:>5.0f} Hz: {mask[i]:9.6f}  ({db:8.2f} dB)")

    ff = f[:half]
    db = 20 * np.log10(np.maximum(mask[:half], 1e-300))
    for target in (-3.0, -40.0, -80.0):
        past = np.where((ff > LP_F_CUT * 0.5) & (db < target))[0]
        edge = ff[past[0]] if len(past) else float("nan")
        print(f"  low-pass crosses {target:6.0f} dB at {edge:7.1f} Hz")

    k_nyq = 1.0 / (2.0 * DX_M)
    print(f"  spatial Nyquist {k_nyq:.4f} c/m at dx = {DX_M:g} m; alias onset:")
    for v in VELOCITIES:
        print(f"    v = {v:6.0f} m/s -> {v * k_nyq:6.1f} Hz")


def figure(raw, filtered, mask, nt, dt, path):
    f = np.abs(np.fft.fftfreq(nt, dt))
    half = nt // 2
    order = np.argsort(f[:half])

    fig, axes = plt.subplots(1, 4, figsize=(19, 6))
    axes[0].plot(f[:half][order],
                 20 * np.log10(np.maximum(mask[:half][order], 1e-300)))
    axes[0].axvline(LP_F_CUT, color="k", ls="--", lw=1,
                    label=f"{LP_F_CUT:g} Hz cut")
    for v in VELOCITIES:
        axes[0].axvline(v / (2 * DX_M), color="r", ls=":", lw=1,
                        label=f"{v:g} m/s alias onset")
    axes[0].set_xlim(0, 400)
    axes[0].set_ylim(-100, 5)
    axes[0].set_xlabel("frequency [Hz]")
    axes[0].set_ylabel("dB")
    axes[0].set_title("filter response")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    clip = np.percentile(np.abs(raw), CLIP_PERC)
    for ax, img, title in ((axes[1], raw, "before"), (axes[2], filtered, "after")):
        ax.imshow(img, cmap="gray", vmin=-clip, vmax=clip, aspect="auto",
                  interpolation="nearest")
        ax.set_title(f"shot {FIGURE_SHOT} {title}")
        ax.set_xlabel("trace")
    axes[1].set_ylabel("sample")

    for img, lab in ((raw, "before"), (filtered, "after")):
        sp = np.abs(np.fft.rfft(img, axis=0)).mean(axis=1)
        fr = np.fft.rfftfreq(img.shape[0], dt)
        axes[3].plot(fr, 20 * np.log10(np.maximum(sp / sp.max(), 1e-12)),
                     label=lab)
    axes[3].set_xlim(0, 400)
    axes[3].set_ylim(-80, 5)
    axes[3].set_xlabel("frequency [Hz]")
    axes[3].set_ylabel("dB re peak")
    axes[3].set_title("mean trace spectrum")
    axes[3].legend()
    axes[3].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    in_path = resolve(INPUT_NPY)
    if not os.path.isfile(in_path):
        raise SystemExit(f"not found: {in_path}")
    out_path = resolve(OUTPUT_NPY)

    data = np.load(in_path, mmap_mode="r")
    n_shots, nt, n_traces = data.shape
    dt = DT_US * 1e-6

    nt_pad = nt + PAD_FRONT
    if CROP_TOP + CROP_BOTTOM >= nt_pad:
        raise SystemExit(f"crops {CROP_TOP} + {CROP_BOTTOM} leave nothing of "
                         f"the {nt_pad} padded samples")
    keep = slice(CROP_TOP, nt_pad - CROP_BOTTOM)
    nt_out = keep.stop - keep.start

    print(f"{INPUT_NPY}\n  {n_shots} shots x {nt} samples @ {DT_US / 1000:g} ms "
          f"x {n_traces} traces")
    print(f"  pad {PAD_FRONT} -> filter on {nt_pad} -> crop {CROP_TOP} top / "
          f"{CROP_BOTTOM} bottom -> {nt_out} samples")
    print("  low-cut  " + (f"{HP_F_CUT:g} Hz" if HIGHPASS else "off"))
    print("  low-pass " + (f"{LP_F_CUT:g} Hz (order {LP_ORDER:g}, "
                           f"decay {LP_DECAY:g})" if LOWPASS else "off"))

    mask = build_mask(nt_pad, dt)
    if REPORT_RESPONSE:
        report(mask, nt_pad, dt)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    out = create_memmap(out_path, (n_shots, nt_out, n_traces))
    print(f"  -> {OUTPUT_NPY}  {out.nbytes / 1e6:.0f} MB")

    t_mute = None
    if MUTE_AFTER:
        t_mute, _ = MU.boundary_ms(n_traces)
        print(MU.summary(t_mute, nt_out, DT_US / 1000.0))

    pad = np.zeros((PAD_FRONT, n_traces), dtype=np.float64)
    raw0 = filt0 = None
    t0 = time.time()
    for i in range(n_shots):
        g = np.asarray(data[i], dtype=np.float64)
        y = f_filtering(np.concatenate([pad, g], axis=0), mask,
                        is_zeroout=ZERO_DC)[keep]
        if i == FIGURE_SHOT:
            raw0 = g[:FIGURE_SAMPLES].copy()
            filt0 = y[:FIGURE_SAMPLES].copy()
        if t_mute is not None:
            y = y * MU.weights(t_mute[i], nt_out, DT_US / 1000.0)
        out[i] = y.astype(np.float32)
        if (i + 1) % 100 == 0 or i + 1 == n_shots:
            el = time.time() - t0
            print(f"    {i + 1}/{n_shots} shots  {el:.0f}s elapsed")
    out.flush()

    print(f"\n  wrote {out_path}  shape {out.shape}")

    if FIGURE_PATH and raw0 is not None:
        p = resolve(FIGURE_PATH)
        figure(raw0, filt0, mask, nt_pad, dt, p)
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
