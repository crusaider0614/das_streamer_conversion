"""Log-scale the pre-interpolation streamer gather and look at it.  Writes nothing.

    numpy/str_data_raw.npy  (551, 8000, 24) @ 0.5 ms

Reading `raw` rather than `bl`: this is the gather as recorded, before the
220 Hz anti-alias low-pass that process/band_limit.py applies on the way into
the interpolation.  Point INPUT_NPY at C.ARRAYS["str"]["bl"] to see the gain
fitted after that filter instead.

Why do this before the interpolation rather than after
------------------------------------------------------
The curvelet solve keeps a fixed fraction of the largest coefficients
(THRESHOLD in process/intp_sweep.py, 5 %).  "Largest" is decided over the whole
gather, and a marine gather is dominated by its direct arrival, so most of the
coefficient budget is spent there.  Everything below it - the reflections that
actually matter - sits near or under the threshold and is reconstructed badly
or not at all.

That shows up in the finished array as the period-4 stripe: recorded positions
keep their amplitude because the misfit term holds them to the data, while the
interpolated positions lose whatever the threshold discarded.  Measured on
str_data_intp.npy the step is worst exactly where the weak energy lives - about
+15 to +35 % between 12 and 50 Hz, against 2-4 % above 75 Hz - which is what
"the direct arrival is fine, the ones below are the problem" looks like in the
frequency domain.

Balancing the amplitudes first is the standard answer: after an envelope gain
the deep events are comparable in size to the shallow ones, so the threshold no
longer throws them away preferentially.

Two steps, as in process/logenv_process.py
------------------------------------------
  1. RMS normalise to config.TARGET_RMS - one scalar, so `log_base` means the
     same thing here as it does anywhere else.
  2. Envelope gain from utils.process.calculate_logscale, with
     config.LOG_SCALE_PARAMS["str_pre"].

Only the first and the last are drawn.  The RMS step is a single scalar and
every panel is percentile-clipped on its own, so a panel for it would be
pixel-identical to the input - the number is reported on the console instead.

If this goes into the pipeline for real, the gain has to be kept and undone
after the interpolation, otherwise everything downstream is working in a
transformed amplitude that nothing accounts for.  The gain is defined on 24
traces and the output has 93, so undoing it means interpolating the gain itself
onto the finer grid - it is smooth by construction, so that is a plain
resample, not another inverse problem.  This script does none of that; it
applies the gain to a few shots and draws them.

Edit the settings block below, then run from the repository root:

    python -m process.logenv_before_intp
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import config as C
from utils.data import get_project_root
from utils.process import calculate_logscale, envelope_1d

# ---------------------------------------------------------------- settings --

# The gather to scale.  "raw" is as recorded, 8000 samples; "bl" is the same
# thing after the 220 Hz band-limit, 7000 samples.
INPUT_NPY = C.ARRAYS["str"]["bl"]

# Sample interval of that array, in microseconds.  `bl` and `raw` are both on
# the SEG-Y's own grid, not the 1 ms one.
DT_US = 500

# Shots to draw.
SHOTS = (0, 275, 550)

CLIP_PERC = 99.0

# Show interactively, and optionally save.  A falsy SAVE_DIR skips saving.
SHOW = True
SAVE_DIR = None

# Zeros appended before the gain is fitted, dropped afterwards; 0 disables it.
# calculate_logscale's envelope is a Hilbert transform built from an FFT along
# time, so it is circular and the end of the record spreads into its start.
# 2000 samples is 1 s on this 0.5 ms grid.  Same as PAD_TAIL in
# process/logenv_process.py.
PAD_TAIL = 2000

# ----------------------------------------------------------------- derived --

TARGET_RMS = C.TARGET_RMS
RMS_SAMPLE_SHOTS = C.RMS_SAMPLE_SHOTS
LSP = C.LOG_SCALE_PARAMS["str_pre"]

# ---------------------------------------------------------------------------


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(get_project_root(), path)


def sample_indices(n_shots):
    if RMS_SAMPLE_SHOTS is None or RMS_SAMPLE_SHOTS >= n_shots:
        return np.arange(n_shots)
    return np.unique(np.linspace(0, n_shots - 1, RMS_SAMPLE_SHOTS).astype(int))


def measure_rms(data, idx):
    total, count = 0.0, 0
    for i in idx:
        g = np.asarray(data[i], dtype=np.float64)
        total += float(np.sum(g ** 2))
        count += g.size
    return float(np.sqrt(total / count))


def logscale(gather):
    """calculate_logscale on a tail-padded gather, with the pad dropped.

    See PAD_TAIL: the envelope is circular, so the record has to be separated
    from its own start before the gain is fitted.
    """
    if not PAD_TAIL:
        return calculate_logscale(gather, **LSP)
    nt = gather.shape[0]
    pad = np.zeros((PAD_TAIL, gather.shape[1]), dtype=gather.dtype)
    return calculate_logscale(np.concatenate([gather, pad], axis=0), **LSP)[:nt]


def coefficient_budget(gather, frac=0.05):
    """How the gather's energy is split between its shallow and deep halves.

    Not the curvelet threshold itself, but the thing the threshold responds to:
    the share of the largest `frac` of samples that come from the deep part.
    If the gain is doing its job this number goes up.
    """
    nt = gather.shape[0]
    a = np.abs(gather)
    cut = np.quantile(a, 1.0 - frac)
    big = a >= cut
    deep = np.zeros_like(big)
    deep[nt // 4:] = True
    return big[deep].sum() / max(big.sum(), 1)


def panels_figure(panels):
    """One row per shot, whole record: input, log-scaled, and the gain.

    No RMS-normalised column.  Each panel is percentile-clipped on its own, so
    multiplying by a single scalar changes the image not at all - the panel
    carried no information.  norm_scale is reported on the console instead.
    """
    fig, axes = plt.subplots(len(panels), 3, figsize=(14, 5.0 * len(panels)),
                             squeeze=False)
    for row, (shot, raw, log, scale) in enumerate(panels):
        cols = ((raw, "input", "seismic"),
                (log, "log envelope", "seismic"),
                (scale, "gain", "viridis"))
        for col, (img, name, cmap) in enumerate(cols):
            ax = axes[row][col]
            if cmap == "seismic":
                clip = np.percentile(np.abs(img), CLIP_PERC)
                clip = clip if clip > 0 else 1.0
                ax.imshow(img, cmap=cmap, vmin=-clip, vmax=clip, aspect="auto",
                          interpolation="nearest")
            else:
                ax.imshow(img, cmap=cmap, aspect="auto",
                          interpolation="nearest")
            ax.set_title(f"shot {shot}: {name}", fontsize=9)
            ax.set_xlabel("trace")
            if col == 0:
                ax.set_ylabel("sample")
    fig.suptitle(f"log-envelope gain on the pre-interpolation gather "
                 f"(base {LSP['log_base']:g}, sigma {LSP['smooth_sigma']})")
    fig.tight_layout()
    return fig


def profile_figure(profiles, dt):
    """Mean |amplitude| against time, before and after, on a log axis.

    The compression is the whole point, and it is much easier to read here than
    off a clipped gather: the curves should go from spanning several decades to
    spanning well under one.
    """
    fig, axes = plt.subplots(1, len(profiles), figsize=(6 * len(profiles), 5),
                             squeeze=False)
    for col, (shot, nor, log) in enumerate(profiles):
        t = np.arange(len(nor)) * dt
        ax = axes[0][col]
        ax.semilogy(t, np.maximum(nor, 1e-12), lw=0.9, label="RMS-normalised")
        ax.semilogy(t, np.maximum(log, 1e-12), lw=0.9, label="log envelope")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("mean |amplitude| across traces")
        ax.set_title(f"shot {shot}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
    fig.suptitle("amplitude against time - how much dynamic range the gain removes")
    fig.tight_layout()
    return fig


def main():
    in_path = resolve(INPUT_NPY)
    if not os.path.isfile(in_path):
        raise SystemExit(f"not found: {in_path}")

    data = np.load(in_path, mmap_mode="r")
    n_shots, nt, n_traces = data.shape
    dt = DT_US * 1e-6
    bad = [s for s in SHOTS if not 0 <= s < n_shots]
    if bad:
        raise SystemExit(f"shots out of range 0..{n_shots - 1}: {bad}")

    print(f"{INPUT_NPY}\n  {n_shots} shots x {nt} samples @ {DT_US / 1000:g} ms "
          f"x {n_traces} traces")

    idx = sample_indices(n_shots)
    rms_in = measure_rms(data, idx)
    norm_scale = TARGET_RMS / rms_in
    print(f"  rms {rms_in:.6g} over {len(idx)} shots -> {TARGET_RMS:g}  "
          f"(scale {norm_scale:.6g})")
    print(f"  log scale params {LSP}")

    panels, profiles = [], []
    for shot in SHOTS:
        raw = np.asarray(data[shot], dtype=np.float32)
        nor = raw * norm_scale
        scale = logscale(nor)
        log = nor * scale

        env = float(np.median(envelope_1d(nor)))
        d_before = coefficient_budget(nor)
        d_after = coefficient_budget(log)
        print(f"    shot {shot:>4}: median envelope {env:.4g}, "
              f"base/median {LSP['log_base'] / env:.3g}")
        print(f"              dynamic range {np.abs(nor).max() / max(env, 1e-30):.4g}"
              f" -> {np.abs(log).max() / max(float(np.median(envelope_1d(log))), 1e-30):.4g}")
        print(f"              deep share of the top 5 % of samples "
              f"{100 * d_before:.1f} % -> {100 * d_after:.1f} %")

        panels.append((shot, raw, log, scale))
        profiles.append((shot, np.abs(nor).mean(axis=1), np.abs(log).mean(axis=1)))

    figs = [("gathers", panels_figure(panels)),
            ("profiles", profile_figure(profiles, dt))]

    if SAVE_DIR:
        out_dir = resolve(SAVE_DIR)
        os.makedirs(out_dir, exist_ok=True)
        for name, fig in figs:
            p = os.path.join(out_dir, f"logenv_before_intp_{name}.png")
            fig.savefig(p, dpi=130)
            print(f"  wrote {p}")

    if SHOW:
        plt.show()
    else:
        for _, fig in figs:
            plt.close(fig)


if __name__ == "__main__":
    main()
