"""Shot gathers before and after translation, both in true-amplitude units.

    left   das_data_norm.npy      the DAS as it was before the envelope gain
    right  das_data_fake_str.npy  with that gain taken back off

Copied from process/plot_shots.py.  The difference is what each panel holds:
that script draws whatever is in the file, and the translated file is in the
log-envelope domain, where a gain that varies down the trace has flattened the
amplitude decay.  Reading a translation there is misleading - the gain hides
exactly the amplitude behaviour the translation is supposed to change - so
this one undoes the gain first and shows both sides on the amplitude scale the
data really has.

The inversion
-------------
`utils.process.calculate_norscale_inversion` is the gain's own inverse: it
does not know the scale that was applied, it re-estimates it from the scaled
data and divides it back out, fifty times, until the estimate stops moving.

It is run with the STREAMER parameters, C.LOG_SCALE_PARAMS["str"], because
that is the domain the translated file is in - inference/translate_das.py
brings the generator's output back through the streamer's `value_range`, and
the generator was trained against streamer gathers that carried the streamer's
gain.  Using the DAS parameters here would undo a gain that was never applied.

The front pad matters.  process/logenv_process.py computed the forward gain on
a gather with PAD_FRONT zeros in front of it, which pins
`calculate_logscale`'s `data_env_log.min()` to log10(log_base) whatever the
gather contains; without it that minimum floats with the gather and the
estimate comes back systematically different.  So the same pad goes on here
and is cropped off afterwards.

Both sides end up on the same footing: process/rms_normalize.py put each
domain at TARGET_RMS = 0.1 before the gain, so the two panels can be read
against each other rather than only against themselves.

Shot order
----------
das_data_fake_str.npy has its shots SORTED along the line; das_data_norm.npy
is in RECORDING order.  Shot i is not the same shot in the two files, so the
DAS side is reordered by process.shot_geometry before anything is drawn.

Edit the settings block below, then run from the repository root:

    python -m inference.plot_translated_shots
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np

import config as C
import process.shot_geometry as G
from utils.data import get_project_root
from utils.process import calculate_norscale_inversion

# ---------------------------------------------------------------- settings --

# The translated file, and the DAS before the envelope gain.  Relative paths
# resolve against the project root, not the working directory.
FAKE_NPY = C.ARRAYS["das"]["fake_str"]
DAS_NPY = C.ARRAYS["das"]["norm"]

# Which shots: every STEP-th from START up to STOP (STOP = None -> the end).
# Indices are into the TRANSLATED file, so they are sorted-order indices.
STEP = 50
START = 0
STOP = None

# Whose gain to undo, and how hard to look for it.  "str" - see the header.
# 50 is calculate_norscale_inversion's own default and costs 5.7 s for a
# 264-trace gather.  Measured against a gather whose gain was applied and then
# taken back off, the error relative to the gather's peak is 2.3e-3 mean after
# 5 iterations, 6.1e-5 after 20 and 3.6e-7 after 50, so 20 is plenty if the
# wait is annoying.
LOG_TAG = "str"
ITERATIONS = 50

# Zeros in front of the gather while the gain is estimated, in samples.  Must
# match process/logenv_process.PAD_FRONT - see the header.
PAD_FRONT = 2000

# Set False to draw the translated file as it is, still carrying the gain.
INVERT_ENV = True

DT_MS = 1.0

# Colour clip: symmetric at this percentile of |amplitude|, per panel.  Both
# domains sit at TARGET_RMS before the gain, so SHARED_CLIP is meaningful here
# in a way it is not in process/plot_shots.py.
PERC = 99.0
SHARED_CLIP = False

CMAP = "seismic"

# Where to write the figures; None only shows them.  The shot index is
# appended to the file name.
SAVE_PATH = None

# ---------------------------------------------------------------------------


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(get_project_root(), path)


def open_array(path, name):
    p = resolve(path)
    if not os.path.isfile(p):
        raise SystemExit(f"not found: {p}  ({name})")
    a = np.load(p, mmap_mode="r")
    if a.ndim != 3:
        raise SystemExit(f"{name} is {a.shape}; expected "
                         f"(shots, samples, channels)")
    return a


def invert_env(gather):
    """Take the envelope gain back off one shot gather.

    The pad is zeros, so it survives the division unchanged and keeps
    `data_env_log.min()` pinned where the forward pass had it; it is cropped
    off before the result is returned.
    """
    lsp = dict(C.LOG_SCALE_PARAMS[LOG_TAG])
    g = np.asarray(gather, dtype=np.float64)
    if PAD_FRONT:
        g = np.concatenate([np.zeros((PAD_FRONT, g.shape[1])), g], axis=0)
    _, est = calculate_norscale_inversion(g, iterations=ITERATIONS, **lsp)
    return est[PAD_FRONT:PAD_FRONT + gather.shape[0]] if PAD_FRONT else est


def clip_value(gather, perc):
    v = float(np.percentile(np.abs(gather), perc))
    return v if v > 0 else float(np.abs(gather).max()) or 1.0


def draw(ax, gather, clip=None):
    n_t, n_ch = gather.shape
    v = clip if clip is not None else clip_value(gather, PERC)
    ax.imshow(gather, cmap=CMAP, vmin=-v, vmax=v, aspect="auto",
              interpolation="nearest",
              extent=[0, n_ch, n_t * DT_MS / 1000.0, 0])
    return v


def main():
    fake = open_array(FAKE_NPY, "translated")
    das = open_array(DAS_NPY, "DAS before the gain")
    if fake.shape != das.shape:
        raise SystemExit(f"translated {fake.shape} against DAS {das.shape}; "
                         f"they have to be the same grid")

    n_shots, n_t, n_ch = fake.shape
    order = np.asarray(G.sorted_order()["orig_idx"], dtype=int)
    if len(order) != n_shots:
        raise SystemExit(f"shot_geometry has {len(order)} shots, the arrays "
                         f"have {n_shots}")

    print(f"{resolve(FAKE_NPY)}")
    print(f"  {n_shots} shots x {n_t} samples @ {DT_MS:g} ms x {n_ch} "
          f"channels  ({fake.dtype})")
    print(f"  against {os.path.basename(resolve(DAS_NPY))}, reordered from "
          f"recording to sorted")
    if INVERT_ENV:
        lsp = C.LOG_SCALE_PARAMS[LOG_TAG]
        print(f"  undoing the {LOG_TAG} envelope gain: log_base "
              f"{lsp['log_base']:g}, sigma {lsp['smooth_sigma']}, "
              f"{ITERATIONS} iterations, {PAD_FRONT}-sample front pad")

    stop = n_shots if STOP is None else min(STOP, n_shots)
    indices = list(range(START, stop, STEP))
    if not indices:
        raise SystemExit("the START/STOP/STEP selection is empty")
    print(f"  plotting {len(indices)} shots: {indices[0]} .. {indices[-1]} "
          f"every {STEP}")

    for idx in indices:
        t0 = time.time()
        before = np.asarray(das[order[idx]], dtype=np.float64)
        after = np.asarray(fake[idx], dtype=np.float64)
        if INVERT_ENV:
            after = invert_env(after)

        clip = None
        if SHARED_CLIP:
            clip = max(clip_value(before, PERC), clip_value(after, PERC))

        # sharex/sharey ties the two panels together, so a zoom or a pan on
        # one moves the other and the same samples stay opposite each other
        fig, axes = plt.subplots(1, 2, figsize=(9, 7),
                                 sharex=True, sharey=True)
        v0 = draw(axes[0], before, clip)
        v1 = draw(axes[1], after, clip)
        axes[0].set_title(f"DAS, before the gain\nclip +-{v0:.4g}", fontsize=9)
        axes[1].set_title(f"translated, gain removed\nclip +-{v1:.4g}"
                          if INVERT_ENV else
                          f"translated, log envelope\nclip +-{v1:.4g}",
                          fontsize=9)
        for ax in axes:
            ax.set_xlabel("channel")
        axes[0].set_ylabel("time [s]")
        fig.suptitle(f"sorted shot {idx}  (recording {order[idx]})",
                     fontsize=11)
        fig.tight_layout()

        print(f"shot {idx:4d} (rec {order[idx]:4d}): "
              f"das [{before.min():.4g}, {before.max():.4g}]  "
              f"translated [{after.min():.4g}, {after.max():.4g}]  "
              f"{time.time() - t0:.1f}s")

        if SAVE_PATH:
            stem, ext = os.path.splitext(resolve(SAVE_PATH))
            path = f"{stem}_shot{idx:04d}{ext or '.png'}"
            fig.savefig(path, dpi=150)
            print(f"  saved {path}")
        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
