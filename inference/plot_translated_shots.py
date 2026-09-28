"""Shot gathers before and after translation, in either amplitude domain.

    left   das_data_rg_{train,infer}.npy   the generator's own input
    right  das_data_fake_str.npy           its output

Copied from process/plot_shots.py.  The difference is what each panel holds
and which amplitude domain they are shown in.

The input side
--------------
Assembled from the two receiver-major arrays the generator was actually given
and put back into receiver order, one strided read per shot, rather than read
from a shot-major file.  das_data_norm.npy would be the wrong thing to
compare against twice over: it is in recording order where the translated
file is sorted along the line, so shot i is not the same shot, and it never
had the envelope gain applied, so it is not what the model saw.

DOMAIN
------
    "log"     both panels exactly as they are stored, in the log-envelope
              domain the generator works in.  Nothing is estimated, so this
              is the honest view of what the model produced.
    "normal"  the envelope gain taken off both sides, which is the domain the
              data really has.  The gain flattens the amplitude decay, so
              reading a translation in "log" hides the amplitude behaviour
              the translation is supposed to change.

The gain is undone with `utils.process.calculate_norscale_inversion`, which
re-estimates the scale from the scaled data and divides it back out.  Each
side is inverted with ITS OWN parameters - the input carries the DAS gain,
the output the streamer's, since translate_das.py brings the generator's
output back through the streamer's `value_range` and the generator was
trained against streamer gathers.  Both then sit at the TARGET_RMS that
rms_normalize.py set before the gain, so the panels can be read against each
other.

The front pad matters.  process/logenv_process.py computed the forward gain
on a gather with PAD_FRONT zeros in front of it, which pins
`calculate_logscale`'s `data_env_log.min()` to log10(log_base) whatever the
gather contains; without it that minimum floats and the estimate comes back
systematically different.  So the same pad goes on here and is cropped off
afterwards.

Shot order
----------
Both sides are in the sorted order the rg arrays and the translated file
share, so shot i is the same shot in both.  The title also gives the
recording index.

Edit the settings block below, then run from the repository root:

    python -m inference.plot_translated_shots
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np

import config as C
import process.shot_geometry as G
from process.plot_shots import RGShots
from utils.data import get_project_root
from utils.process import calculate_norscale_inversion

# ---------------------------------------------------------------- settings --

# The translated file.  A relative path resolves against the project root,
# not the working directory.  The input side is assembled from
# C.ARRAYS["das"]["rg_train"] and ["rg_infer"] - see the header.
FAKE_NPY = C.ARRAYS["das"]["fake_str"]

# Which shots: every STEP-th from START up to STOP (STOP = None -> the end).
# Indices are into the TRANSLATED file, so they are sorted-order indices.
STEP = 50
START = 0
STOP = None

# "log" draws both sides as stored; "normal" takes the envelope gain off
# both.  See the header.
DOMAIN = "normal"

# Whose gain each side carries.  The input is DAS, the output is the streamer
# it was translated into.
IN_LOG_TAG = "das"
OUT_LOG_TAG = "str"

# 50 is calculate_norscale_inversion's own default and costs 5.7 s for a
# 264-trace gather, so a two-panel figure is about twelve seconds.  Measured
# against a gather whose gain was applied and then taken back off, the error
# relative to the gather's peak is 2.3e-3 mean after 5 iterations, 6.1e-5
# after 20 and 3.6e-7 after 50, so 20 is plenty if the wait is annoying.
ITERATIONS = 50

# Zeros in front of the gather while the gain is estimated, in samples.  Must
# match process/logenv_process.PAD_FRONT - see the header.
PAD_FRONT = 2000

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


def invert_env(gather, tag):
    """Take the `tag` envelope gain back off one shot gather.

    The pad is zeros, so it survives the division unchanged and keeps
    `data_env_log.min()` pinned where the forward pass had it; it is cropped
    off before the result is returned.
    """
    lsp = dict(C.LOG_SCALE_PARAMS[tag])
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
    if DOMAIN not in ("log", "normal"):
        raise SystemExit(f"DOMAIN {DOMAIN!r} is not 'log' or 'normal'")

    fake = open_array(FAKE_NPY, "translated")
    das = RGShots()                     # the generator's own input, assembled
    if tuple(fake.shape) != tuple(das.shape):
        raise SystemExit(f"translated {fake.shape} against the rg arrays "
                         f"{das.shape}; they have to be the same grid")

    n_shots, n_t, n_ch = fake.shape
    order = np.asarray(G.sorted_order()["orig_idx"], dtype=int)
    if len(order) != n_shots:
        raise SystemExit(f"shot_geometry has {len(order)} shots, the arrays "
                         f"have {n_shots}")

    print(f"{resolve(FAKE_NPY)}")
    print(f"  {n_shots} shots x {n_t} samples @ {DT_MS:g} ms x {n_ch} "
          f"channels  ({fake.dtype})")
    print(f"  against rg_train + rg_infer, {len(das.inside)} + "
          f"{len(das.outside)} receivers put back in order; both sides are "
          f"in sorted shot order")
    if DOMAIN == "normal":
        for tag in (IN_LOG_TAG, OUT_LOG_TAG):
            lsp = C.LOG_SCALE_PARAMS[tag]
            print(f"  undoing the {tag} envelope gain: log_base "
                  f"{lsp['log_base']:g}, sigma {lsp['smooth_sigma']}, "
                  f"{ITERATIONS} iterations, {PAD_FRONT}-sample front pad")
    else:
        print(f"  log-envelope domain, both sides as stored")

    stop = n_shots if STOP is None else min(STOP, n_shots)
    indices = list(range(START, stop, STEP))
    if not indices:
        raise SystemExit("the START/STOP/STEP selection is empty")
    print(f"  plotting {len(indices)} shots: {indices[0]} .. {indices[-1]} "
          f"every {STEP}")

    titles = (f"input DAS, {DOMAIN}", f"translated, {DOMAIN}")
    for idx in indices:
        t0 = time.time()
        before = np.asarray(das[idx], dtype=np.float64)
        after = np.asarray(fake[idx], dtype=np.float64)
        if DOMAIN == "normal":
            before = invert_env(before, IN_LOG_TAG)
            after = invert_env(after, OUT_LOG_TAG)

        clip = None
        if SHARED_CLIP:
            clip = max(clip_value(before, PERC), clip_value(after, PERC))

        # sharex/sharey ties the two panels together, so a zoom or a pan on
        # one moves the other and the same samples stay opposite each other
        fig, axes = plt.subplots(1, 2, figsize=(9, 7),
                                 sharex=True, sharey=True)
        for ax, img, name in zip(axes, (before, after), titles):
            v = draw(ax, img, clip)
            ax.set_title(f"{name}\nclip +-{v:.4g}", fontsize=9)
            ax.set_xlabel("channel")
        axes[0].set_ylabel("time [s]")
        fig.suptitle(f"sorted shot {idx}  (recording {order[idx]})",
                     fontsize=11)
        fig.tight_layout()

        print(f"shot {idx:4d} (rec {order[idx]:4d}): "
              f"input [{before.min():.4g}, {before.max():.4g}]  "
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
