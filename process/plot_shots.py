"""Quick look at the shot gathers in a training .npy.

Loads an (n_shots, n_samples, n_channels) array from the project tree and draws
every STEP-th shot, tiled on one figure by default.  Meant for eyeballing the
output of `process/make_streamer_npy.py` before any bandpass or normalisation
has been applied, so each panel is clipped by percentile rather than shown on a
fixed scale.

The array is memory-mapped and only the plotted shots are read, so this stays
cheap on the multi-GB DAS array too.

Edit the settings block below, then run from the repository root:

    python -m process.plot_shots
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from utils.data import get_project_root

# ---------------------------------------------------------------- settings --

# Array to plot.  A relative path resolves against the project root, not the
# working directory, so this works the same however the script is launched.
NPY_PATH = os.path.join("data", "pohang_shore", "numpy", "str_data_raw.npy")

# Which shots: every STEP-th from START up to STOP (STOP = None -> to the end).
STEP = 10
START = 0
STOP = None

# Sample interval of the array, for the time axis in seconds.  str_data_raw.npy
# is on the SEG-Y's own 0.5 ms grid; set 1.0 once it has been resampled onto
# the DAS grid.
DT_MS = 0.5

# Colour clip: symmetric at this percentile of |amplitude|.
PERC = 99.0

# False -> clip each shot on its own, which shows the shape of every gather but
# hides how amplitude falls off with offset.  True -> one clip for all panels,
# so amplitudes are comparable across shots.
GLOBAL_CLIP = False

CMAP = "seismic"

# Tiled layout.  ONE_BY_ONE = True instead opens one full-size figure per shot.
NCOLS = 8
ONE_BY_ONE = False

# Annotate each panel with the source-to-array offset, read from the
# <NPY_PATH stem>_meta.npz written by make_streamer_npy.py.  Ignored if that
# file is not there.
SHOW_OFFSET = True

# Where to write the figure; None to only show it on screen.  In ONE_BY_ONE
# mode the shot index is appended to the file name.
SAVE_PATH = None

# ---------------------------------------------------------------------------


def resolve(path):
    """Interpret a relative path against the project root, not the cwd."""
    return path if os.path.isabs(path) else os.path.join(get_project_root(), path)


def clip_value(gather, perc):
    """Symmetric clip at the given percentile of |amplitude|."""
    v = float(np.percentile(np.abs(gather), perc))
    return v if v > 0 else float(np.abs(gather).max()) or 1.0


def load_offsets(npy_path, n_shots):
    """Source-to-array-centre offset per shot, from the companion _meta.npz.
    None if it is missing or does not line up with the array."""
    meta_path = os.path.splitext(npy_path)[0] + "_meta.npz"
    if not os.path.isfile(meta_path):
        return None
    with np.load(meta_path, allow_pickle=True) as m:
        if "source_xy" not in m or "receiver_xy" not in m:
            return None
        src = np.asarray(m["source_xy"], dtype=np.float64)
        rcv = np.asarray(m["receiver_xy"], dtype=np.float64)
    if len(src) != n_shots:
        print(f"  (ignoring {os.path.basename(meta_path)}: {len(src)} shots "
              f"there vs {n_shots} in the array)")
        return None
    return np.hypot(*(src - rcv.mean(axis=0)).T)


def draw(ax, gather, clip=None):
    n_t, n_ch = gather.shape
    v = clip if clip is not None else clip_value(gather, PERC)
    ax.imshow(gather, cmap=CMAP, vmin=-v, vmax=v, aspect="auto",
              interpolation="nearest",
              extent=[0, n_ch, n_t * DT_MS / 1000.0, 0])
    return v


def panel_title(idx, offsets):
    if offsets is None:
        return f"shot {idx}"
    return f"shot {idx}\n{offsets[idx]:.0f} m"


def plot_grid(data, indices, offsets, clip):
    nrows = int(np.ceil(len(indices) / NCOLS))
    fig, axes = plt.subplots(nrows, NCOLS, figsize=(1.5 * NCOLS, 3.2 * nrows),
                             squeeze=False)

    for ax, idx in zip(axes.ravel(), indices):
        draw(ax, np.asarray(data[idx]), clip)
        ax.set_title(panel_title(idx, offsets), fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xticks([])
    for ax in axes.ravel()[len(indices):]:
        ax.axis("off")

    for r in range(nrows):
        axes[r, 0].set_ylabel("time [s]", fontsize=8)

    scale = "shared clip" if clip is not None else f"{PERC:g}th-percentile clip"
    fig.suptitle(f"{len(indices)} shots, every {STEP} ({scale})", fontsize=11)
    fig.tight_layout()

    if SAVE_PATH:
        path = resolve(SAVE_PATH)
        fig.savefig(path, dpi=150)
        print(f"saved {path}")
    plt.show()


def plot_one_by_one(data, indices, offsets, clip):
    for idx in indices:
        gather = np.asarray(data[idx])
        fig, ax = plt.subplots(figsize=(4, 7))
        v = draw(ax, gather, clip)
        ax.set_title(panel_title(idx, offsets).replace("\n", "  -  "))
        ax.set_xlabel("channel")
        ax.set_ylabel("time [s]")
        fig.tight_layout()

        print(f"shot {idx:4d}: clip +-{v:.4g}, "
              f"range [{gather.min():.4g}, {gather.max():.4g}]")

        if SAVE_PATH:
            stem, ext = os.path.splitext(resolve(SAVE_PATH))
            path = f"{stem}_shot{idx:04d}{ext or '.png'}"
            fig.savefig(path, dpi=150)
            print(f"  saved {path}")
        plt.show()
        plt.close(fig)


def main():
    path = resolve(NPY_PATH)
    if not os.path.isfile(path):
        raise SystemExit(f"not found: {path}")

    data = np.load(path, mmap_mode="r")
    if data.ndim != 3:
        raise SystemExit(f"expected (shots, samples, channels), got {data.shape}")

    n_shots, n_t, n_ch = data.shape
    print(f"{path}")
    print(f"  {n_shots} shots x {n_t} samples @ {DT_MS:g} ms x {n_ch} channels"
          f"  ({data.dtype})")

    stop = n_shots if STOP is None else min(STOP, n_shots)
    indices = list(range(START, stop, STEP))
    if not indices:
        raise SystemExit("the START/STOP/STEP selection is empty")
    print(f"  plotting {len(indices)} shots: {indices[0]} .. {indices[-1]} "
          f"every {STEP}")

    offsets = load_offsets(path, n_shots) if SHOW_OFFSET else None

    clip = None
    if GLOBAL_CLIP:
        clip = float(np.percentile(np.abs(np.asarray(data[indices])), PERC))
        print(f"  shared clip +-{clip:.4g}")

    if ONE_BY_ONE:
        plot_one_by_one(data, indices, offsets, clip)
    else:
        plot_grid(data, indices, offsets, clip)


if __name__ == "__main__":
    main()
