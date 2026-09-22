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

import config as C
from utils.data import get_project_root

# ---------------------------------------------------------------- settings --

# Array to plot.  A relative path resolves against the project root, not the
# working directory, so this works the same however the script is launched.
# NPY_PATH = os.path.join("data", "pohang_shore", "numpy", "str_data_raw.npy")
NPY_PATH = os.path.join("data", "pohang_shore", "das_data_fake_str.npy")

# Which shots: every STEP-th from START up to STOP (STOP = None -> to the end).
STEP = 10
START = 0
STOP = None

# Sample interval of the array, for the time axis in seconds.  1.0 for
# anything from `line` onwards, including das_data_fake_str.npy; 0.5 only for
# str_data_raw.npy, which is still on the SEG-Y's own grid.
DT_MS = 1.0

# Colour clip: symmetric at this percentile of |amplitude|.
PERC = 99.0

# False -> clip each shot on its own, which shows the shape of every gather but
# hides how amplitude falls off with offset.  True -> one clip for all panels,
# so amplitudes are comparable across shots.
GLOBAL_CLIP = False

CMAP = "seismic"

# One full-size figure per shot, shown and closed one at a time.  False tiles
# them NCOLS wide on a single figure instead, which is the cheaper way to scan
# a whole survey but too small to judge a waveform on.
ONE_BY_ONE = True
NCOLS = 8

# Draw the generator's input beside its output, on shared axes, so a zoom or a
# pan on one panel moves the other with it and the same samples stay side by
# side.  ONE_BY_ONE only - two panels per shot do not tile usefully.
#
#   "rg"    the arrays the generator was actually given,
#           das_data_rg_{train,infer}.npy, put back into receiver order.  This
#           is the right comparison for das_data_fake_str.npy: same log
#           envelope gain, same shot order, same receivers, so the only
#           difference on screen is the translation.
#   a path  any other shot-major (shots, samples, channels) array.  Watch the
#           shot order - everything from `deci` and `norm` is in RECORDING
#           order while the rg arrays and the translated file are SORTED along
#           the line, so shot i is not the same shot in both.
#   None    no second panel.
COMPARE = "rg"

# Each panel is clipped on its own even when GLOBAL_CLIP is set, because the
# two sides do not share a scale: the translated file comes back through the
# streamer's value_range, the input never left the DAS's.
COMPARE_PERC = PERC

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


def shared_split(rec_xy):
    """Receiver indices inside / outside the streamer's aperture.

    The same rule as process/logenv_process.shared_receivers, by position
    rather than by index range, so this agrees with whatever split wrote the
    two arrays.
    """
    m = np.load(resolve(C.META["str_line"]))
    a_rec = (rec_xy - m["line_centroid"]) @ m["line_direction"]
    a_str = (m["receiver_xy"] - m["line_centroid"]) @ m["line_direction"]
    lo, hi = a_str.min(), a_str.max()
    return (np.flatnonzero((a_rec >= lo) & (a_rec <= hi)),
            np.flatnonzero((a_rec < lo) | (a_rec > hi)))


class RGShots:
    """The generator's input, read back as shot gathers.

    das_data_rg_{train,infer}.npy are receiver-major - (receiver, sample,
    shot) - and split by whether the receiver has a streamer counterpart, so
    one shot gather is a strided read from each of them, put back in receiver
    order.  Indexed [shot] like a shot-major array, which is all the plotting
    needs.
    """

    def __init__(self):
        paths = [resolve(C.ARRAYS["das"][k]) for k in ("rg_train", "rg_infer")]
        missing = [q for q in paths if not os.path.isfile(q)]
        if missing:
            raise SystemExit("not found: " + ", ".join(missing))
        self.arrays = [np.load(q, mmap_mode="r") for q in paths]
        rec = np.load(resolve(C.META["das_deci"]))["receiver_xy"]
        self.inside, self.outside = shared_split(rec)
        self.n_recv = len(rec)
        if (len(self.inside) != self.arrays[0].shape[0]
                or len(self.outside) != self.arrays[1].shape[0]):
            raise SystemExit(
                f"split gives {len(self.inside)}/{len(self.outside)}, the "
                f"arrays have {self.arrays[0].shape[0]}/"
                f"{self.arrays[1].shape[0]}")
        self.shape = (self.arrays[0].shape[2], self.arrays[0].shape[1],
                      self.n_recv)

    def __getitem__(self, shot):
        out = np.empty(self.shape[1:], dtype=np.float32)
        for a, where in zip(self.arrays, (self.inside, self.outside)):
            out[:, where] = np.asarray(a[:, :, shot]).T
        return out


def open_compare(n_shots):
    """The second panel's source, or None."""
    if not COMPARE:
        return None, ""
    if COMPARE == "rg":
        rg = RGShots()
        if rg.shape[0] != n_shots:
            raise SystemExit(f"the rg arrays hold {rg.shape[0]} shots, the "
                             f"array being plotted has {n_shots}")
        return rg, "input (rg, log envelope)"
    q = resolve(COMPARE)
    if not os.path.isfile(q):
        raise SystemExit(f"not found: {q}")
    a = np.load(q, mmap_mode="r")
    if a.shape[0] != n_shots:
        raise SystemExit(f"{os.path.basename(q)} holds {a.shape[0]} shots, "
                         f"the array being plotted has {n_shots}")
    return a, os.path.basename(q)


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


def plot_one_by_one(data, indices, offsets, clip, other=None, other_name=""):
    for idx in indices:
        gather = np.asarray(data[idx])
        panels = [(gather, os.path.basename(resolve(NPY_PATH)), clip)]
        if other is not None:
            panels.insert(0, (np.asarray(other[idx]), other_name, None))

        # sharex/sharey is what makes a zoom or a pan on one panel move the
        # other: matplotlib keeps the limits tied, so the same samples stay
        # opposite each other however far in you go.
        fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 7),
                                 sharex=True, sharey=True, squeeze=False)
        line = f"shot {idx:4d}:"
        for ax, (img, name, cl) in zip(axes[0], panels):
            v = draw(ax, img, cl)
            ax.set_title(name, fontsize=9)
            ax.set_xlabel("channel")
            line += (f"  {name} clip +-{v:.4g} "
                     f"[{img.min():.4g}, {img.max():.4g}]")
        axes[0, 0].set_ylabel("time [s]")
        fig.suptitle(panel_title(idx, offsets).replace("\n", "  -  "),
                     fontsize=11)
        fig.tight_layout()
        print(line)

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
        other, other_name = open_compare(n_shots)
        if other is not None:
            print(f"  beside {other_name}")
        plot_one_by_one(data, indices, offsets, clip, other, other_name)
    else:
        if COMPARE:
            print("  (COMPARE is ignored unless ONE_BY_ONE)")
        plot_grid(data, indices, offsets, clip)


if __name__ == "__main__":
    main()
