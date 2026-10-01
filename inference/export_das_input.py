"""The DAS input in the same layout as the translated files.

    das_data_norm.npy          (551, 2000, 264)  recording order, no gain
    das_data_rg_train.npy      ( 24, 2000, 551)  sorted, receiver-major, gain
    das_data_rg_infer.npy      (240, 2000, 551)
        ->  das_data_input_nl.npy  (2, 551, 2000, 264)
                channel 0  normal - das_data_norm, shots sorted
                channel 1  log    - the two rg arrays put back in receiver order

inference/translate_das.py writes (channel, shot, sample, receiver) with the
shots sorted along the line; this puts the input it translated on the same
axes, so a shot index, a receiver index and a channel mean the same thing in
both files and they can be read side by side without any reordering.

Channel 1 is the generator's own input, read from the arrays it was given -
as stored, before the dataset's clip and its division by `value_range`.
Channel 0 is that input before the envelope gain went on: das_data_norm, put
in the same sorted order.  The two are the same data - norm with the DAS gain
on is rg - and the run checks it on CHECK_SHOTS.

Edit the settings block, then run from the repository root:

    python -m inference.export_das_input
"""

import os
import time

import numpy as np

import config as C
import process.shot_geometry as G
from utils.data import create_memmap, get_project_root
from utils.process import calculate_logscale

# ---------------------------------------------------------------- settings --

OUT_NPY = C.ARRAYS["das"]["input_nl"]

# Shots on which channel 1 is recomputed from channel 0 with the DAS gain and
# compared, as a check that the two channels and the shot order agree.
CHECK_SHOTS = (0, 275, 550)

# Must match process/logenv_process.PAD_FRONT - the pad pins the gain's
# minimum, so a different one is a different gain.
GAIN_PAD_FRONT = 2000

# ---------------------------------------------------------------------------


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(get_project_root(), path)


def load(key):
    p = resolve(C.ARRAYS["das"][key])
    if not os.path.isfile(p):
        raise SystemExit(f"not found: {p}")
    return np.load(p, mmap_mode="r")


def shared_split(rec_xy):
    """Receiver indices inside / outside the streamer aperture, by position.

    The rule process/logenv_process.py split the rg arrays with.
    """
    s = np.load(resolve(C.META["str_line"]))
    a_rec = (rec_xy - s["line_centroid"]) @ s["line_direction"]
    a_str = (s["receiver_xy"] - s["line_centroid"]) @ s["line_direction"]
    lo, hi = a_str.min(), a_str.max()
    return (np.flatnonzero((a_rec >= lo) & (a_rec <= hi)),
            np.flatnonzero((a_rec < lo) | (a_rec > hi)))


def main():
    norm = load("norm")
    rg = [load("rg_train"), load("rg_infer")]
    n_shot, n_samp, n_recv = norm.shape
    rec = np.load(resolve(C.META["das_deci"]))["receiver_xy"]
    inside, outside = shared_split(rec)
    if (len(rec) != n_recv or len(inside) != rg[0].shape[0]
            or len(outside) != rg[1].shape[0]):
        raise SystemExit(f"norm has {n_recv} receivers, the split gives "
                         f"{len(inside)}/{len(outside)}, the rg arrays have "
                         f"{rg[0].shape[0]}/{rg[1].shape[0]}")
    if rg[0].shape[1:] != (n_samp, n_shot):
        raise SystemExit(f"rg_train is {rg[0].shape}, norm {norm.shape}")
    order = np.asarray(G.sorted_order()["orig_idx"], dtype=int)

    out_path = resolve(OUT_NPY)
    out = create_memmap(out_path, (2, n_shot, n_samp, n_recv))
    print(f"-> {OUT_NPY}  {out.shape}, {out.nbytes / 1e9:.2f} GB")

    # channel 1: receiver-major in, so one receiver block at a time and then
    # shot by shot out of memory, rather than 264 strided reads per shot
    t0 = time.time()
    log = np.empty((n_shot, n_samp, n_recv), dtype=np.float32)
    for a, where in zip(rg, (inside, outside)):
        block = np.asarray(a)                          # (n, n_samp, n_shot)
        log[:, :, where] = block.transpose(2, 1, 0)
    for i in range(n_shot):
        out[1, i] = log[i]
        out[0, i] = norm[order[i]]
    out.flush()
    print(f"  written in {time.time() - t0:.0f}s")

    lsp = dict(C.LOG_SCALE_PARAMS["das"])
    for i in CHECK_SHOTS:
        g = np.asarray(out[0, i], dtype=np.float64)
        pad = np.zeros((GAIN_PAD_FRONT, n_recv))
        sc = calculate_logscale(np.concatenate([pad, g]), **lsp)[GAIN_PAD_FRONT:]
        ref = np.asarray(out[1, i], dtype=np.float64)
        err = np.abs(g * sc - ref).max() / np.abs(ref).max()
        print(f"  shot {i}: normal with the DAS gain on vs log, max diff / "
              f"peak {err:.1e}")
    del out


if __name__ == "__main__":
    main()
