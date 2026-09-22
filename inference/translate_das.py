"""Run G_A2B over the whole DAS survey and write it as one array.

    das_data_rg_train.npy  (24, 2000, 551)   receivers 12..35
    das_data_rg_infer.npy  (240, 2000, 551)  the other 240
        ->  das_data_fake_str.npy  (551, 2000, 264)

The generator takes one CROP-sized window, 2000 samples by 128 shots, which
is what it was trained on: the time axis is the whole record and must stay
that way - the FPA bottleneck opens with AdaptiveAvgPool2d(1) and every
DecodeBlock's CBAM pools globally, so a window of a different height is a
different problem to the model.  Only the shot axis is tiled.

How the windows are put back together
-------------------------------------
Two windows that overlap do not agree in the overlap.  They cannot: each one
was translated from a different context, and the global pooling means the
context reaches every output sample.  Butt-joining them therefore leaves a
visible seam at every boundary, so the overlap is blended.

The taper is a Hanning window evaluated half a grid point off,

    w[i] = 0.5 - 0.5 cos(2 pi (i + 0.5) / n)

rather than np.hanning's w[0] = w[n-1] = 0.  A Hanning that reaches exactly
zero gives the two end columns of every window no weight at all, and at the
two ends of the line - where only one window covers - that divides 0 by 0.
The half-grid form is strictly positive everywhere (1.5e-4 at the ends here),
so every output column has weight from at least one window, and the pair of
windows either side of a seam still sums smoothly.  The accumulated weights
are divided out at the end, which makes the blend a partition of unity
whatever the stride is.

The stride is the largest that is at most half a window, so that every column
inside the line is covered by two windows and the taper always has something
to blend with; and it is then spread evenly so the last window ends exactly
on the last shot rather than hanging off it.  With 551 shots and a window of
128 that is 8 windows at a stride of 60.4, which rounds to strides of 60 and 61
and an overlap of 67 columns - a little over half a window, as intended.

Amplitude
---------
The input is normalised the way the dataset does it: clipped at the training
percentile, then divided by `value_range`.  That divisor is taken from
rg_train for BOTH arrays.  Measured, rg_train's is 3.2716 and rg_infer's
3.1104, 5 % apart - taking each array's own would feed the generator two
differently scaled versions of the same instrument, and it never saw the
second one.  The output comes back through the streamer's own `value_range`
so the file sits on the same amplitude scale as str_data_rg_train.npy.

Edit the settings block, then run from the repository root:

    python -m inference.translate_das
"""

import os
import time

import numpy as np
import torch
import yacs.config

import config as C
from module.dataset_pohang_shore import PohangShoreDataset
from network.pohang_shore_network_aniso import get_gen_model
from utils.data import create_memmap, get_project_root

# ---------------------------------------------------------------- settings --

CONFIG = os.path.join("config", "pohang_shore_das_str_cut.yaml")
EPOCH = 150

# Where the result goes, relative to the project root.
OUT_NPY = os.path.join("data", "pohang_shore", "das_data_fake_str.npy")

# The window the generator sees, (samples, shots).  Match the training
# CROP_SIZE; the time entry has to be the full record.
CROP = (2000, 128)

# Largest stride allowed, in shots.  At most half the window, so every
# interior column falls in two windows.  The stride actually used is this or
# less - see `window_starts`.
MAX_STRIDE = 64

# Axis order of the output.  ("shot", "sample", "receiver") is what
# das_data_deci.npy and every other shot-major stage uses, which is what the
# SEG-Y writer and the preview scripts expect.  ("shot", "receiver",
# "sample") is also accepted if the file is wanted that way instead.
OUT_ORDER = ("shot", "sample", "receiver")

# Clip bounds and divisor for the input, and the multiplier for the output.
# None takes them from the datasets - rg_train's for the input, the
# streamer's for the output.  The clip is pinned to rg_train for the same
# reason the divisor is: they are two halves of one normalisation, and the
# model only ever saw rg_train's.  See the note on amplitude above.
IN_CLIP = None
IN_VALUE_RANGE = None
OUT_VALUE_RANGE = None

# The generator ends in tanh, so its output is never exactly zero and the
# muted top of the record comes back filled with low-level texture.  This
# puts the mute back, taken from the input: the streamer arrays are muted, so
# leaving it out would be the one obvious way to tell this file from a real
# one.
REMUTE_FROM_INPUT = True

# How far above the mute boundary the zeroing stops, in ms.  The boundary is
# not a hard edge - the mute carries a 30 ms raised-cosine taper - and only
# the samples that taper drove to exactly zero are reset here.  The pad pulls
# the reset back further still, so a translated arrival that has moved a
# little, and anything the generator put just under the boundary, survives.
# Nothing inside the taper is ever touched.
MUTE_PAD_MS = 20.0

# Windows per forward pass.  Each is 2000 x 128 floats.
BATCH = 4

# Which GPU.  "cuda" on its own means cuda:0, which on a shared box is
# whichever card came first and probably not the one this job was given - the
# training config names GPUS [4, 5, 6, 7] and the older test script hard-codes
# cuda:9.  So the index is written out.  None runs on the CPU, which works
# but is slow: the FID's Inception dominates everything there.
GPU = 0
DEVICE = (f"cuda:{GPU}" if GPU is not None and torch.cuda.is_available()
          else "cpu")

# Print a line every this many receivers.
REPORT_EVERY = 10

# ---------------------------------------------------------------------------


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(get_project_root(), path)


def window_starts(n_shot, nx, max_stride):
    """Evenly spaced starts, stride as large as possible but <= max_stride.

    Taking `ceil` of the number of intervals and then spreading them evenly
    is what keeps the last window flush with the last shot: a fixed stride of
    exactly max_stride would leave a remainder, and the final window would
    either overrun or need a different overlap from all the others.
    """
    span = n_shot - nx
    if span <= 0:
        return [0]
    n_int = int(np.ceil(span / max_stride))
    return list(np.linspace(0, span, n_int + 1).round().astype(int))


def leading_zeros(g):
    """Per trace, the length of the exact-zero run at the top of the record.

    That run is the mute and nothing else.  The taper below it never reaches
    zero, and measured on das_data_rg_train there is not one exact zero
    anywhere below the run, so this finds the hard-muted part exactly without
    reconstructing the boundary from the geometry.

    It runs 172..1347 samples with a median of 758 on the first receiver,
    which is why it is taken per trace rather than as one number.
    """
    nz = np.asarray(g) != 0.0
    return np.where(nz.any(axis=0), np.argmax(nz, axis=0), g.shape[0])


def remute(fake, gather, pad):
    """Zero what the input had hard-muted, stopping `pad` samples short.

    Only the exact-zero run is reset.  The partially muted samples - the
    taper - are left exactly as the generator produced them, and so is
    everything within `pad` of the boundary.  Returns the cut per trace.
    """
    cut = np.maximum(leading_zeros(gather) - pad, 0)
    rows = np.arange(fake.shape[0])[:, None]
    fake[rows < cut[None, :]] = 0.0
    return cut


def taper(n):
    """Hanning on the half-grid, so it never reaches zero.  See the header."""
    i = np.arange(n)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * (i + 0.5) / n)).astype(np.float64)


def shared_split(rec_xy):
    """Receiver indices inside / outside the streamer's aperture.

    The same rule as process/logenv_process.shared_receivers, by position
    rather than by index range, so this agrees with whatever split wrote the
    two arrays.
    """
    s = np.load(resolve(C.META["str_line"]))
    a_rec = (rec_xy - s["line_centroid"]) @ s["line_direction"]
    a_str = (s["receiver_xy"] - s["line_centroid"]) @ s["line_direction"]
    lo, hi = a_str.min(), a_str.max()
    return (np.flatnonzero((a_rec >= lo) & (a_rec <= hi)),
            np.flatnonzero((a_rec < lo) | (a_rec > hi)))


def load_generator(CF):
    path = resolve(os.path.join("checkpoint",
                                f"{CF.TAG}_{str(EPOCH).zfill(3)}"))
    if not os.path.isfile(path):
        raise SystemExit(f"not found: {path}")
    state = torch.load(path, map_location="cpu")
    g = get_gen_model(CF, True)
    g.load_state_dict(state["G_A2B"])
    print(f"  {os.path.basename(path)}  "
          f"GEN_CHANNELS {CF.MODEL.GEN_CHANNELS}")
    return g.to(DEVICE).eval()


def translate_gather(gen, gather, starts, w, in_range, out_range):
    """One receiver gather, (n_samp, n_shot) -> the same, translated.

    The windows are accumulated into a weighted sum and divided by the
    accumulated weight, which is a partition of unity by construction - so
    the blend is exact wherever one, two or more windows overlap, and no
    column is left out.
    """
    nt, nx = CROP
    n_samp, n_shot = gather.shape
    acc = np.zeros((n_samp, n_shot), dtype=np.float64)
    wsum = np.zeros(n_shot, dtype=np.float64)

    for i in range(0, len(starts), BATCH):
        chunk = starts[i:i + BATCH]
        batch = np.stack([gather[:, s:s + nx] for s in chunk])
        t = torch.from_numpy(batch / in_range)[:, None].float().to(DEVICE)
        with torch.no_grad():
            out = gen(t).cpu().numpy()[:, 0].astype(np.float64)
        for j, s in enumerate(chunk):
            acc[:, s:s + nx] += out[j] * w
            wsum[s:s + nx] += w

    if not np.all(wsum > 0):
        raise SystemExit(f"{int(np.sum(wsum <= 0))} shot columns got no "
                         f"window; starts {starts}")
    return acc / wsum * out_range


def main():
    with open(resolve(CONFIG), "rt") as f:
        CF = yacs.config.load_cfg(f)
    print(f"{CF.TAG}, epoch {EPOCH}   device {DEVICE}")

    A_train = PohangShoreDataset(is_das=True, crop_size=None, total_length=1,
                                 stage="rg_train")
    A_infer = PohangShoreDataset(is_das=True, crop_size=None, total_length=1,
                                 stage="rg_infer")
    B_train = PohangShoreDataset(is_das=False, crop_size=None, total_length=1,
                                 stage="rg_train")

    in_clip = ((A_train.lower_clip, A_train.upper_clip) if IN_CLIP is None
               else tuple(IN_CLIP))
    in_range = (A_train.value_range if IN_VALUE_RANGE is None
                else IN_VALUE_RANGE)
    out_range = (B_train.value_range if OUT_VALUE_RANGE is None
                 else OUT_VALUE_RANGE)
    print(f"  input  clip [{in_clip[0]:.4f}, {in_clip[1]:.4f}] /{in_range:.4f}"
          f"  (rg_train; rg_infer's own divisor would be "
          f"{A_infer.value_range:.4f})")
    print(f"  output x{out_range:.4f} (streamer)")

    rec = np.load(resolve(C.META["das_deci"]))["receiver_xy"]
    inside, outside = shared_split(rec)
    n_recv = len(rec)
    if len(inside) != A_train.n_data or len(outside) != A_infer.n_data:
        raise SystemExit(f"split gives {len(inside)}/{len(outside)}, the "
                         f"arrays have {A_train.n_data}/{A_infer.n_data}")

    n_samp, n_shot = A_train.n_samp, A_train.n_shot
    if n_samp != CROP[0]:
        raise SystemExit(f"the arrays are {n_samp} samples, CROP is {CROP[0]}"
                         f" - the time axis is not tiled, so they must match")
    dt_ms = C.STAGE_DT_US["das"]["norm"] / 1000.0
    pad = int(round(MUTE_PAD_MS / dt_ms))
    cut_ms = []
    if REMUTE_FROM_INPUT:
        print(f"  remute: the input's exact-zero run less a {MUTE_PAD_MS:g} "
              f"ms pad ({pad} samples); the taper is left alone")
    starts = window_starts(n_shot, CROP[1], MAX_STRIDE)
    step = np.diff(starts)
    print(f"  {len(starts)} windows of {CROP[0]}x{CROP[1]} per receiver, "
          f"stride {step.min()}-{step.max()} (<= {MAX_STRIDE}), overlap "
          f"{CROP[1] - step.max()}")
    print(f"  starts {starts}")
    w = taper(CROP[1])

    shape = {("shot", "sample", "receiver"): (n_shot, n_samp, n_recv),
             ("shot", "receiver", "sample"): (n_shot, n_recv, n_samp)}
    if tuple(OUT_ORDER) not in shape:
        raise SystemExit(f"OUT_ORDER {OUT_ORDER} is not one of "
                         f"{list(shape)}")
    shape = shape[tuple(OUT_ORDER)]
    out_path = resolve(OUT_NPY)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out = create_memmap(out_path, shape)
    print(f"  output {out.shape} {OUT_ORDER}, {out.nbytes / 1e9:.2f} GB")

    gen = load_generator(CF)

    t0 = time.time()
    done = 0
    for ds, where in ((A_train, inside), (A_infer, outside)):
        for k, r in enumerate(where):
            g = np.array(ds.data[k], dtype=np.float32)
            np.clip(g, in_clip[0], in_clip[1], out=g)
            fake = translate_gather(gen, g, starts, w, in_range, out_range)
            if REMUTE_FROM_INPUT:
                cut_ms.append(remute(fake, g, pad).mean() * dt_ms)
            # (n_samp, n_shot) -> the output's own axes
            if OUT_ORDER[1] == "sample":
                out[:, :, r] = fake.T.astype(np.float32)
            else:
                out[:, r, :] = fake.T.astype(np.float32)
            done += 1
            if done % REPORT_EVERY == 0 or done == n_recv:
                el = time.time() - t0
                print(f"    {done}/{n_recv} receivers  {el:.0f}s elapsed, "
                      f"eta {el / done * (n_recv - done):.0f}s", flush=True)

    out.flush()
    del out
    if cut_ms:
        print(f"  remuted to a mean of {np.mean(cut_ms):.0f} ms, "
              f"{np.min(cut_ms):.0f}..{np.max(cut_ms):.0f} across receivers")
    print(f"  wrote {OUT_NPY}")


if __name__ == "__main__":
    main()
