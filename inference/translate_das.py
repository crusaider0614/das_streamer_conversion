"""Run G_A2B over the whole DAS survey and write it as one array.

    das_data_rg_train.npy  (24, 2000, 551)   receivers 12..35
    das_data_rg_infer.npy  (240, 2000, 551)  the other 240
        ->  das_data_fake_str_<...>.npy  (2, 551, 2000, 264)
                channel 0  normal - the envelope gain off, band-passed, muted
                channel 1  log    - the same with the streamer gain put back

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

Post-processing
---------------
The generator's raw output is band-unlimited: it carries 20-30 dB more than
the streamer above 300 Hz, where neither input has anything, and a DC offset.
Both instruments were band-passed to 20-300 Hz before they were normalised,
so the target domain has nothing out there by construction.  The output is
therefore taken back through the input's own chain, in the input's order:

    1. the streamer envelope gain off - calculate_norscale_inversion
    2. the band-pass, FILTER_PARAMS[POST_FILTER]     (process/freq_filter.py)
    3. the mute, DAS boundary from the deci sidecar  (process/rms_normalize.py)
    4. the streamer envelope gain back on            (process/logenv_process.py)

The filter has to act in the normal domain.  The gain is a smooth multiplier
in time, which spreads the spectrum by a few tens of hertz - the streamer's
own log-domain spectrum rolls off between 300 and 380 Hz rather than at 300 -
so cutting in the log domain would also remove that legitimate spread.

The mute comes before the gain because that is the order the input was built
in: rms_normalize mutes `deci` into `norm`, and logenv_process gains `norm`,
which on disk already has the muted top at exact zero.  It matters even
though both are multiplications: the gain is estimated from the envelope, and
an envelope with the pre-arrival noise still in it gives a different gain.
No RMS step - the inverted output is already at the normalised scale.

Both ends are kept: channel 0 after step 3, channel 1 after step 4, so the
file has a channel axis in front.  POST_PROCESS False writes the raw 3-D
output as before.

Edit the settings block, then run from the repository root:

    python -m inference.translate_das
"""

import os
import time

import numpy as np
import torch
import yacs.config

import config as C
import process.shot_geometry as G
import utils.mute as MU
from module.dataset_pohang_shore import PohangShoreDataset
from network.pohang_shore_network_aniso import get_gen_model
from utils.data import create_memmap, get_project_root
from utils.process import (calculate_logscale, calculate_norscale_inversion,
                           f_filter, f_filtering)

# ---------------------------------------------------------------- settings --

CONFIG = os.path.join("config", "pohang_shore_das_str_cut.yaml")

# (checkpoint tag, epoch, key in C.ARRAYS["das"]), run in turn.  CONFIG's
# model section builds the generator for all of them; only the weights
# differ, and load_state_dict is strict, so a mismatch fails loudly.
JOBS = (
    ("pohang_shore_das_str_cut_iden", 150, "fake_str_150"),
    ("pohang_shore_das_str_cut_iden_3", 80, "fake_str_m_inp_080"),
    ("pohang_shore_das_str_cut_iden_3", 140, "fake_str_m_inp_140"),
)

# Take the output back through the input's chain and write both domains -
# see "Post-processing" in the header.  False writes the raw 3-D output.
POST_PROCESS = True
POST_LOG_TAG = "str"        # whose envelope gain the output carries
POST_FILTER = "str"         # key into C.FILTER_PARAMS; None skips the filter
POST_MUTE = True

# Iterations of the gain inversion.  Measured against a gather whose gain
# was applied and taken off again, the error relative to its peak is 6.1e-5
# after 20 and 3.6e-7 after 50; 20 is about 2 s a shot, 50 about 6.
INV_ITERATIONS = 20

# Zeros in front of each shot gather while the gain is fitted or inverted.
# Must match process/logenv_process.PAD_FRONT: the pad pins
# calculate_logscale's minimum, so a different one is a different gain.
GAIN_PAD_FRONT = 2000

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
REMUTE_FROM_INPUT = False

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


def padded(g, n):
    return np.concatenate([np.zeros((n, g.shape[1])), g], axis=0)


def invert_gain(g, lsp):
    """The envelope gain off one shot gather."""
    _, est = calculate_norscale_inversion(padded(g, GAIN_PAD_FRONT),
                                          iterations=INV_ITERATIONS, **lsp)
    return est[GAIN_PAD_FRONT:]


def apply_gain(g, lsp):
    """The envelope gain on, as process/logenv_process.logscale fits it."""
    scale = calculate_logscale(padded(g, GAIN_PAD_FRONT), **lsp)
    return g * scale[GAIN_PAD_FRONT:]


class BandPass:
    """process/freq_filter.py's mask and padding, for one gather length."""

    def __init__(self, params, n_samp, dt_s):
        self.pad = params["pad_front"]
        nt = self.pad + n_samp
        mask = np.ones(nt)
        if params["lowpass"]:
            mask *= f_filter(nt, dt_s, params["lp_f_cut"], params["lp_order"],
                             params["lp_decay"], is_lowpass=True)
        if params["highpass"]:
            mask *= f_filter(nt, dt_s, params["hp_f_cut"], params["hp_order"],
                             params["hp_decay"], is_lowpass=False)
        if params["zero_dc"]:
            mask[0] = 0.0
        self.mask = mask

    def __call__(self, g):
        return np.real(f_filtering(padded(g, self.pad), self.mask,
                                   is_zeroout=False))[self.pad:]


def mute_boundary(n_shot, n_recv):
    """(shot, receiver) mute boundary in ms, in the arrays' sorted order.

    The deci sidecar's, which is the boundary rms_normalize muted the input
    with; it is in recording order, the arrays here are sorted.
    """
    t = np.load(resolve(C.META["das_deci"]))["mute_boundary_ms"]
    if t.shape != (n_shot, n_recv):
        raise SystemExit(f"mute boundary is {t.shape}, expected "
                         f"({n_shot}, {n_recv})")
    order = np.asarray(G.sorted_order()["orig_idx"], dtype=int)
    return t[order]


def post_process(out, lsp, band, t_mute, dt_ms):
    """Channel 1 holds the raw output on entry; both channels on exit."""
    n_shot = out.shape[1]
    sample_major = OUT_ORDER[1] == "sample"
    t0 = time.time()
    for i in range(n_shot):
        g = np.asarray(out[1, i], dtype=np.float64)
        g = g if sample_major else g.T
        g = invert_gain(g, lsp)
        if band is not None:
            g = band(g)
        if t_mute is not None:
            g = g * MU.weights(t_mute[i], g.shape[0], dt_ms)
        log = apply_gain(g, lsp)
        out[0, i] = (g if sample_major else g.T).astype(np.float32)
        out[1, i] = (log if sample_major else log.T).astype(np.float32)
        if (i + 1) % 50 == 0 or i + 1 == n_shot:
            el = time.time() - t0
            print(f"    {i + 1}/{n_shot} shots  {el:.0f}s elapsed, "
                  f"eta {el / (i + 1) * (n_shot - i - 1):.0f}s", flush=True)


def load_generator(CF, tag, epoch):
    path = resolve(os.path.join("checkpoint", f"{tag}_{str(epoch).zfill(3)}"))
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
    print(f"{len(JOBS)} job(s)   device {DEVICE}")

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
    if POST_PROCESS:
        shape = (2,) + shape
        lsp = dict(C.LOG_SCALE_PARAMS[POST_LOG_TAG])
        band = (None if POST_FILTER is None else
                BandPass(C.FILTER_PARAMS[POST_FILTER], n_samp, dt_ms / 1000.0))
        t_mute = mute_boundary(n_shot, n_recv) if POST_MUTE else None
        print(f"  post: {POST_LOG_TAG} gain off ({INV_ITERATIONS} iterations,"
              f" {GAIN_PAD_FRONT} pad) -> "
              + (f"band-pass {POST_FILTER}" if band is not None
                 else "no filter")
              + (" -> mute" if t_mute is not None else "")
              + " -> gain on;  channels (normal, log)")

    for tag, epoch, key in JOBS:
        out_rel = C.ARRAYS["das"][key]
        print(f"\n{tag}, epoch {epoch}  ->  {out_rel}")
        out_path = resolve(out_rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out = create_memmap(out_path, shape)
        axes = (("channel",) if POST_PROCESS else ()) + tuple(OUT_ORDER)
        print(f"  output {out.shape} {axes}, {out.nbytes / 1e9:.2f} GB")
        # the raw output goes where the log channel will end up
        raw = out[1] if POST_PROCESS else out

        gen = load_generator(CF, tag, epoch)
        cut_ms = []
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
                    raw[:, :, r] = fake.T.astype(np.float32)
                else:
                    raw[:, r, :] = fake.T.astype(np.float32)
                done += 1
                if done % REPORT_EVERY == 0 or done == n_recv:
                    el = time.time() - t0
                    print(f"    {done}/{n_recv} receivers  {el:.0f}s elapsed, "
                          f"eta {el / done * (n_recv - done):.0f}s",
                          flush=True)
        del gen
        if DEVICE.startswith("cuda"):
            torch.cuda.empty_cache()
        if cut_ms:
            print(f"  remuted to a mean of {np.mean(cut_ms):.0f} ms, "
                  f"{np.min(cut_ms):.0f}..{np.max(cut_ms):.0f} across "
                  f"receivers")

        if POST_PROCESS:
            print("  post-processing shot gathers")
            post_process(out, lsp, band, t_mute, dt_ms)
        out.flush()
        del out
        print(f"  wrote {out_rel}")


if __name__ == "__main__":
    main()
