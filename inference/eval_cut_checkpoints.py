"""Score every saved CUT checkpoint, so the metrics can be read against epoch.

    checkpoint/<TAG>_010, _020, ...  ->  a table, and metrics_<TAG>.csv

Five metrics, chosen because the usual ones do not apply here.  The two
instruments are co-located and share the shot geometry, so the data IS paired -
but the waveform correlation between matched traces is 0.015.  Anything that
compares samples to the streamer as a target (PSNR, MSE, SSIM) therefore scores
a faithful translation and a lazy one about the same.  What follows compares
distributions and positions instead.

  0  Frechet distance between the patches of G(A) and the patches of real B.
     A distribution distance, so it does not need the pairing at all.  Two
     of them, over the same tiles, because they fail in different ways:

     0a  FD, over 30 hand-designed seismic features - 24 log band powers
         across 20-300 Hz and 6 log envelope quantiles.  This is the one to
         steer by.  Every dimension means something, so a change can be
         traced to the dimension that moved, and a 30 x 30 covariance is
         properly determined by a thousand patches.
             das vs str  25.1        floors  das 0.13,  str 0.24
         measured on 209 tiles of 6 rg_train receivers with this build.
         It is also gameable: phase-randomised streamer against streamer
         scores 0.03, i.e. at the floor, because none of the 30 features
         sees phase.  Read it with metric 5 and the envelope correlation,
         never alone.  And it moves with the sample count like the FID does
         - 35.0 at n=100, 25.7 at n=200, 25.1 at n=209 - so the equal-n rule
         below covers both.
     0b  FID, the usual InceptionV3 one, which is what the wider GAN
         literature reports.  The caveats are in module/fid.py: Inception
         has never seen seismic and the 2048 x 2048 covariance is estimated
         from a few thousand patches, so the value is biased and only
         comparable against the A and str rows of the same run, never
         against a published image FID.

  1  envelope correlation, corr(env(G(A)), env(B)) on the matched receiver
     Phase-insensitive, so the waveform change the translation is supposed to
     make does not penalise it, and it uses the pairing that does exist.
     Measured on whole traces: over a short window the matched and mismatched
     cases are indistinguishable (0.148 against 0.100), because what carries
     the pairing is the long-range trend.
         before translation  0.44        shuffled-shot control  0.02
     Needs a streamer counterpart, so it only exists for STAGE = "rg_train".

  2  instantaneous frequency, median over the envelope's top quartile
     The clearest measured difference between the instruments, and the thing
     the user wants the translation to move.
         das  85.4 Hz        str  112.1 Hz

  3  spectral centroid, median over traces
     The same story as 2 for a tenth of the cost, so it is the one to watch
     per epoch.
         das  98.3 Hz        str  132.1 Hz

  5  local 2-D envelope lag between G(A) and its own input A
     Did events move?  Envelopes rather than waveforms, so a legitimate change
     of waveform leaves the lag alone and only a real displacement shows.
     Validated by shifting a gather a known (7 ms, 3 traces): recovered exactly,
     peak correlation 0.93.  Both axes are reported - there is no mechanism
     that should move anything sideways, so dx is the control.
         target  (0, 0)

The table opens with two rows that are not checkpoints:

    A      the input, put through the identical code path with the generator
           replaced by the identity - so every metric, not only the three
           that used to have a "before" printed, is measured on the same
           windows by the same estimator.  Metric 5 is then A against A and
           must come out (0, 0) at peak 1.000; if it does not, the lag
           machinery is broken and the G(A) rows mean nothing.
    str    the target.  Metrics 1 and 5 are undefined against itself, and
           its FD and FID are the floor - the streamer shuffled and split in
           half.

Every G(A) row is read between those two.

Metric 4, shot-axis semblance, was dropped: both domains sit at the incoherent
floor (das 0.1087, str 0.1018, floor 1/9 = 0.1111) because the sorted shot axis
interleaves four passes at an uneven 3.33 m median spacing.  It cannot measure
a change in noise here.

The two offset bins
-------------------
Every metric above is offset dependent - instantaneous frequency alone runs
83.8 -> 138.4 Hz across the offset range - so each is reported twice, split at
OFFSET_SPLIT_M.  The split is not a round number: 184.1 m is the smallest
source-receiver distance anywhere in the streamer data, and it is smallest
because the streamer sits at 1240-1312 m along the line while the shots run
-1059..+1056 m.  The array is off the end of the sail line, so no shot ever
passes over it.

    near   < 184.1 m     DAS only.  There is no streamer trace at this offset
                         anywhere in the survey, so the near bin has no
                         reference of its own: FID compares it against the
                         streamer's far patches and metric 1 does not exist.
                         Read it as an extrapolation check - the model was
                         never shown what a streamer looks like here - not as
                         a score.  With 87 m of water this is also the only
                         bin that contains the moveout apex.
    far   >= 184.1 m     The band both instruments cover, filled continuously
                         (largest gap between consecutive traces 3.18 m) up to
                         2371 m.

Which bin holds what, by trace:

                    near        far
    das all 264    10.90 %    89.10 %
    das rg_train    0.00 %   100.00 %
    das rg_infer   11.99 %    88.01 %
    str               0        100 %

So STAGE = "rg_train" puts every trace in the far bin and the split does
nothing: those 24 receivers sit inside the streamer array and inherit its
geometry exactly.  The near bin only appears at STAGE = "rg_infer", which is
also where metric 1 disappears.  Run it both ways.

Two things worth knowing before reading the numbers
---------------------------------------------------
The windows are CROP_SIZE wide on purpose.  The generator's FPA bottleneck
begins with AdaptiveAvgPool2d(1) and every DecodeBlock's CBAM pools globally,
so the model sees statistics over whatever extent it is given.  Scoring on the
full 551-shot gather would hand it an extent it never saw in training.

And there is no held-out set: training uses all 24 receivers of `rg_train`.
These numbers are therefore in-sample.  EVAL_RECEIVERS exists so a few can be
held out of the next training run and scored here instead.

Edit the settings block below, then run from the repository root:

    python -m inference.eval_cut_checkpoints
"""

import os
import re

import numpy as np
import torch

import config as C
import process.shot_geometry as G
from module.dataset_pohang_shore import PohangShoreDataset
from module.fid import (InceptionNetwork, calculate_activation_statistics,
                        calculate_frechet_distance)
from network.pohang_shore_network_aniso import get_gen_model
from utils.data import get_project_root
from utils.process import envelope_1d

# ---------------------------------------------------------------- settings --

# Checkpoint tag, and which epochs to score.  None scans the directory for
# every `<TAG>_<digits>` it holds and sorts them.
TAG = "pohang_shore_das_str_cut_iden"
EPOCHS = None

# Which DAS array to score.  "rg_train" is the 24 receivers the streamer
# overlaps - the only ones metric 1 and the FID reference exist for, and every
# trace of them is in the far bin.  "rg_infer" is the other 240, which is
# where the near bin lives.
STAGE = "rg_train"

# The window each metric is measured on.  Match the training CROP_SIZE - see
# the note about global pooling above.
CROP = (2000, 128)

# Receivers to score, indices into STAGE's array.  None uses all of them.
EVAL_RECEIVERS = None

# Shot windows per receiver, evenly spaced across the 551 sorted shots.
N_WINDOWS = 4

# The offset the bins split at, in metres.  184.1 is the streamer's own
# minimum offset - see the header.
OFFSET_SPLIT_M = 184.1
BINS = ("near", "far")

# Metric 5: the sub-window the local lag is measured on, and how far the
# search looks.  +-16 ms and +-4 shots is generous for a displacement that
# should be zero.
LAG_WINDOW = (256, 64)
LAG_SEARCH = (16, 4)

# Metric 2: instantaneous frequency is only defined where there is amplitude,
# so it is taken over the envelope's top quartile.
IF_QUANTILE = 0.75
IF_RANGE = (5.0, 450.0)

DT_S = 1.0e-3

# Metric 0.  The CROP windows are cut into tiles of this size, because
# Inception takes a roughly square image and a 2000 x 128 gather resized to
# 299 x 299 would be flattened 16:1 along time.  A tile that is mostly mute
# carries no information and would only pull every set towards the same
# all-zero mode, so FID_MIN_LIVE drops those.  Both 0a and 0b read the same
# tiles.
USE_FD = True
USE_FID = True
FID_PATCH = (128, 128)
FID_MIN_LIVE = 0.7

# Metric 0a.  The feature vector: log power in FD_BANDS equal bands across
# FD_BAND_HZ, then the log envelope at FD_PCTL.  30 numbers, so a 30 x 30
# covariance from a thousand patches is comfortably determined - which is the
# whole reason this exists next to the FID.
FD_BANDS = 24
FD_BAND_HZ = (20.0, 300.0)
FD_PCTL = (25, 50, 75, 90, 95, 99)
FD_FLOOR = 1e-12

# Every set - G(A), real B, real A - is subsampled to exactly this many
# patches.  The FID of a rank-deficient covariance grows as the sample count
# falls, so two FIDs are only comparable at equal n.  None uses whatever the
# smallest set has, which is still equal across sets.
FID_MAX_SAMPLES = 1024
FID_BATCH = 32
FID_SEED = 71138602

# Write metrics_<TAG>.csv next to the checkpoints.  A falsy value skips it.
CSV_DIR = "checkpoint"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(get_project_root(), path)


def find_checkpoints():
    d = resolve("checkpoint")
    if EPOCHS is not None:
        return [(e, os.path.join(d, f"{TAG}_{e:03d}")) for e in EPOCHS]
    pat = re.compile(re.escape(TAG) + r"_(\d+)$")
    out = []
    for name in os.listdir(d):
        m = pat.match(name)
        if m:
            out.append((int(m.group(1)), os.path.join(d, name)))
    return sorted(out)


def gen_channels(state):
    """Read GEN_CHANNELS off the checkpoint rather than trusting the config.

    The config in the repository and the one a run was launched with drift
    apart; the stem conv's output width is the truth.
    """
    for k in ("conv.0.weight", "conv.0.conv.parametrizations.weight.original"):
        if k in state:
            return int(state[k].shape[0])
    raise SystemExit(f"cannot read GEN_CHANNELS from the checkpoint; keys "
                     f"start with {sorted(state)[:4]}")


def build_generator(state):
    from yacs.config import CfgNode
    ch = gen_channels(state)
    CF = CfgNode({"MODEL": CfgNode({
        "A_CHANNELS": 1, "B_CHANNELS": 1, "GEN_CHANNELS": ch,
        "GEN_NORM": True, "GEN_SPEC": False, "GEN_CBAM": True})})
    g = get_gen_model(CF, True)
    g.load_state_dict(state)
    return g.to(DEVICE).eval(), ch


# ---------------------------------------------------------------- geometry --


def shared_split(rec_xy):
    """Receiver indices inside / outside the streamer's aperture.

    The same rule as process/logenv_process.shared_receivers - by position,
    not by index range - so the two agree if the decimation stride changes.
    """
    s = np.load(resolve(C.META["str_line"]))
    a_rec = (rec_xy - s["line_centroid"]) @ s["line_direction"]
    a_str = (s["receiver_xy"] - s["line_centroid"]) @ s["line_direction"]
    lo, hi = a_str.min(), a_str.max()
    inside = np.flatnonzero((a_rec >= lo) & (a_rec <= hi))
    outside = np.flatnonzero((a_rec < lo) | (a_rec > hi))
    return inside, outside


def offsets(stage):
    """(n_receiver, n_shot) offsets on the same axes as the rg arrays.

    Two reorderings have to be undone.  The sidecars are written at the
    `line` / `deci` stage, which is shot-major and in RECORDING order, while
    logenv_process sorts the shots along the line and transposes to
    receiver-major; and its sort comes from process.shot_geometry, on the DAS
    source positions, for both domains - so the streamer offsets are put in
    that same order rather than in the one str_data_line_meta carries.
    """
    order = np.asarray(G.sorted_order()["orig_idx"], dtype=int)
    if stage == "str":
        off = np.load(resolve(C.META["str_line"]))["offset_projected_m"]
        return off[order].T.astype(np.float64)

    z = np.load(resolve(C.META["das_deci"]))
    off = z["offset_m"][order].T.astype(np.float64)
    inside, outside = shared_split(z["receiver_xy"])
    keep = inside if stage == "rg_train" else outside
    return off[keep]


def which_bin(offset):
    """0 near, 1 far.  Array in, array out."""
    return (np.asarray(offset) >= OFFSET_SPLIT_M).astype(int)


# ------------------------------------------------------------- the windows --


def take(ds, k, t0, x0, nt, nx):
    """One window, normalised exactly as the dataset would."""
    p = np.array(ds.data[k, t0:t0 + nt, x0:x0 + nx], dtype=np.float32)
    np.clip(p, ds.lower_clip, ds.upper_clip, out=p)
    return p / ds.value_range


def window_starts(n_shot, nx):
    if N_WINDOWS <= 1:
        return [0]
    return list(np.linspace(0, n_shot - nx, N_WINDOWS).round().astype(int))


# ----------------------------------------------------------------- metrics --


def env(x):
    return envelope_1d(np.ascontiguousarray(x, dtype=np.float32)).astype(np.float64)


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a * b).sum() / d) if d > 0 else 0.0


def envelope_corr(x, y):
    """Metric 1, one value per trace - a whole-gather correlation would be
    dominated by the loud traces, and a per-trace value is what the offset
    split needs.  nan where either side is dead."""
    ex, ey = env(x), env(y)
    out = np.full(x.shape[1], np.nan)
    live = (np.abs(x).max(0) > 0) & (np.abs(y).max(0) > 0)
    for j in np.flatnonzero(live):
        out[j] = corr(ex[:, j], ey[:, j])
    return out


def analytic(x):
    """Same convention as utils.process.hilbert_1d: along time, per trace."""
    n = x.shape[0]
    h = np.zeros(n)
    h[0] = 1.0
    if n % 2 == 0:
        h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[1:(n + 1) // 2] = 2.0
    return np.fft.ifft(np.fft.fft(x, axis=0) * h[:, None], axis=0)


def inst_freq(x):
    """Metric 2, one median per trace.

    Phase is only meaningful where the envelope is, so the weakest three
    quarters of each trace are dropped.  Per trace rather than pooled over
    the window: the offset bin is a property of the trace, and it also stops
    the loudest traces contributing more samples than the rest.
    """
    a = analytic(x)
    e = np.abs(a)
    d = np.diff(np.angle(a), axis=0)
    d = np.remainder(d + np.pi, 2 * np.pi) - np.pi
    f = d / (2 * np.pi * DT_S)
    w = e[1:]
    out = np.full(x.shape[1], np.nan)
    for j in range(x.shape[1]):
        wj, fj = w[:, j], f[:, j]
        live = wj > 0
        if not live.any():
            continue
        thr = np.quantile(wj[live], IF_QUANTILE)
        keep = (wj > thr) & (fj > IF_RANGE[0]) & (fj < IF_RANGE[1])
        if keep.any():
            out[j] = np.median(fj[keep])
    return out


def spectral_centroid(x):
    """Metric 3, one value per trace."""
    f = np.fft.rfftfreq(x.shape[0], DT_S)
    P = np.abs(np.fft.rfft(x, axis=0)) ** 2
    s = P.sum(0)
    out = np.full(x.shape[1], np.nan)
    live = s > 0
    out[live] = (P[:, live] * f[:, None]).sum(0) / s[live]
    return out


def lag2d(a, b):
    """Metric 5.  (dt, dx) that brings `a` onto `b`, plus the peak value."""
    A, B = env(a), env(b)
    A = A - A.mean()
    B = B - B.mean()
    na, nb = np.linalg.norm(A), np.linalg.norm(B)
    if na == 0 or nb == 0:
        return np.nan, np.nan, 0.0
    n = [1 << int(np.ceil(np.log2(2 * s))) for s in A.shape]
    Cc = np.fft.irfft2(np.fft.rfft2(A, n) * np.conj(np.fft.rfft2(B, n)), n)
    Cc /= na * nb
    mt, mx = LAG_SEARCH
    Cc = np.roll(Cc, (mt, mx), (0, 1))[:2 * mt + 1, :2 * mx + 1]
    i, j = np.unravel_index(np.argmax(Cc), Cc.shape)
    return i - mt, j - mx, float(Cc.max())


def lag_over(fake, real, off):
    """Metric 5 across sub-windows of one translated gather.

    Returns (values, bin) - a sub-window spans LAG_WINDOW[1] shots and so a
    range of offsets, and it is binned by the smallest of them: the near bin
    is defined by what it contains (the apex), not by its average.
    """
    nt, nx = LAG_WINDOW
    out, bins = [], []
    for t0 in range(0, fake.shape[0] - nt + 1, nt):
        for x0 in range(0, fake.shape[1] - nx + 1, nx):
            a = fake[t0:t0 + nt, x0:x0 + nx]
            b = real[t0:t0 + nt, x0:x0 + nx]
            if np.count_nonzero(b) < 0.5 * b.size:
                continue          # mostly mute: nothing to line up
            out.append(lag2d(a, b))
            bins.append(int(which_bin(off[x0:x0 + nx].min())))
    return (np.array(out) if out else np.empty((0, 3)),
            np.array(bins, dtype=int))


# --------------------------------------------------------------- metric 0 --


def tiles(gather, off, live_from=None):
    """Non-overlapping FID_PATCH tiles of one window, mute ones dropped.

    `live_from` is the gather liveness is judged on, when that is not the
    gather being tiled.  G(A) ends in tanh and is essentially never exactly
    zero, so judged on itself it would keep the tiles that sit inside the
    mute while the real-B set drops them - two sets tiled by different rules.
    Passing the input A keeps the selection geometric.

    Binned like the lag sub-windows, by the tile's smallest offset.
    """
    nt, nx = FID_PATCH
    mask = gather if live_from is None else live_from
    out = [[], []]
    for t0 in range(0, gather.shape[0] - nt + 1, nt):
        for x0 in range(0, gather.shape[1] - nx + 1, nx):
            m = mask[t0:t0 + nt, x0:x0 + nx]
            if np.count_nonzero(m) >= FID_MIN_LIVE * m.size:
                b = int(which_bin(off[x0:x0 + nx].min()))
                out[b].append(np.asarray(gather[t0:t0 + nt, x0:x0 + nx],
                                         dtype=np.float32))
    return out


def subsample(patches, n, seed=FID_SEED):
    """The same n for every set, drawn the same way - see FID_MAX_SAMPLES."""
    patches = np.asarray(patches, dtype=np.float32)
    if n is None or len(patches) <= n:
        return patches
    rng = np.random.default_rng(seed)
    return patches[np.sort(rng.choice(len(patches), n, replace=False))]


def fd_features(patches):
    """(n, 30) seismic features: log band powers, then log envelope quantiles.

    Deliberately hand-designed rather than learned.  Every dimension is a
    quantity that means something here - where the energy sits in frequency,
    and how the amplitude distribution is shaped - so an FD that moves can be
    traced to a dimension that moved, which is not true of an Inception
    activation.  And 30 x 30 is estimated properly from a thousand patches,
    where 2048 x 2048 is not.
    """
    x = np.asarray(patches, dtype=np.float64)
    if x.ndim == 2:
        x = x[None]
    n, nt, _ = x.shape
    f = np.fft.rfftfreq(nt, DT_S)
    P = (np.abs(np.fft.rfft(x, axis=1)) ** 2).mean(axis=2)     # (n, nf)
    edges = np.linspace(FD_BAND_HZ[0], FD_BAND_HZ[1], FD_BANDS + 1)
    band = np.empty((n, FD_BANDS))
    for i in range(FD_BANDS):
        m = (f >= edges[i]) & (f < edges[i + 1])
        if not m.any():                      # band narrower than the bin
            m = np.array([int(np.argmin(np.abs(f - 0.5 * (edges[i]
                                                          + edges[i + 1]))))])
        band[:, i] = P[:, m].mean(axis=1)
    e = np.stack([env(p).ravel() for p in x])                  # (n, nt*nx)
    pct = np.percentile(e, FD_PCTL, axis=1).T                  # (n, 6)
    return np.log10(np.concatenate([band, pct], axis=1) + FD_FLOOR)


def fd_stats(patches, n):
    f = fd_features(subsample(patches, n))
    return f.mean(axis=0), np.cov(f, rowvar=False)


def halves(patches, n):
    """Up to 2n patches, shuffled, for a floor to be measured on.

    The shuffle is the point.  The tiles arrive ordered by receiver and then
    by shot window, so splitting them as they come measures the difference
    between one end of the array and the other - a real quantity, but not
    the sampling floor, and on the streamer it came out 17.5 against a
    das-vs-str distance of 25.1.  Permuted first, the two halves differ by
    sampling alone, which is what the floor is supposed to be.
    """
    p = np.asarray(patches, dtype=np.float32)
    rng = np.random.default_rng(FID_SEED)
    p = p[rng.permutation(len(p))]
    return p if n is None else p[:2 * n]


def fd_floor(patches, n):
    """As fid_floor, for metric 0a.  See the note there about n."""
    p = halves(patches, n)
    h = len(p) // 2
    if h < 8:
        return np.nan, h
    return calculate_frechet_distance(*fd_stats(p[:h], None),
                                      *fd_stats(p[h:2 * h], None)), h


def fid_stats(patches, n, network):
    return calculate_activation_statistics(
        subsample(patches, n)[:, None], FID_BATCH, DEVICE, network)


def fid_floor(patches, n, network):
    """str against str, split in half: what a perfect translation would score.

    It is not zero.  Two halves of the same distribution differ by sampling
    alone, and at this sample count against a 2048 x 2048 covariance that
    difference is large - which is exactly why the raw FID cannot be read
    without it.

    Returns (value, n_per_half).  Two disjoint halves of n each need 2n
    tiles and there are usually not that many, so n_per_half is normally
    below the n the other FIDs use.  Fewer samples means a larger distance,
    so the floor printed is an upper bound on the floor at the full n - if
    a checkpoint's FID sits near it, it is comfortably at the floor.
    """
    p = halves(patches, n)
    h = len(p) // 2
    if h < 8:
        return np.nan, h
    mu1, s1 = calculate_activation_statistics(p[:h, None], FID_BATCH,
                                              DEVICE, network)
    mu2, s2 = calculate_activation_statistics(p[h:2 * h, None], FID_BATCH,
                                              DEVICE, network)
    return calculate_frechet_distance(mu1, s1, mu2, s2), h


# ------------------------------------------------------------- aggregation --


def nanmed(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else np.nan


def nanavg(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else np.nan


class Split:
    """Per-trace (or per-sub-window) values kept apart by offset bin."""

    def __init__(self):
        self.v = [[], []]

    def add(self, values, bins):
        values = np.asarray(values, dtype=float).ravel()
        bins = np.asarray(bins, dtype=int).ravel()
        for b in (0, 1):
            m = bins == b
            if m.any():
                self.v[b].append(values[m])

    def get(self, b):
        return np.concatenate(self.v[b]) if self.v[b] else np.array([])

    def n(self, b):
        return int(self.get(b).size)


# ------------------------------------------------------------------- stage --


def identity(a):
    """The `transform` that makes `score` measure the input itself."""
    return a.astype(np.float64)


def translator(gen):
    def f(a):
        t = torch.from_numpy(a)[None, None].float().to(DEVICE)
        with torch.no_grad():
            return gen(t).cpu().numpy()[0, 0].astype(np.float64)
    return f


def reference_str(B_ds, off_b, recv, starts):
    """Metrics 2 and 3 on the streamer, per bin, and its tiles for the FID.

    The streamer is the target, so these are what the G(A) columns are
    supposed to move towards.  Metrics 0, 1 and 5 have no meaning against
    itself - 1 and 5 would be exactly 1 and (0, 0), and 0 is the floor,
    which fid_floor reports separately.
    """
    nt, nx = CROP
    fb, cb, tl = Split(), Split(), [[], []]
    for k in recv:
        for x0 in starts:
            b = take(B_ds, k, 0, x0, nt, nx)
            ob = off_b[k, x0:x0 + nx]
            fb.add(inst_freq(b), which_bin(ob))
            cb.add(spectral_centroid(b), which_bin(ob))
            if USE_FD or USE_FID:
                for i, t in enumerate(tiles(b, ob)):
                    tl[i] += t
    return fb, cb, tl


def score(transform, A_ds, B_ds, off_a, recv, starts):
    """Every metric for `transform(A)`, per offset bin.

    `transform` is the generator, or `identity` for the untranslated input -
    which is how the baseline row is produced, by the same code and on the
    same windows rather than by a separate path that could drift from it.
    Metric 5 is then A against A, so it returns exactly (0, 0) at peak 1.000;
    that row is a check on the lag machinery, not a measurement.
    """
    nt, nx = CROP
    ec, fq, ct = Split(), Split(), Split()
    dt_, dx_, pk_ = Split(), Split(), Split()
    fake_tiles = [[], []]
    for k in recv:
        for x0 in starts:
            a = take(A_ds, k, 0, x0, nt, nx)
            off = off_a[k, x0:x0 + nx]
            bins = which_bin(off)
            fake = transform(a)

            if B_ds is not None:
                b = take(B_ds, k, 0, x0, nt, nx)
                ec.add(envelope_corr(fake, b), bins)      # metric 1
            fq.add(inst_freq(fake), bins)                 # metric 2
            ct.add(spectral_centroid(fake), bins)         # metric 3

            L, lb = lag_over(fake, a, off)                # metric 5
            if len(L):
                dt_.add(L[:, 0], lb)
                dx_.add(L[:, 1], lb)
                pk_.add(L[:, 2], lb)

            if USE_FD or USE_FID:                         # metric 0
                for bb, tt in enumerate(tiles(fake, off, live_from=a)):
                    fake_tiles[bb] += tt

    rows = {}
    for b, name in enumerate(BINS):
        d = dt_.get(b)
        rows[name] = dict(
            n_trace=ec.n(b) if B_ds is not None else fq.n(b),
            env=nanavg(ec.get(b)),
            if_med=nanmed(fq.get(b)),
            centroid=nanmed(ct.get(b)),
            dt=nanmed(d),
            dx=nanmed(dx_.get(b)),
            dt_within2=float(np.mean(np.abs(d) <= 2)) if d.size else np.nan,
            peak=nanmed(pk_.get(b)),
        )
    return fake_tiles, rows


def fmt(v, w, p, suffix=""):
    return ("nan" if not np.isfinite(v) else f"{v:.{p}f}").rjust(
        w - len(suffix)) + suffix


def main():
    ck = find_checkpoints()
    if not ck:
        raise SystemExit(f"no checkpoints named {TAG}_<epoch> under "
                         f"{resolve('checkpoint')}")

    A_ds = PohangShoreDataset(is_das=True, crop_size=None, total_length=1,
                              stage=STAGE)
    B_ds = (PohangShoreDataset(is_das=False, crop_size=None, total_length=1,
                               stage="rg_train")
            if STAGE == "rg_train" else None)
    off_a = offsets(STAGE)
    off_b = offsets("str") if B_ds is not None else None
    if off_a.shape[0] != A_ds.n_data or off_a.shape[1] != A_ds.n_shot:
        raise SystemExit(f"offsets are {off_a.shape}, the {STAGE} array is "
                         f"({A_ds.n_data}, ., {A_ds.n_shot})")

    recv = (list(range(A_ds.n_data)) if EVAL_RECEIVERS is None
            else list(EVAL_RECEIVERS))
    starts = window_starts(A_ds.n_shot, CROP[1])

    print(f"{TAG}: {len(ck)} checkpoints, epochs "
          f"{ck[0][0]}..{ck[-1][0]}   device {DEVICE}   stage {STAGE}")
    print(f"  {len(recv)} receivers x {len(starts)} windows of {CROP[0]}x"
          f"{CROP[1]}   shot starts {starts}")
    seen = off_a[recv][:, [x for s in starts for x in range(s, s + CROP[1])]]
    frac = float(np.mean(seen < OFFSET_SPLIT_M))
    print(f"  offset split at {OFFSET_SPLIT_M} m: {100 * frac:.2f} % of the "
          f"scored traces are near, {100 * (1 - frac):.2f} % far")
    if B_ds is None:
        print(f"  no streamer counterpart at this stage - metric 1 is nan, "
              f"and the FID reference is the streamer's far patches")

    # The input, through the same code path as a checkpoint.  This is the row
    # every other row is read against: what the metrics say when nothing has
    # been translated at all.
    print("")
    print("  scoring the input", flush=True)
    A_tiles, A_rows = score(identity, A_ds, B_ds, off_a, recv, starts)

    S_rows, B_tiles = None, [[], []]
    if B_ds is not None:
        print(f"  scoring the streamer", flush=True)
        fb, cb, B_tiles = reference_str(B_ds, off_b, recv, starts)
        S_rows = {n: dict(n_trace=fb.n(b), env=np.nan,
                          if_med=nanmed(fb.get(b)),
                          centroid=nanmed(cb.get(b)), dt=np.nan, dx=np.nan,
                          dt_within2=np.nan, peak=np.nan,
                          fd=np.nan, fid=np.nan)
                  for b, n in enumerate(BINS)}

    inception, B_stats, B_fd, fid_n = None, {}, {}, 0
    use_fd, use_fid = USE_FD, USE_FID
    for name in BINS:
        A_rows[name]["fd"] = np.nan
        A_rows[name]["fid"] = np.nan
    if use_fd or use_fid:
        counts = [len(A_tiles[b]) for b in (0, 1)]
        sizes = [c for c in counts + [len(t) for t in B_tiles] if c]
        fid_n = min(sizes) if sizes else 0
        if FID_MAX_SAMPLES is not None:
            fid_n = min(fid_n, FID_MAX_SAMPLES)
        print(f"  metric 0 on {FID_PATCH[0]}x{FID_PATCH[1]} tiles, {fid_n} "
              f"per set   das near {counts[0]} / far {counts[1]}, "
              f"str near {len(B_tiles[0])} / far {len(B_tiles[1])}")
        if fid_n < 8:
            print("  too few live tiles; skipping metric 0")
            use_fd = use_fid = False

    # The reference is the streamer patches for that bin, and the near bin
    # has none anywhere in the survey, so it falls back to the far ones.
    # That is the honest comparison available, not a like-for-like one - see
    # the header.  With no streamer at all the reference is the input itself,
    # which makes metric 0 a distance travelled rather than a distance
    # remaining; the table says so.
    ref = B_tiles if B_ds is not None else A_tiles
    if use_fd:
        for b, name in enumerate(BINS):
            B_fd[name] = fd_stats(ref[b] if ref[b] else ref[1], fid_n)
            A_rows[name]["fd"] = (
                calculate_frechet_distance(*fd_stats(A_tiles[b], fid_n),
                                           *B_fd[name])
                if A_tiles[b] else np.nan)
            if S_rows is not None and len(ref[b]) >= 16:
                # the str row carries the floor: the same distribution
                # against itself, which is what a perfect G(A) would score
                S_rows[name]["fd"], zn = fd_floor(ref[b], fid_n)
                S_rows[name]["floor_n"] = zn
    if use_fid:
        inception = InceptionNetwork().eval().to(DEVICE)
        for b, name in enumerate(BINS):
            B_stats[name] = fid_stats(ref[b] if ref[b] else ref[1],
                                      fid_n, inception)
            A_rows[name]["fid"] = (
                calculate_frechet_distance(
                    *fid_stats(A_tiles[b], fid_n, inception), *B_stats[name])
                if A_tiles[b] else np.nan)
            if S_rows is not None and len(ref[b]) >= 16:
                S_rows[name]["fid"], zn = fid_floor(ref[b], fid_n, inception)
                S_rows[name]["floor_n"] = zn
    del A_tiles, B_tiles, ref

    hdr = ("what", "bin", "FD", "FID", "env corr", "inst f", "centroid",
           "dt", "dx", "|dt|<=2", "peak", "n")
    w = [7, 6, 8, 9, 10, 9, 10, 7, 7, 9, 7, 8]

    def emit(label, table):
        for b, name in enumerate(BINS):
            r = table[name]
            print(f"  {label if b == 0 else '':>7}{name:>6}"
                  f"{fmt(r['fd'], 8, 2)}{fmt(r['fid'], 9, 1)}"
                  f"{fmt(r['env'], 10, 3)}"
                  f"{fmt(r['if_med'], 9, 1, 'H')}"
                  f"{fmt(r['centroid'], 10, 1, 'H')}"
                  f"{fmt(r['dt'], 7, 1)}{fmt(r['dx'], 7, 1)}"
                  f"{fmt(100 * r['dt_within2'], 9, 0, '%')}"
                  f"{fmt(r['peak'], 7, 3)}{r['n_trace']:>8}", flush=True)
            rows.append((label, name, r))

    print("")
    print("  " + "".join(h.rjust(x) for h, x in zip(hdr, w)))
    print("  " + "".join("-" * x for x in w))
    rows = []
    emit("A", A_rows)
    if S_rows is not None:
        emit("str", S_rows)
    print("  " + "".join("-" * x for x in w))
    print("    A    the input, untranslated - metric 5 is A against itself, "
          "so (0, 0) at 1.000 is the check that it works")
    if S_rows is not None:
        print("    str  the target - metrics 1 and 5 are undefined against "
              "itself, and its FD / FID are the floor "
              f"(n={S_rows['far'].get('floor_n', '?')} per half, an upper "
              "bound)")

    for ep, path in ck:
        state = torch.load(path, map_location="cpu")
        g, ch = build_generator(state["G_A2B"])
        fake_tiles, m = score(translator(g), A_ds, B_ds, off_a, recv, starts)
        for b, name in enumerate(BINS):
            ok = len(fake_tiles[b]) >= 8
            m[name]["fd"] = (
                calculate_frechet_distance(*fd_stats(fake_tiles[b], fid_n),
                                           *B_fd[name])
                if use_fd and ok else np.nan)
            m[name]["fid"] = (
                calculate_frechet_distance(
                    *fid_stats(fake_tiles[b], fid_n, inception),
                    *B_stats[name])
                if use_fid and ok else np.nan)
        emit(str(ep), m)
        del g, state, fake_tiles
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    if CSV_DIR:
        p = os.path.join(resolve(CSV_DIR), f"metrics_{TAG}_{STAGE}.csv")
        keys = [k for k in rows[0][2] if k != "floor_n"]
        with open(p, "w", encoding="utf-8") as f:
            f.write("what,bin," + ",".join(keys) + "\n")
            for label, name, r in rows:
                f.write(f"{label},{name}," +
                        ",".join(f"{r[k]:.6g}" for k in keys) + "\n")
        print("")
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
