"""RMS normalisation and log-envelope scaling: the last stage before the dataset.

    data/pohang_shore/das_data_norm.npy  (551, 2000, 264)
      ->  das_data_log.npy       (551, 2000, 264)   shot-major, the gain applied
      ->  das_data_rg_train.npy  ( 24, 2000, 551)   receiver-major, shared aperture
      ->  das_data_rg_infer.npy  (240, 2000, 551)   receiver-major, the rest

    data/pohang_shore/str_data_norm.npy  (551, 2000,  24)
      ->  str_data_log.npy       (551, 2000,  24)
      ->  str_data_rg_train.npy  ( 24, 2000, 551)

Three steps:

  1. RMS normalise.  One scalar per domain, `TARGET_RMS / rms(input)`, applied
     to every shot.  Relative amplitudes between shots and between traces are
     untouched - only the unit changes.  (`norm` has already had this done, so
     on that input the scalar comes out at 1.)
  2. Log-envelope gain.  `utils.process.calculate_logscale` builds a smoothly
     varying scale from the trace envelope and the data is multiplied by it, so
     a gather shows the late, weak arrivals alongside the early, strong ones
     instead of only the direct wave.  Fitted ON SHOT GATHERS, one at a time,
     which is what SCALE_MODE means; the result is `log`.
  3. Transpose to receiver-major and split by shared aperture, which is what
     module/dataset_pohang_shore.py reads - see write_all and
     shared_receivers().

Both follow `process_data.py` in the 4d-noise-attenuation project, which
normalises to a common RMS (NORMALIZED_RMS) before fitting the envelope gain
and saves the gain and its parameters alongside the data.

Why the normalisation has to come first
---------------------------------------
`calculate_logscale` is not scale invariant.  Its `log_base` is added to the
envelope in the envelope's own units, so it only means something relative to
the amplitudes it meets - the same number applied to two differently scaled
arrays compresses them by different amounts.

That matters here more than usual, because the two instruments report
amplitude four orders of magnitude apart.  Normalising both to TARGET_RMS
first puts `log_base` on a common footing, which for an unpaired translation
matters directly: if one domain comes out visibly more compressed than the
other, that difference is a cue the network can learn instead of the physics.

log_base has to sit BELOW the amplitudes it is meant to lift
-----------------------------------------------------------
The gain works out to

    scale(env) = log10(1 + env / log_base) / (env + log_base)

which is zero at env = 0, peaks at env = (e - 1) * log_base, and falls off as
1/env above that.  Only the falling side compresses.  So log_base belongs
below the envelope of the quietest thing worth keeping; above it the gain
rises with amplitude, which is the opposite of a gain control.  Measured on
the DAS at RMS 0.1, envelope median 0.025:

    log_base   output p95/p50 (input 15.65)
    1e-4        1.93   compressed
    1e-3        2.39   compressed
    1e-2        5.04   compressed
    5e-2       16.25   EXPANDED
    1e-1       30.06   EXPANDED

The run prints base/median for each domain; keep those two ratios close and
the two domains are being compressed alike.

SCALE_MODE
----------
"per_shot" fits a separate gain for every shot gather.  The shot moves along
the line here, so the moveout apex moves with it; one gain map averaged over
shots would smear across that.  This is the default.

"average" fits a single gain from the mean gather and applies it to all shots,
the way 4d-noise-attenuation does - correct there because its geometry is
fixed.  It preserves relative amplitude between shots, which "per_shot" does
not.

Parameters and measurements are written next to `rg_train` as
`<name>_meta.json` and `<name>_meta.npz`, including `shot_order` so the
sorting is reversible.  Nothing downstream reads them; they are there so a
saved array can be traced back to what produced it.

The preview
-----------
PREVIEW_MODE picks between shot gathers and receiver gathers.  The gain is
always fitted on shot gathers for the written arrays; the receiver-gather
preview fits in that domain instead, which for these parameters is the same
transform - see independent_traces(), which checks the condition at run time
and says so when it does not hold.

Edit the settings block below, then run from the repository root:

    python -m process.logenv_process
"""

import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np

import config as C
# The along-line shot ordering, for the receiver-gather preview.  Taken from
# there rather than from a stage sidecar so it does not depend on INPUT_STAGE.
import process.shot_geometry as G
from utils.data import create_memmap, get_project_root, npy_shape
from utils.process import calculate_logscale, envelope_1d

# ---------------------------------------------------------------- settings --

# Domains to process, in order.  The input, the output and every parameter
# come from config - see config.ARRAYS, config.TARGET_RMS and
# config.LOG_SCALE_PARAMS.  Change the values there, not here.
#
# "str" is left out while str_data_freq.npy does not exist - a missing input
# aborts the run, so listing it would kill the das preview on the way past.
# Put it back once the streamer band-pass has been written, or set
# INPUT_STAGE = "geom" to read both sides pre-filter.
DOMAINS = ("das",)

# Which stage to read.  "freq" is the band-passed array; use "geom" for a
# domain whose band-pass has not been run.
INPUT_STAGE = "norm"

# Draw the input, the log-scaled result and the gain between them, and show
# them.  Costs one extra read and writes nothing.
PREVIEW = True

# "shot" draws shot gathers, one figure per shot in PREVIEW_SHOTS.
#
# "receiver" draws receiver gathers instead - one receiver, all 551 shots
# across.  That is the domain the training uses, sorted along the survey line.
#
# The gain is fitted on the receiver gather directly, and for these parameters
# that is not an approximation: with smooth_sigma's second entry at 0 nothing
# couples the traces, so fitting per shot gather and fitting per receiver
# gather give the same number for every sample - measured, they agree to
# 0.0006-0.010 %, which is float32 rounding.  See independent_traces(), which
# checks the condition and says so when it does not hold.
#
# It costs one pass over the shots to read the columns, then one fit per
# receiver rather than 551.
PREVIEW_MODE = "shot"

# Shot gathers to draw, in "shot" mode.
PREVIEW_SHOTS = range(0, 551, 50)

# Receivers to draw, in "receiver" mode.  None picks PREVIEW_N evenly spaced.
PREVIEW_RECEIVERS = None
PREVIEW_N = 3

# Order the shot axis of a receiver gather along the survey line rather than
# by recording number.  The survey is four passes over one line, so recording
# order visits each stretch of seabed four times hundreds of shots apart;
# sorted, neighbours on the axis are neighbours on the ground and the shot
# interval drops from 15.24 m to a median of 3.33 m.  The ordering comes from
# process/shot_geometry.py, so it does not depend on which stage is read.
SORT_BY_POSITION = True

SAMPLES = None
CLIP_PERC = 99.0

# Write the outputs: log-scaled and transposed to receiver-major,
# (n_receivers, n_samples, n_shots), split into `rg_train` and `rg_infer`.
WRITE_ALL = False

# Receivers held in memory per block while transposing.  Each block costs one
# strided pass over the input, so bigger is fewer passes and more memory:
# 64 receivers of the DAS is 551 x 2000 x 64 x 4 = 282 MB.
RG_BLOCK = 64

# Order the shot axis along the survey line in the written arrays, not just in
# the preview.  A receiver gather is only readable sorted - see
# SORT_BY_POSITION - and this is the axis the training convolves along, so the
# ordering belongs in the file rather than in whatever loads it.  The order is
# recorded in the sidecar as `shot_order`, so it is reversible.
SORT_WRITTEN = True

# The stage whose sidecar carries the receiver coordinates, for deciding which
# receivers the streamer overlaps.  `norm` has no sidecar of its own; it came
# from these.
RG_GEOM_STAGE = "deci"


# Show the preview interactively, and optionally save it.  A falsy SAVE_DIR
# skips saving.
SHOW = True
SAVE_DIR = None

# Write the sidecar parameter/measurement files.  Independent of WRITE_ALL -
# a preview that settled on a norm_scale and a log_base is worth recording on
# its own - so it is off while nothing is being kept.
WRITE_META = False

# Zeros put in FRONT of each gather before the gain is fitted, and dropped
# afterwards.  0 disables it.
#
# Why a pad at all.  calculate_logscale builds the envelope from a Hilbert
# transform along time, whose kernel is 1/t and whose FFT is circular, so
# without a pad the record's late energy wraps round into the envelope at
# t = 0.  The muted window sits at t = 0 and its own envelope is only ~1e-5,
# so a small absolute leak is an enormous relative one.  Measured against a
# 2000-sample pad, running with none at all leaves the middle of the record
# alone (median 0.003-0.008 %, and at most 5.9 % over 500-1500 ms) and gets
# the ends wrong by orders of magnitude - 111 693 % at 0-250 ms on the
# streamer, 26 652 % on the DAS.
#
# Why the FRONT rather than the end.  The envelope itself does not care: the
# transform is circular, [pad, record] is a cyclic rotation of [record, pad],
# and the two give bit-identical envelopes (measured, 0.000e+00).  What breaks
# the tie is that calculate_logscale then runs gaussian_filter over the padded
# array, and THAT is not circular - it reflects at the array's edges.  So the
# pad decides which end of the record sits against a real edge:
#
#     pad at the front   the record's start is interior, its end is at the
#                        edge.  0-250 ms is protected; 1750-2000 ms is not,
#                        and comes out as if unpadded (370.7 % against a
#                        tail pad, which is the same 371 % an unpadded run
#                        gives there).
#     pad at the end     the reverse.
#
# The front is the right choice here because the muted window is the region
# being looked at, and because the record's end at 2000 ms is deep and quiet -
# a filter-edge artefact there costs far less than one on top of the mute.
#
# Set both to 2000 to protect both ends; the only cost is FFT length.
PAD_FRONT = 2000

# Zeros appended to the END of each gather.  See PAD_FRONT.
#
# 2000 samples is 2 s on the 1 ms grid, comfortably longer than the envelope
# smoothing reaches.  The gain in the pad region is discarded, so the only
# cost is the extra FFT length.
PAD_TAIL = 0

# ----------------------------------------------------------------- derived --

TARGET_RMS = C.TARGET_RMS
RMS_SAMPLE_SHOTS = C.RMS_SAMPLE_SHOTS
SCALE_MODE = C.SCALE_MODE

# (tag, input, output) per domain.  Inputs sit in numpy/ with the rest of the
# working arrays; outputs go to the pohang_shore root, where the dataset looks
# for them.
JOBS = tuple((tag, C.ARRAYS[tag][INPUT_STAGE]) for tag in DOMAINS)

# ---------------------------------------------------------------------------


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(get_project_root(), path)


def sample_indices(n_shots):
    if RMS_SAMPLE_SHOTS is None or RMS_SAMPLE_SHOTS >= n_shots:
        return np.arange(n_shots)
    return np.unique(np.linspace(0, n_shots - 1, RMS_SAMPLE_SHOTS).astype(int))


def measure_rms(data, idx):
    """Overall RMS over the sampled shots, plus the per-shot values.

    Accumulated as a sum of squares rather than a mean of means so the result
    is the true RMS of the sampled block even though it is read shot by shot.
    """
    total, count = 0.0, 0
    per_shot = np.zeros(len(idx))
    for k, i in enumerate(idx):
        g = np.asarray(data[i], dtype=np.float64)
        ss = float(np.sum(g ** 2))
        total += ss
        count += g.size
        per_shot[k] = np.sqrt(ss / g.size)
    return float(np.sqrt(total / count)), per_shot


def average_gather(data, idx):
    """Mean gather over the sampled shots, for SCALE_MODE == "average"."""
    acc = np.zeros(data.shape[1:], dtype=np.float64)
    for i in idx:
        acc += np.asarray(data[i], dtype=np.float64)
    return acc / len(idx)


def logscale(gather, lsp):
    """calculate_logscale on a padded gather, with the pad dropped.

    See PAD_FRONT and PAD_TAIL for why, and which end each protects.  The pad
    is zeros, so it also becomes the array's minimum envelope, which pins
    calculate_logscale's `data_env_log.min()` to log10(log_base) - the same
    number whatever the gather contains.
    """
    if not (PAD_FRONT or PAD_TAIL):
        return calculate_logscale(gather, **lsp)
    nt, ntr = gather.shape
    parts = []
    if PAD_FRONT:
        parts.append(np.zeros((PAD_FRONT, ntr), dtype=gather.dtype))
    parts.append(gather)
    if PAD_TAIL:
        parts.append(np.zeros((PAD_TAIL, ntr), dtype=gather.dtype))
    padded = np.concatenate(parts, axis=0)
    return calculate_logscale(padded, **lsp)[PAD_FRONT:PAD_FRONT + nt]


def shared_receivers(tag, n_traces):
    """Receiver indices inside the aperture the two instruments share.

    The streamer array spans 71.3 m of the DAS's 793 m of fibre, so most DAS
    receivers have no streamer counterpart at all.  Those can be inferred on
    but never trained against, which is why the two get separate arrays.

    Membership is decided on position, not on an index range, so it stays
    right if the decimation stride changes: a receiver is shared when its
    distance along the fibre falls inside the streamer array's own extent.
    """
    if tag == "str":
        return np.arange(n_traces), None
    side = resolve(C.META[f"{tag}_{RG_GEOM_STAGE}"])
    str_side = resolve(C.META["str_line"])
    for p in (side, str_side):
        if not os.path.isfile(p):
            raise SystemExit(f"not found: {p} - needed to split the shared "
                             f"aperture")
    rec = np.load(side)["receiver_xy"]
    if len(rec) != n_traces:
        raise SystemExit(f"{os.path.basename(side)} has {len(rec)} receivers, "
                         f"the array has {n_traces}")
    s = np.load(str_side)
    sr = s["receiver_xy"]
    u = s["line_direction"]
    c = s["line_centroid"]
    a_rec = (rec - c) @ u
    a_str = (sr - c) @ u
    lo, hi = a_str.min(), a_str.max()
    inside = np.flatnonzero((a_rec >= lo) & (a_rec <= hi))
    outside = np.flatnonzero((a_rec < lo) | (a_rec > hi))
    print(f"    shared aperture: streamer spans {lo:.1f}..{hi:.1f} m along the "
          f"line ({hi - lo:.1f} m)")
    print(f"      {len(inside)} of {n_traces} receivers inside "
          f"(indices {inside.min()}..{inside.max()}), {len(outside)} outside")
    return inside, outside


def stage_figure(tag, shot, nor, log, scale, shared_scale, lsp):
    """One shot: the input, the log-scaled result, and the gain between them.

    No separate panel for the un-normalised input.  The RMS step is a single
    scalar and every panel is percentile-clipped on its own, so it would come
    out pixel-identical to the normalised one; norm_scale is on the console.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    cols = ((nor, f"input, RMS -> {TARGET_RMS:g}", "seismic"),
            (log, "log envelope", "seismic"),
            (scale, "gain" + (" (shared)" if shared_scale else ""), "viridis"))
    for col, (img, name, cmap) in enumerate(cols):
        ax = axes[col]
        if cmap == "seismic":
            clip = np.percentile(np.abs(img), CLIP_PERC)
            clip = clip if clip > 0 else 1.0
            ax.imshow(img, cmap=cmap, vmin=-clip, vmax=clip, aspect="auto",
                      interpolation="nearest")
        else:
            ax.imshow(img, cmap=cmap, aspect="auto", interpolation="nearest")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("trace")
    axes[0].set_ylabel("sample")
    fig.suptitle(f"{tag} shot {shot}  -  base {lsp['log_base']:g}, "
                 f"sigma {lsp['smooth_sigma']}, {SCALE_MODE}")
    fig.tight_layout()
    return fig


def sorted_shots(n_shots):
    """(shot order, along-line position) for the receiver-gather axis.

    None for the position when the shots are left in recording order.
    """
    if not SORT_BY_POSITION:
        return np.arange(n_shots), None
    o = G.sorted_order()
    order = np.asarray(o["orig_idx"], dtype=int)
    if len(order) != n_shots:
        raise SystemExit(f"shot_geometry has {len(order)} shots, the array "
                         f"has {n_shots}")
    return order, np.asarray(o["along_m"], dtype=float)


def pick_receivers(n_traces):
    want = PREVIEW_RECEIVERS
    if want is None:
        want = np.linspace(0, n_traces - 1, PREVIEW_N).round().astype(int)
    bad = [int(j) for j in want if not 0 <= j < n_traces]
    if bad:
        raise SystemExit(f"preview receivers out of range 0..{n_traces - 1}: "
                         f"{bad}")
    return [int(j) for j in want]


def independent_traces(lsp):
    """Does the gain decouple the second axis, so the domain cannot matter?

    Only the receiver-gather PREVIEW fits in that domain; the written arrays
    always fit on shot gathers and are transposed afterwards - see write_all.
    So this answers whether the preview shows the same gain the output carries.

    calculate_logscale is per trace when three things hold:

      * envelope_1d is a Hilbert transform along time, one trace at a time.
        (It runs a 2-D FFT, but the filter depends on the time frequency only
        and is broadcast across the other axis, so that axis is transformed
        and inverted unchanged.)
      * smooth_sigma's second entry is 0, so the gaussian filter does not mix
        neighbours on the second axis either.
      * the minimum it subtracts is the same in both, which the zero pad sees
        to: with PAD_FRONT or PAD_TAIL set, the pad is the array's minimum
        envelope and data_env_log.min() is log10(log_base) whatever the gather
        holds.

    Measured: fitting on shot gathers and reading the receiver columns out,
    against fitting on the receiver gather directly, agree to 0.0006-0.010 % -
    float32 rounding.
    """
    s = lsp.get("smooth_sigma")
    if not isinstance(s, (tuple, list)):
        return False, f"smooth_sigma {s!r} is scalar, so it smooths both axes"
    if len(s) < 2 or s[1] != 0:
        return False, (f"smooth_sigma {tuple(s)} smooths the second axis, "
                       f"which is traces in a shot gather and shots here")
    if not (PAD_FRONT or PAD_TAIL):
        return False, ("no pad, so data_env_log.min() is measured on "
                       "whichever gather is passed in")
    return True, ""


def receiver_panels(tag, data, norm_scale, lsp, shared_scale, recv, order):
    """(input, log, gain) receiver gathers, one set per entry of `recv`.

    The gain is fitted ON the receiver gather, not carried over from shot
    gathers, because for these parameters the two are the same thing - see
    independent_traces().  That turns 551 fits into one per receiver.

    The one shot-major pass is for the reading, not the fitting: pulling
    data[:, :, j] out of a memmap directly is a strided read that touches
    every page of the file for each receiver, while one pass over the shots
    collects every preview column at once.
    """
    n_shots, nt, _ = data.shape
    ns = nt if SAMPLES is None else min(SAMPLES, nt)
    nor_p = np.zeros((len(recv), ns, n_shots), dtype=np.float32)

    t0 = time.time()
    for n, s in enumerate(order):
        g = np.asarray(data[s, :ns], dtype=np.float32)
        for r, j in enumerate(recv):
            nor_p[r, :, n] = g[:, j]
        if (n + 1) % 100 == 0 or n + 1 == n_shots:
            el = time.time() - t0
            print(f"      read {n + 1}/{n_shots} shots  {el:.0f}s elapsed, "
                  f"eta {el / (n + 1) * (n_shots - n - 1):.0f}s", flush=True)
    nor_p *= norm_scale

    ok, why = independent_traces(lsp)
    if shared_scale is not None:
        # "average" mode fits one map on the shot-gather grid; it has no
        # meaning transposed, so the receiver preview refits per gather and
        # says so.
        print(f"      SCALE_MODE {SCALE_MODE!r}: the shared gain is fitted on "
              f"the shot-gather grid, so this preview refits per receiver "
              f"gather instead")
    elif not ok:
        print(f"      NOTE: {why} - so the gain fitted here is NOT the gain "
              f"the written array would carry")
    gain_p = np.stack([logscale(nor_p[r], lsp)
                       for r in range(len(recv))])
    return nor_p, nor_p * gain_p, gain_p


def receiver_figure(tag, j, nor, log, gain, shared, lsp, along, dt_ms):
    """One receiver: the input, the log-scaled result, and the gain.

    The horizontal axis is the shot, ordered along the line when
    SORT_BY_POSITION is on; a second axis on top carries the metres, because
    the sorted shot numbers are no longer in order and the spacing is uneven.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6))
    n_sh = nor.shape[1]
    xs = np.arange(n_sh)
    cols = ((nor, f"input, RMS -> {TARGET_RMS:g}", "seismic"),
            (log, "log envelope", "seismic"),
            (gain, "gain" + (" (shared)" if shared else " (fitted per shot)"),
             "viridis"))
    for col, (img, name, cmap) in enumerate(cols):
        ax = axes[col]
        ext = [-0.5, n_sh - 0.5, img.shape[0] * dt_ms, 0.0]
        if cmap == "seismic":
            clip = np.percentile(np.abs(img), CLIP_PERC)
            clip = clip if clip > 0 else 1.0
            ax.imshow(img, cmap=cmap, vmin=-clip, vmax=clip, aspect="auto",
                      interpolation="nearest", extent=ext)
        else:
            ax.imshow(img, cmap=cmap, aspect="auto",
                      interpolation="nearest", extent=ext)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("position along the line [sorted shot]"
                      if along is not None else "shot")
        if along is not None:
            sec = ax.secondary_xaxis(
                "top", functions=(lambda v: np.interp(v, xs, along),
                                  lambda m: np.interp(m, along, xs)))
            sec.set_xlabel("along the line [m]", fontsize=8)
            sec.tick_params(labelsize=7)
    axes[0].set_ylabel("time [ms]")
    fig.suptitle(f"{tag} receiver gather, trace {j}  -  base "
                 f"{lsp['log_base']:g}, sigma {lsp['smooth_sigma']}, "
                 f"{SCALE_MODE}\nthe gain is fitted on shot gathers, so along "
                 f"this axis it is a series of independent fits")
    fig.tight_layout()
    return fig


def preview_receiver(tag, data, norm_scale, lsp, shared_scale):
    """One figure per receiver, drawn and shown before the next is built."""
    n_shots, nt, n_traces = data.shape
    dt_ms = C.STAGE_DT_US[tag][INPUT_STAGE] / 1000.0
    recv = pick_receivers(n_traces)
    order, along = sorted_shots(n_shots)
    out_dir = None
    if SAVE_DIR:
        out_dir = resolve(SAVE_DIR)
        os.makedirs(out_dir, exist_ok=True)

    print(f"    preview: {len(recv)} receiver gathers {recv} over "
          f"{n_shots} shots"
          + (", sorted along the line" if along is not None
             else ", in recording order"))
    if along is not None:
        gap = np.diff(along)
        print(f"      {along[0]:.0f}..{along[-1]:.0f} m, gaps median "
              f"{np.median(gap):.2f} m, max {gap.max():.2f} m")
    nor_p, log_p, gain_p = receiver_panels(tag, data, norm_scale, lsp,
                                           shared_scale, recv, order)

    for r, j in enumerate(recv):
        print(f"      trace {j:>5}: rms "
              f"{np.sqrt(np.mean(nor_p[r] ** 2)):.4g} -> "
              f"{np.sqrt(np.mean(log_p[r] ** 2)):.4g}, "
              f"|log| max {np.abs(log_p[r]).max():.4g}, "
              f"gain {gain_p[r].min():.3g}..{gain_p[r].max():.3g}")
        fig = receiver_figure(tag, j, nor_p[r], log_p[r], gain_p[r],
                              shared_scale is not None, lsp, along, dt_ms)
        if out_dir:
            p = os.path.join(out_dir,
                             f"logenv_process_{tag}_recv{j:05d}.png")
            fig.savefig(p, dpi=130)
            print(f"        wrote {p}")
        if SHOW:
            plt.show()
        plt.close(fig)


def preview(tag, data, norm_scale, lsp, shared_scale):
    """The preview PREVIEW_MODE asks for."""
    if PREVIEW_MODE == "receiver":
        return preview_receiver(tag, data, norm_scale, lsp, shared_scale)
    if PREVIEW_MODE != "shot":
        raise SystemExit(f'PREVIEW_MODE must be "shot" or "receiver", not '
                         f'{PREVIEW_MODE!r}')
    return preview_shot(tag, data, norm_scale, lsp, shared_scale)


def preview_shot(tag, data, norm_scale, lsp, shared_scale):
    """One figure per shot, drawn and shown before the next is built.

    plt.show() blocks until the window is closed, so the shots arrive one at a
    time rather than all at once - and only one gather is held in memory.
    """
    n_shots = data.shape[0]
    bad = [s for s in PREVIEW_SHOTS if not 0 <= s < n_shots]
    if bad:
        raise SystemExit(f"preview shots out of range 0..{n_shots - 1}: {bad}")

    ns = data.shape[1] if SAMPLES is None else min(SAMPLES, data.shape[1])
    out_dir = None
    if SAVE_DIR:
        out_dir = resolve(SAVE_DIR)
        os.makedirs(out_dir, exist_ok=True)

    print(f"    preview: {len(PREVIEW_SHOTS)} shots, one window at a time")
    for n, shot in enumerate(PREVIEW_SHOTS, 1):
        nor = np.asarray(data[shot], dtype=np.float32) * norm_scale
        scale = (shared_scale if shared_scale is not None
                 else logscale(nor, lsp))
        log = nor * scale
        print(f"      [{n}/{len(PREVIEW_SHOTS)}] shot {shot:>4}: "
              f"rms {np.sqrt(np.mean(nor ** 2)):.4g} "
              f"-> {np.sqrt(np.mean(log ** 2)):.4g}, "
              f"|log| max {np.abs(log).max():.4g}, "
              f"gain {scale.min():.3g}..{scale.max():.3g}")

        fig = stage_figure(tag, shot, nor[:ns], log[:ns], scale[:ns],
                           shared_scale is not None, lsp)
        if out_dir:
            p = os.path.join(out_dir, f"logenv_process_{tag}_shot{shot:04d}.png")
            fig.savefig(p, dpi=130)
            print(f"        wrote {p}")
        if SHOW:
            plt.show()
        plt.close(fig)


def write_all(tag, data, norm_scale, lsp, shared_scale, shot_order):
    """Fit the gain on shot gathers, then write it out receiver-major.

        input       (n_shots,     n_samples, n_receivers)   `norm`
        step 1      (n_shots,     n_samples, n_receivers)   `log`, shot-major
        step 2      (n_receivers, n_samples, n_shots)       `rg_train`/`rg_infer`

    Two passes, and the split is deliberate.  The gain is fitted where
    SCALE_MODE means it to be - on the shot gather, one shot at a time - so
    `log` is exactly what it always was.  The transpose is then a separate,
    purely mechanical pass over that array.

    Doing both at once is what does not work: writing out[:, :, i] one shot at
    a time scatters every value across the whole output file, so each shot
    touches every page of it.  Reading a block of receiver columns back
    instead costs one strided pass per block and writes each receiver gather
    contiguously.

    One receiver gather per leading index in the output, which is what the
    training reads.  The DAS gets two files - the receivers the streamer
    overlaps and the rest - see shared_receivers().
    """
    n_shots, n_samp, n_trace = data.shape

    # --- step 1: the gain, on shot gathers, into the shot-major `log` array
    log_rel = C.ARRAYS[tag]["log"]
    log_path = resolve(log_rel)
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    if os.path.isfile(log_path):
        print(f"    overwriting {log_rel} (was {npy_shape(log_path)})")
    log = create_memmap(log_path, data.shape)
    print(f"    -> {log_rel}  {log.shape}, {log.nbytes / 1e9:.2f} GB "
          f"(shot-major, the gain fitted per shot gather)")

    rms_shot = np.zeros(n_shots)
    lo, hi = np.inf, -np.inf
    t0 = time.time()
    for i in range(n_shots):
        nor = np.asarray(data[i], dtype=np.float32) * norm_scale
        scale = shared_scale if shared_scale is not None else logscale(nor, lsp)
        y = nor * scale
        log[i] = y
        rms_shot[i] = np.sqrt(np.mean(y.astype(np.float64) ** 2))
        lo = min(lo, float(y.min()))
        hi = max(hi, float(y.max()))
        if (i + 1) % 50 == 0 or i + 1 == n_shots:
            el = time.time() - t0
            print(f"      {i + 1}/{n_shots} shots  {el:.0f}s elapsed, "
                  f"eta {el / (i + 1) * (n_shots - i - 1):.0f}s", flush=True)
    log.flush()
    print(f"    amplitude range [{lo:.4g}, {hi:.4g}], "
          f"rms {rms_shot.mean():.4g}")

    # --- step 2: transpose into the receiver-major outputs
    inside, outside = shared_receivers(tag, n_trace)
    parts = [("rg_train", inside)]
    if outside is not None and len(outside):
        parts.append(("rg_infer", outside))

    out_info = dict(
        log=dict(output=log_rel.replace("\\", "/"),
                 shape=list(data.shape),
                 axis_order="shot, sample, receiver",
                 amplitude_min=lo, amplitude_max=hi,
                 value_range_max_abs=max(abs(lo), abs(hi)),
                 rms_mean=float(rms_shot.mean()),
                 rms_min=float(rms_shot.min()),
                 rms_max=float(rms_shot.max())),
    )
    rms = {"log_per_shot": rms_shot}
    for stage, recv in parts:
        out_rel = C.ARRAYS[tag][stage]
        out_path = resolve(out_rel)
        if os.path.isfile(out_path):
            print(f"    overwriting {out_rel} (was {npy_shape(out_path)})")
        out = create_memmap(out_path, (len(recv), n_samp, n_shots))
        print(f"    -> {out_rel}  {out.shape}, {out.nbytes / 1e9:.2f} GB")

        per_rec = np.zeros(len(recv))
        t0 = time.time()
        for a in range(0, len(recv), RG_BLOCK):
            cols = recv[a:a + RG_BLOCK]
            blk = np.asarray(log[:, :, cols], dtype=np.float32)[shot_order]
            for k in range(len(cols)):
                rg = np.ascontiguousarray(blk[:, :, k].T)   # (n_samp, n_shots)
                out[a + k] = rg
                per_rec[a + k] = np.sqrt(np.mean(rg.astype(np.float64) ** 2))
            el = time.time() - t0
            done = min(a + RG_BLOCK, len(recv))
            print(f"      {done}/{len(recv)} receivers  {el:.0f}s elapsed, "
                  f"eta {el / done * (len(recv) - done):.0f}s", flush=True)
        out.flush()
        del out
        print(f"    rms per receiver {per_rec.min():.4g}.."
              f"{per_rec.max():.4g}, mean {per_rec.mean():.4g}")
        out_info[stage] = dict(
            output=out_rel.replace("\\", "/"),
            shape=[len(recv), n_samp, n_shots],
            axis_order="receiver, sample, shot",
            receivers=[int(v) for v in recv],
            rms_mean=float(per_rec.mean()),
            rms_min=float(per_rec.min()), rms_max=float(per_rec.max()),
        )
        rms[stage] = per_rec
    del log
    return out_info, rms


def write_meta(out_rel, tag, params, info, arrays):
    stem = resolve(os.path.splitext(out_rel)[0])
    with open(stem + "_meta.json", "w", encoding="utf-8") as f:
        json.dump({"tag": tag, "parameters": params, "measured": info}, f,
                  indent=2, sort_keys=True)
    np.savez(stem + "_meta.npz", **arrays)
    print(f"    wrote {os.path.basename(stem)}_meta.json / _meta.npz")


def main():
    if SCALE_MODE not in ("per_shot", "average"):
        raise SystemExit(f"SCALE_MODE must be 'per_shot' or 'average', "
                         f"got {SCALE_MODE}")

    for tag, in_rel in JOBS:
        in_path = resolve(in_rel)
        if not os.path.isfile(in_path):
            raise SystemExit(f"not found: {in_path}")

        data = np.load(in_path, mmap_mode="r")
        n_shots, n_samp, n_trace = data.shape
        lsp = C.LOG_SCALE_PARAMS[tag]
        idx = sample_indices(n_shots)
        print(f"{tag}: {in_rel}  {data.shape}")
        print(f"    pad front {PAD_FRONT}, tail {PAD_TAIL}")
        shot_order = np.arange(n_shots)
        if SORT_WRITTEN or PREVIEW_MODE == "receiver":
            shot_order, _ = sorted_shots(n_shots)

        # --- step 1: RMS normalisation
        rms_in, rms_in_per_shot = measure_rms(data, idx)
        norm_scale = TARGET_RMS / rms_in
        print(f"    rms {rms_in:.6g} over {len(idx)} shots "
              f"-> {TARGET_RMS:g}  (scale {norm_scale:.6g})")

        # --- step 2: log-envelope gain
        shared_scale = None
        if SCALE_MODE == "average":
            shared_scale = logscale(average_gather(data, idx) * norm_scale,
                                    lsp)
            print(f"    shared gain from the mean of {len(idx)} shots, "
                  f"range [{shared_scale.min():.4g}, {shared_scale.max():.4g}]")

        probe = np.asarray(data[idx[0]], dtype=np.float32) * norm_scale
        env_median = float(np.median(envelope_1d(probe)))
        log_base = lsp["log_base"]
        print(f"    log base {log_base:g} vs median envelope {env_median:.4g} "
              f"-> base/median {log_base / env_median:.3g}")

        if PREVIEW:
            preview(tag, data, norm_scale, lsp, shared_scale)

        # Everything that goes in the sidecar except the output-side numbers is
        # known by now, so the record is written whether or not WRITE_ALL runs:
        # a preview that settled on a norm_scale and a log_base is exactly the
        # thing worth keeping, and losing it because nothing was written would
        # defeat the point.
        params = dict(
            input=in_rel.replace("\\", "/"),
            input_shape=list(data.shape),
            axis_order="receiver, sample, shot",
            shots_sorted_along_line=bool(SORT_WRITTEN),
            target_rms=TARGET_RMS,
            rms_sample_shots=RMS_SAMPLE_SHOTS,
            scale_mode=SCALE_MODE,
            pad_tail=PAD_TAIL,
            **{k: (list(v) if isinstance(v, tuple) else v)
               for k, v in lsp.items()},
        )
        info = dict(
            rms_input=rms_in,
            norm_scale=norm_scale,
            rms_shots_measured=len(idx),
            envelope_median_normalised=env_median,
            base_over_envelope_median=log_base / env_median,
            output_written=bool(WRITE_ALL),
        )
        arrays = dict(rms_input_per_shot=rms_in_per_shot,
                      rms_input_shot_index=idx)
        if shared_scale is not None:
            arrays["shared_scale"] = shared_scale

        arrays["shot_order"] = shot_order.astype(np.int32)
        if WRITE_ALL:
            out_info, rms_out = write_all(tag, data, norm_scale, lsp,
                                          shared_scale, shot_order)
            info["outputs"] = out_info
            for stage, v in rms_out.items():
                arrays[f"rms_output_per_receiver_{stage}"] = v
        else:
            print(f"    WRITE_ALL is off - nothing written")

        if WRITE_META:
            write_meta(C.ARRAYS[tag]["rg_train"], tag, params, info, arrays)

        print()


if __name__ == "__main__":
    main()
