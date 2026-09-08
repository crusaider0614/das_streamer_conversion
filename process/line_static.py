"""Put the shots back on the line and onto one time axis.

    numpy/{das,str}_data_raw.npy   ->  numpy/{das,str}_data_line.npy
                                  ->  numpy/{das,str}_data_line_meta.npz

    das  (551, 3500, 3175) @ 1 ms    ->  (551, 2000, 3175) @ 1 ms
    str  (551, 8000,   24) @ 0.5 ms  ->  (551, 2000,   24) @ 1 ms

The order is

    str   roll -> filter -> decimate 2:1 -> crop
    das   roll -> filter ->                 crop

and each step is where it is for a reason:

  * the roll is first, on the input grid, so the streamer's sub-millisecond
    shift is resolved at 0.5 ms rather than at 1 ms.
  * the filter follows it in BOTH domains, so whatever mutes this record later
    meets the same band on both sides.  It is also the anti-alias filter the
    streamer's 2:1 decimation needs - 0.5 -> 1 ms halves the Nyquist from 1000
    to 500 Hz - which the DAS does not, being already on 1 ms.
  * the crop is last.  The DAS shift reaches -22.5 ms, so a sample belonging
    at 1999 ms in the output sits past 2000 ms beforehand.

There is no mute here.  The boundary is computed, reported and written to the
sidecar, and the samples are left alone - see APPLY_MUTE for the measurement
that moved it downstream of the receiver decimation.

The survey was meant to be four passes along one line and the boat wandered:
the shots sit up to 37 m to the side of their principal axis, 13.9 m rms.  That
does not matter while the data is looked at as shot gathers, and it matters a
great deal in a receiver gather sorted by along-line shot position - which is
what makes the shot axis usable here at all, taking the 15.2 m shot interval
down to 3.3 m.  Neighbouring traces in that gather are then at inconsistent
offsets and the moveout comes out ragged rather than smooth.

What this does
--------------
For every source-receiver pair, with s the recorded source position and s0 the
same source projected perpendicularly onto the line:

    dt = ( |s0 - r| - |s - r| ) / velocity_m_s

and the trace is shifted by dt, so it holds what a source at s0 would have
recorded.  Then the mute boundary is recomputed from |s0 - r| - not from the
recorded offsets - and applied.

Recomputing the boundary is the half that is easy to get wrong.  The DAS shift
is mostly negative, up to -22.5 ms, so the arrival moves earlier while a
boundary built from the recorded positions stays put: it would end up inside
the first break.  config.MUTE_PARAMS carries the same warning for the stages
after this one.

How much there is to correct
----------------------------
d(offset)/d(lateral) vanishes at large offset, so this is entirely a question
of how close the sail line comes to the receivers:

    str    24 nodes at along 1240..1312 m, shots at -1059..+1056 m, minimum
           offset 184 m.  Shift -1.50..+0.04 ms, 0.19 ms rms - at most 3
           samples of 0.5 ms, and under half a sample almost everywhere.
    das  1058 groups at along 540..1348 m, the line crosses the fibre, minimum
           offset 0.2 m.  Shift -22.49..+7.49 ms, 1.19 ms rms - up to 22
           samples of 1 ms.  Under 50 m offset the mean correction is 3.9 ms,
           and in sorted shot order neighbouring shots differ by 7.4 ms at the
           99th percentile.  That difference is the raggedness.

So the streamer gets a correction it barely needs and the DAS gets the one it
does.  Both are written, because a stage that runs on one domain and not the
other is a trap later.

Two things this is not
----------------------
It is a static shift, not a redatuming.  The offset error is a distance, and
turning it into a time needs a velocity; the correction is exact for a straight
ray at LINE_STATIC_PARAMS["velocity_m_s"] and scales as 1/v for anything else.
The water velocity is the right choice because the near-offset traces, where
the correction is big enough to matter, are the ones the direct arrival
dominates.  A 10 % velocity error leaves at most 2.2 ms on the worst trace,
against a 30 ms mute taper.

And it does not reorder the array.  Shots stay in their recorded order, which
is what every other stage and every sidecar assumes; the sorted order lives in
data/pohang_shore/shot_order.txt and process/shot_geometry.py, and
process/receiver_gather.py sorts at display time.

The shift is per trace, so shot-major and receiver-major passes give the same
answer; this runs shot-major because that is how the arrays are laid out.  The
preview receiver gathers are accumulated during that one pass rather than read
back with a strided slice, which on the 24 GB DAS array would touch every page.

Edit the settings block below, then run from the repository root:

    python -m process.line_static
"""

import os
import time

import matplotlib
import numpy as np

import scipy.fft as sf

import config as C
import process.logenv_process as L
import process.shot_geometry as G
import utils.mute as M
from utils.data import create_memmap, get_project_root, npy_shape
from utils.process import f_filter

import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------- settings --

# Domains to run, in order.  Both, normally - see the docstring.
DOMAINS = ("str", "das")

# The stage to read.  "raw" is where this belongs: it is a geometry
# correction, so nothing should have been done to the samples first.
INPUT_STAGE = "raw"

# The band-pass to apply, as a key of config.FILTER_PARAMS, per domain.
# None means no filter.
#
# Both domains are filtered here, and that is the point: the filter has to sit
# on the same side of the mute in both, or the mute boundary meets a different
# band on each side and the two records are no longer comparable at their
# first break.  So the order is filter -> crop -> mute everywhere.
#
# It also happens to be what the streamer needs anyway.  Going 0.5 -> 1 ms
# halves its Nyquist from 1000 to 500 Hz, so everything above 500 would fold;
# "str_bl" puts the mask 843 dB down there, which is an anti-alias filter with
# room to spare.  The DAS is not decimated in time and so needs no anti-alias
# filter - it is filtered because of the ordering argument above, not that one.
#
# Note this makes `line` band-passed on both sides, so the `freq` stage would
# be filtering a second time on this route - see config.NORM_FILTER_CHAIN.
FILTER_KEY = {"str": "str_bl", "das": "das"}

# Zeros prepended before the band-pass, then dropped.  f_filtering is
# circular, so the wrap has to land somewhere that gets thrown away; the same
# device as band_limit.PAD_FRONT, and it leaves the time origin unchanged.
FILTER_PAD_FRONT = 2000

# Apply the direct-arrival mute to the samples.  OFF, deliberately.
#
# The boundary is still computed, reported and written to the sidecar as
# `mute_boundary_ms` - only the multiplication is skipped, so the stage that
# ends up muting can do it on its own grid.
#
# Why it moved.  At a fixed time the mute edge is a STEP along the receiver
# axis: some channels are muted and their neighbours are not.  A step is not
# band-limited in x, so any filter with reach along that axis rings across it,
# and what leaks in comes from the direct arrival - the loudest thing in the
# record.  Measured on the DAS at 12:1, as rms above the boundary over the rms
# in the 30 ms below it, 6 shots x 264 output receivers:
#
#     boxcar 3 / 6 / 12     0.00 %                 reach 0 samples
#     k-sinc                5.36 % mean, 48.3 % p99  reach median 86,
#                                                    max 435 samples
#
# The boundary varies by only 1.538 ms median inside one 12-channel block, so
# the compact boxcars mix nothing a 2-sample guard does not cover.  The k-sinc
# is the filter worth using - it is the only one that cannot alias - and its
# support is the whole array, so muting before it is the wrong order.  Re-
# zeroing afterwards does not undo the ringing that landed BELOW the boundary.
#
# So `line` carries an unmuted record and the mute belongs after the receiver
# decimation - process/decimate_receiver.py, REMUTE.
APPLY_MUTE = False

# Write the output arrays and their sidecars.  Off runs the measurement and
# the preview and touches nothing.
WRITE_ALL = True

# Draw a few receiver gathers, before and after, with both mute boundaries on
# them.  Costs nothing extra: the columns are collected during the write pass.
PREVIEW = True

# Trace indices to draw, per domain.  None picks three evenly spaced.
PREVIEW_TRACES = {"str": (0, 11, 23), "das": None}
PREVIEW_N = 3

# Time window of the preview, in milliseconds.  The correction lives at the
# top of the record.
VIEW_MS = (0.0, 1200.0)

# "clip" scales by a percentile of |amplitude|; "log" applies
# utils.process.calculate_logscale first, which is what makes the deep part of
# a receiver gather visible at all.
DISPLAY = "log"
CLIP_PERC = 99.0
LOG_PARAMS_KEY = {"str": "str_pre", "das": "das"}

# Show interactively, and optionally save.  A falsy SAVE_DIR skips saving.
SHOW = True
SAVE_DIR = C.DATA_DIR
BACKENDS = ("QtAgg", "Qt5Agg", "TkAgg", "MacOSX")

# ----------------------------------------------------------------- derived --

_L = C.LINE_STATIC_PARAMS
SHIFT_V = _L["velocity_m_s"]
MODE = _L["mode"]
PAD = int(_L["pad_samples"])
TAPER = int(_L["taper_samples"])
OUT_DT_US = int(_L["out_dt_us"])
OUT_SAMPLES = int(_L["out_samples"])
OUT_DT_MS = OUT_DT_US / 1000.0

_M = C.MUTE_PARAMS
DEPTH = _M["depth_m"]

# Single precision through the transforms, which is what makes this stage
# tolerable to run.  The DAS shot is 3500 x 3175, every step along time is a
# transform pair, and the work is memory-bandwidth bound rather than compute
# bound - scipy's `workers` gives nothing (measured: 0.89 s at 1 thread, 0.90 s
# at 24), while halving the bytes gives everything.
#
# Against the previous numpy float64 path, same order and same padding:
# 1.900 s -> 0.560 s per DAS shot, 17.4 min -> 5.1 min for 551, with an rms
# difference of 0.0001 % of the record.  Two things pay: complex64 instead of
# complex128, and rfft instead of the full complex fft utils.process.f_filtering
# uses.  The arrays are float32 on disk and written back as float32, so there
# was never any float64 to preserve.
WORK_DTYPE = np.float32
WORK_CDTYPE = np.complex64

# ---------------------------------------------------------------------------


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(get_project_root(), path)


def show_figures():
    """Swap an interactive backend in if the current one cannot draw."""
    if not SHOW:
        return False
    if matplotlib.get_backend().lower() not in ("agg", "pdf", "ps", "svg"):
        return True
    for name in BACKENDS:
        try:
            matplotlib.use(name, force=True)
        except Exception:
            continue
        print(f"  backend: {matplotlib.get_backend()}")
        return True
    print(f"  warning: none of {', '.join(BACKENDS)} is installed - "
          f"set SAVE_DIR and look at the files")
    return False


# ------------------------------------------------------------- the geometry --


def line_frame(src):
    """The fitted line, and the sources projected onto it.

    The line is the principal axis of the shot positions through their
    centroid, the same one process/shot_geometry.py sorts against and writes
    into shot_order.txt - taken from there rather than refitted so the two
    cannot drift apart.  A straight line is the right thing to fit: a degree-5
    polynomial reduces the residual by 4 % and its arc length matches the
    straight projection to 0.02 %.
    """
    along, perp, u = G.frame(src)
    centroid = src.mean(axis=0)
    proj = centroid + along[:, None] * u[None, :]
    return proj, along, perp, u, centroid


def offsets(src, rec):
    """Source-receiver distance, shape (n_shots, n_traces).

    Includes MUTE_PARAMS["depth_m"] so this stage and the mute measure the
    same distance.  At the current depth of 0 that is a no-op, and it cancels
    almost exactly in the difference below in any case.
    """
    h = np.hypot(src[:, None, 0] - rec[None, :, 0],
                 src[:, None, 1] - rec[None, :, 1])
    return np.sqrt(h ** 2 + DEPTH ** 2), h


def shift_ms(src, proj, rec):
    """(shift, recorded offset, projected offset).  Positive shift is later."""
    off_a, h_a = offsets(src, rec)
    off_i, h_i = offsets(proj, rec)
    return (off_i - off_a) / SHIFT_V * 1e3, off_a, off_i, h_a, h_i


def mute_boundary_ms(off_i):
    """The boundary MUTE_PARAMS asks for, on the projected offsets."""
    t = off_i / _M["velocity_m_s"] * 1e3 - _M["lead_ms"]
    return np.maximum(t, 0.0)


# --------------------------------------------------------------- the shift --


def tail_pad(g, pad, taper):
    """Record with a faded tail and then zeros, for the circular shift.

    The last sample is extended and faded to zero by a raised cosine so the
    join is continuous, then zeros out to `pad`.  No sample of the record is
    modified - the fade lives entirely in the pad, which is cropped off after
    the shift.
    """
    nt, ntr = g.shape
    y = np.zeros((nt + pad, ntr), dtype=WORK_DTYPE)
    y[:nt] = g
    n = min(taper, pad)
    if n > 0:
        w = (0.5 * (1.0 + np.cos(np.pi * np.arange(1, n + 1) / n))
             ).astype(WORK_DTYPE)
        y[nt:nt + n] = g[nt - 1:nt] * w[:, None]
    return y


def fourier_shift(g, dt_ms, d_ms):
    """Fractional delay per trace, by a phase ramp on the rfft.

    exp(-2 pi i f tau) delays by tau, so a positive d_ms moves the trace
    later.  The ramp is a circular shift on the padded record: content pushed
    off either end lands in the pad and is cropped, and what wraps back in is
    the pad's zeros.  Both directions are covered by a tail pad alone, which
    is why there is none at the front.
    """
    nt = g.shape[0]
    y = tail_pad(g, PAD, TAPER)
    n = y.shape[0]
    f = sf.rfftfreq(n, dt_ms * 1e-3).astype(WORK_DTYPE)
    ph = np.exp((-2j * np.pi * 1e-3)
                * (f[:, None] * d_ms[None, :].astype(WORK_DTYPE)))
    F = sf.rfft(y, axis=0)
    F *= ph.astype(WORK_CDTYPE)
    return sf.irfft(F, n=n, axis=0)[:nt]


def integer_shift(g, dt_ms, d_ms):
    """Whole-sample delay per trace, zero-filled rather than wrapped."""
    out = np.zeros_like(g)
    k = np.round(d_ms / dt_ms).astype(int)
    nt = g.shape[0]
    for j, s in enumerate(k):
        if s == 0:
            out[:, j] = g[:, j]
        elif 0 < s < nt:
            out[s:, j] = g[:nt - s, j]
        elif -nt < s < 0:
            out[:nt + s, j] = g[-s:, j]
    return out


def build_mask(tag, nt, dt_s):
    """The anti-alias mask for this domain, or None.

    Built for the padded length, because f_filtering transforms the whole
    trace at once and the mask has to match what it is handed.
    """
    key = FILTER_KEY.get(tag)
    if key is None:
        return None, None
    p = C.FILTER_PARAMS[key]
    m = np.ones(nt)
    if p.get("lowpass"):
        m *= f_filter(nt, dt_s, p["lp_f_cut"], p["lp_order"], p["lp_decay"],
                      is_lowpass=True)
    if p.get("highpass"):
        m *= f_filter(nt, dt_s, p["hp_f_cut"], p["hp_order"], p["hp_decay"],
                      is_lowpass=False)
    return m, key


def report_mask(tag, key, m, nt, dt_s, decim):
    """The band, and what it leaves at the frequency the decimation folds."""
    if m is None:
        print(f"  no band-pass for {tag}")
        return
    p = C.FILTER_PARAMS[key]
    print(f"  band-pass FILTER_PARAMS[{key!r}]: "
          f"{p['hp_f_cut']:g}-{p['lp_f_cut']:g} Hz "
          f"(order/decay {p['hp_order']:g}/{p['hp_decay']:g} low-cut, "
          f"{p['lp_order']:g}/{p['lp_decay']:g} low-pass)")
    if decim == 1:
        print(f"    decimation is 1:1, so this is the band only - "
              f"nothing folds")
        return
    f = np.abs(np.fft.fftfreq(nt, dt_s))
    f_new_nyq = 0.5 / (dt_s * decim)
    at = m[np.argmin(np.abs(f - f_new_nyq))]
    worst = m[f > f_new_nyq].max()
    print(f"    also the anti-alias filter for {decim}:1 - at the new "
          f"{f_new_nyq:.0f} Hz Nyquist the mask is "
          f"{20 * np.log10(max(at, 1e-300)):.0f} dB, and its worst gain "
          f"anywhere above it is {20 * np.log10(max(worst, 1e-300)):.0f} dB")


def half_mask(mask):
    """The rfft half of a full fftfreq mask, or None.

    f_filter builds its mask on abs(fftfreq(n, dt)), so it is symmetric and
    the first n // 2 + 1 entries are exactly what rfft needs.
    """
    if mask is None:
        return None
    return mask[:len(mask) // 2 + 1].astype(WORK_DTYPE)


def bandpass(g, mask):
    """The band-pass behind a front pad of zeros, which is then dropped.

    The transform is circular, so the wrap has to land in the pad rather than
    at the end of the record.  `mask` is the rfft half from half_mask(), built
    for the padded length.
    """
    if mask is None:
        return g
    nt = g.shape[0]
    n = nt + FILTER_PAD_FRONT
    y = np.zeros((n, g.shape[1]), dtype=WORK_DTYPE)
    y[FILTER_PAD_FRONT:] = g
    F = sf.rfft(y, axis=0)
    F *= mask[:, None]
    return sf.irfft(F, n=n, axis=0)[FILTER_PAD_FRONT:]


def decimate(g, decim):
    """Subsample the time axis.

    Subsampling rather than block-averaging: bandpass() has already set the
    band, and averaging would put a second low-pass on top of it.
    """
    return g if decim == 1 else g[::decim]


def apply_shift(g, dt_ms, d_ms):
    if MODE == "none":
        return g
    if MODE == "fourier":
        return fourier_shift(g, dt_ms, d_ms)
    if MODE == "integer":
        return integer_shift(g, dt_ms, d_ms)
    raise SystemExit(f'LINE_STATIC_PARAMS["mode"] = {MODE!r}; expected '
                     f'"fourier", "integer" or "none"')


# --------------------------------------------------------------- reporting --


def report_geometry(along, perp, off_a, off_i, d_ms, t_mute, dt_ms, nt):
    """Everything about the correction that can be known before reading data."""
    n = np.abs(d_ms) / dt_ms
    print(f"  line: {len(along)} shots, along {along.min():.0f}.."
          f"{along.max():.0f} m ({along.max() - along.min():.0f} m span)")
    print(f"    perpendicular offset {perp.min():+.1f}..{perp.max():+.1f} m, "
          f"{np.sqrt((perp ** 2).mean()):.1f} m rms")
    print(f"  offsets: recorded {off_a.min():.1f}..{off_a.max():.0f} m, "
          f"projected {off_i.min():.1f}..{off_i.max():.0f} m")
    print(f"    change {np.abs(off_i - off_a).max():.2f} m at worst")
    print(f"  shift at {SHIFT_V:g} m/s, mode {MODE!r}: "
          f"{d_ms.min():+.3f}..{d_ms.max():+.3f} ms, "
          f"{np.sqrt((d_ms ** 2).mean()):.3f} ms rms")
    print(f"    in samples of {dt_ms:g} ms: {n.max():.2f} at worst, "
          f"{(n < 0.5).mean() * 100:.1f} % under half a sample")
    if MODE == "integer":
        lost = np.abs(d_ms - np.round(d_ms / dt_ms) * dt_ms)
        print(f"    rounding discards {lost.mean():.3f} ms on average, "
              f"{lost.max():.3f} ms at worst; "
              f"{(np.round(d_ms / dt_ms) == 0).mean() * 100:.1f} % of traces "
              f"round to no shift at all")
    if np.abs(d_ms).max() / dt_ms >= PAD:
        raise SystemExit(f"pad_samples is {PAD} but the largest shift is "
                         f"{np.abs(d_ms).max() / dt_ms:.0f} samples")
    # what an unrevised mute would now do: the arrival has moved, the boundary
    # has not
    t_old = mute_boundary_ms(off_a)
    err = t_old - t_mute
    print(f"  mute: boundary {t_mute.min():.0f}..{t_mute.max():.0f} ms of a "
          f"{nt * dt_ms:.0f} ms record")
    print(f"    {M.summary(t_mute, nt, dt_ms).strip()}")
    print(f"    recomputing it moved the boundary by {err.min():+.2f}.."
          f"{err.max():+.2f} ms; had it not been recomputed it would sit up to "
          f"{max(err.max(), 0.0):.1f} ms inside the shifted first break")


def report_sorted_jump(d_ms, order):
    """How much of the shot-to-shot raggedness the shift takes out.

    In sorted order, neighbouring traces at one receiver should differ only by
    the moveout over 3.3 m.  Whatever the shift removes is offset error that
    was masquerading as moveout.
    """
    j = np.abs(np.diff(d_ms[order], axis=0))
    print(f"    raggedness removed, neighbouring shots in sorted order: "
          f"median {np.median(j):.4f} ms, p99 {np.percentile(j, 99):.3f} ms, "
          f"max {j.max():.3f} ms")


# ---------------------------------------------------------------- preview --


def gain(panel, tag):
    """Display scaling: the envelope gain, or the panel untouched.

    process.logenv_process.logscale returns the gain, not the gained data, and
    fits it on a tail-padded copy because the envelope is circular.  The axes
    here are (time, shot), so smooth_sigma's second number smooths across
    shots - the right thing, since neighbouring shots see nearly the same
    wavefield.  Reused rather than reimplemented so the display cannot drift
    from process/receiver_gather.py's.
    """
    if DISPLAY == "clip":
        return panel
    if DISPLAY != "log":
        raise SystemExit(f'DISPLAY must be "clip" or "log", not {DISPLAY!r}')
    p = dict(C.LOG_SCALE_PARAMS[LOG_PARAMS_KEY[tag]])
    return panel * L.logscale(panel, p)


def receiver_figure(tag, traces, before, after, t_old, t_new, dt_ms):
    """One row per receiver: the gather before and after, boundaries drawn.

    The x axis is sorted shot index, so this is the gather the sorting was for.
    Both boundaries are drawn on both panels - the one from the recorded
    offsets and the one from the projected offsets - because the gap between
    them is what the stage is about.
    """
    i0, i1 = (int(round(v / dt_ms)) for v in VIEW_MS)
    n_rows = len(traces)
    fig, axes = plt.subplots(n_rows, 2, figsize=(13.5, 3.6 * n_rows + 1.0),
                             squeeze=False)
    x = np.arange(before.shape[2])
    for r, j in enumerate(traces):
        panels = ((before[r], "decimated and cropped, not shifted or muted"),
                  (after[r], "projected onto the line, shifted and muted"))
        for c, (img, name) in enumerate(panels):
            v = gain(np.asarray(img[i0:i1], dtype=np.float64), tag)
            clip = np.percentile(np.abs(v), CLIP_PERC)
            clip = clip if clip > 0 else 1.0
            ax = axes[r][c]
            ax.imshow(v, cmap="seismic", vmin=-clip, vmax=clip, aspect="auto",
                      interpolation="nearest",
                      extent=[0, len(x), i1 * dt_ms, i0 * dt_ms])
            ax.plot(x + 0.5, t_old[r], color="k", lw=0.8, ls="--",
                    label="boundary from the recorded offsets")
            ax.plot(x + 0.5, t_new[r], color="lime", lw=1.0,
                    label="boundary from the projected offsets")
            ax.set_ylim(i1 * dt_ms, i0 * dt_ms)
            ax.set_title(f"trace {j}  -  {name}", fontsize=9)
            ax.set_xlabel("shot, sorted along the line")
            if c == 0:
                ax.set_ylabel("time [ms]")
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc="lower left")
    fig.suptitle(f"{tag} {INPUT_STAGE} -> line   "
                 f"shift at {SHIFT_V:g} m/s, {MODE}, mute at "
                 f"{_M['velocity_m_s']:g} m/s / lead {_M['lead_ms']:g} ms / "
                 f"taper {_M['taper_ms']:g} ms   "
                 f"({'log gain' if DISPLAY == 'log' else f'{CLIP_PERC:g}th pct clip'})")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    return fig


def pick_traces(tag, n_traces):
    want = PREVIEW_TRACES.get(tag)
    if want is None:
        want = np.linspace(0, n_traces - 1, PREVIEW_N).round().astype(int)
    bad = [int(j) for j in want if not 0 <= j < n_traces]
    if bad:
        raise SystemExit(f"{tag}: preview traces out of range {bad} "
                         f"(array has {n_traces})")
    return [int(j) for j in want]


# ------------------------------------------------------------------- stage --


def run(tag):
    in_path = resolve(C.ARRAYS[tag][INPUT_STAGE])
    if not os.path.isfile(in_path):
        raise SystemExit(f"not found: {in_path}")
    data = np.load(in_path, mmap_mode="r")
    n_shots, nt, n_traces = data.shape
    dt_us = C.STAGE_DT_US[tag][INPUT_STAGE]
    dt_ms = dt_us / 1000.0

    print(f"\n{'=' * 76}\n{tag}  {C.ARRAYS[tag][INPUT_STAGE]}")
    print(f"  {n_shots} shots x {nt} samples @ {dt_ms:g} ms x {n_traces} traces")

    # --- the time axis, decided before anything reads a sample -------------
    if OUT_DT_US % dt_us:
        raise SystemExit(f"{tag}: {OUT_DT_US} us output does not divide by the "
                         f"{dt_us} us input")
    decim = OUT_DT_US // dt_us
    nt_dec = nt // decim
    if OUT_SAMPLES > nt_dec:
        raise SystemExit(f"{tag}: {OUT_SAMPLES} samples asked for but the "
                         f"record is only {nt_dec} after {decim}:1 decimation")
    print(f"  time axis: {decim}:1 -> {nt_dec} @ {OUT_DT_MS:g} ms, "
          f"cropped to {OUT_SAMPLES} ({OUT_SAMPLES * OUT_DT_MS:g} ms)")
    mask, key = build_mask(tag, nt + FILTER_PAD_FRONT, dt_ms * 1e-3)
    report_mask(tag, key, mask, nt + FILTER_PAD_FRONT, dt_ms * 1e-3, decim)
    mask = half_mask(mask)
    print(f"  transforms in {np.dtype(WORK_DTYPE).name} via scipy.fft "
          f"(see WORK_DTYPE)")

    src, rec = M.geometry(n_traces)
    if len(src) != n_shots:
        raise SystemExit(f"{len(src)} source positions for {n_shots} shots")
    proj, along, perp, u, centroid = line_frame(src)
    d_ms, off_a, off_i, h_a, h_i = shift_ms(src, proj, rec)
    t_new = mute_boundary_ms(off_i)
    t_old = mute_boundary_ms(off_a)

    # The mute is built on the OUTPUT grid, which is the point of the
    # reordering: a boundary applied on the fine grid and then decimated has
    # its onset resampled, and that corner rings.
    report_geometry(along, perp, off_a, off_i, d_ms, t_new, OUT_DT_MS,
                    OUT_SAMPLES)
    order = np.argsort(along)
    report_sorted_jump(d_ms, order)

    traces = pick_traces(tag, n_traces) if PREVIEW else []
    if traces:
        print(f"  preview traces {traces}")
    shape = (len(traces), OUT_SAMPLES, n_shots)
    before = np.zeros(shape) if traces else None
    after = np.zeros(shape) if traces else None

    out_shape = (n_shots, OUT_SAMPLES, n_traces)
    out = None
    if WRITE_ALL:
        out_path = resolve(C.ARRAYS[tag]["line"])
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        if os.path.isfile(out_path):
            print(f"  overwriting {C.ARRAYS[tag]['line']} "
                  f"(was {npy_shape(out_path)})")
        out = create_memmap(out_path, out_shape)
        print(f"  output {out.shape}, {out.nbytes / 1e9:.2f} GB")
    else:
        print(f"  WRITE_ALL is off - {C.ARRAYS[tag]['line']} not touched")

    # roll -> filter -> decimate -> crop.  No mute; see APPLY_MUTE.
    #
    # The roll runs first, on the input grid, so the streamer's shift is
    # resolved at 0.5 ms rather than 1 ms.  The filter follows it, in both
    # domains, so whatever mutes this record later meets the same band on both
    # sides.  The crop is last because the DAS shift reaches -22.5 ms, so a
    # sample belonging at 1999 ms in the output sits past 2000 ms beforehand.
    t0 = time.time()
    kept = np.zeros(n_shots)
    keep = slice(0, OUT_SAMPLES)
    for i in range(n_shots):
        g = np.asarray(data[i], dtype=WORK_DTYPE)
        y = apply_shift(g, dt_ms, d_ms[i])
        y = decimate(bandpass(y, mask), decim)[keep]
        if APPLY_MUTE:
            y = y * M.weights(t_new[i], OUT_SAMPLES, OUT_DT_MS)
        # what a mute at t_new would take, measured but not applied
        kept[i] = (np.sum((y * M.weights(t_new[i], OUT_SAMPLES,
                                         OUT_DT_MS)) ** 2)
                   / max(np.sum(y ** 2), 1e-30))
        if out is not None:
            out[i] = y.astype(np.float32)
        if traces:
            # the same chain without the roll, on the preview columns only -
            # filtering three traces rather than all of them
            ref = decimate(bandpass(g[:, traces], mask), decim)[keep]
            for r in range(len(traces)):
                before[r, :, i] = ref[:, r]
                after[r, :, i] = y[:, traces[r]]
        if (i + 1) % 50 == 0 or i + 1 == n_shots:
            el = time.time() - t0
            print(f"    {i + 1}/{n_shots} shots  {el:.0f}s elapsed, "
                  f"eta {el / (i + 1) * (n_shots - i - 1):.0f}s")
    print(f"  a mute at this boundary would keep {100 * kept.mean():.1f} % of "
          f"the energy on average, {100 * kept.min():.1f} % at worst"
          + ("" if APPLY_MUTE else " - not applied, the boundary is in the "
                                   "sidecar for the stage that does"))

    if out is not None:
        out.flush()
        del out
        print(f"  wrote {resolve(C.ARRAYS[tag]['line'])}")
        meta = resolve(C.META[f"{tag}_line"])
        np.savez(
            meta,
            source_xy=src, source_xy_projected=proj, receiver_xy=rec,
            along_m=along, perp_m=perp,
            line_centroid=centroid, line_direction=u,
            shift_ms=d_ms.astype(np.float32),
            offset_recorded_m=h_a.astype(np.float32),
            offset_projected_m=h_i.astype(np.float32),
            mute_boundary_ms=t_new.astype(np.float32),
            sorted_order=order.astype(np.int32),
            input_stage=INPUT_STAGE, input_dt_us=dt_us,
            dt_us=OUT_DT_US, n_samples=OUT_SAMPLES, decimation=decim,
            antialias_filter=("" if key is None else key),
            shift_velocity_m_s=SHIFT_V, shift_mode=MODE,
            pad_samples=PAD, taper_samples=TAPER,
            mute_velocity_m_s=_M["velocity_m_s"], mute_depth_m=DEPTH,
            mute_lead_ms=_M["lead_ms"], mute_taper_ms=_M["taper_ms"],
            # whether mute_boundary_ms was applied to these samples, or is
            # only recorded for a later stage to apply
            mute_applied=APPLY_MUTE,
        )
        print(f"  wrote {meta}")
        print(f"  NOTE: source_xy_projected is the geometry this output "
              f"holds.  Anything downstream that mutes again must use it, "
              f"not source_xy - see config.MUTE_PARAMS.")

    if not traces:
        return None
    return receiver_figure(tag, traces, before[:, :, order],
                           after[:, :, order], t_old[order].T[traces],
                           t_new[order].T[traces], OUT_DT_MS)


def main():
    show = show_figures()
    bad = [t for t in DOMAINS if t not in C.ARRAYS]
    if bad:
        raise SystemExit(f"unknown domains {bad}; expected from "
                         f"{sorted(C.ARRAYS)}")
    print(f"line static: {INPUT_STAGE} -> line for {', '.join(DOMAINS)}")
    print(f"  output {OUT_SAMPLES} samples @ {OUT_DT_MS:g} ms "
          f"({OUT_SAMPLES * OUT_DT_MS:g} ms), both domains")
    print(f"  shift {SHIFT_V:g} m/s, mode {MODE!r}, pad {PAD} + taper {TAPER}")
    print(f"  mute  {_M['velocity_m_s']:g} m/s, depth {DEPTH:g} m, "
          f"lead {_M['lead_ms']:g} ms, taper {_M['taper_ms']:g} ms")

    out_dir = resolve(SAVE_DIR) if SAVE_DIR else None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    for tag in DOMAINS:
        fig = run(tag)
        if fig is None:
            continue
        if out_dir:
            p = os.path.join(out_dir, f"line_static_{tag}.png")
            fig.savefig(p, dpi=130)
            print(f"  wrote {p}")
        if show:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
