"""Preprocessing parameters for the pohang_shore DAS / streamer pipeline.

One place for the numbers that describe the survey and the numbers that decide
how it gets processed, so a stage can be re-run - or an output traced back -
without hunting through the scripts.  Modelled on `config.py` in the
4d-noise-attenuation project.

What lives here and what does not
---------------------------------
Here: the acquisition geometry, where each stage's array sits, and the
parameters that change the samples - filter corners, the normalisation target,
the envelope-gain settings.  These are per-domain dicts keyed "das" / "str",
because the two sides genuinely want different values.

Not here: how a script is being run.  Which shots to preview, whether to show a
figure, whether to write the full output - those stay in the script's own
settings block, since they change from run to run and say nothing about the
data.

The pipeline
------------
    raw    SEG-Y concatenated, nothing done to the samples
    bl     streamer only: band-limited to 220 Hz for the interpolation
    intp   streamer only: curvelet trace interpolation, 24 -> 93
    geom   both on the common 1 ms / 3500 sample grid
    freq   band-passed  (process/freq_filter.py)
    norm   RMS normalised, no envelope gain  (process/rms_normalize.py)
    log    RMS normalised and envelope scaled  (process/logenv_process.py)

`norm` is what module/dataset_pohang_shore.py loads.  `log` is the same thing
with calculate_logscale applied as well - kept because the option is still
there, not because anything reads it now.  Both sit in the pohang_shore root
rather than in numpy/.

Two routes reach `geom` on the streamer side, and the current array came from
the first:

  * the curvelet cascade, band_limit (220 Hz) -> interpolate_traces ->
    decimate_intp.  Note the 220 Hz: the cascade needs its input unaliased at
    the 3.12 m group interval, so this route cannot deliver the 320 Hz that
    FILTER_PARAMS["str"] asks for.
  * process/interpolate_kernel.py, a windowed-sinc (or linear) kernel straight
    from `raw`, which does reach 320 Hz.

The cascade leaves a period-4 amplitude step at the recorded positions -
+4.2 %, because the recorded traces are mutually incoherent below about 55 Hz
(coherence 0.13 against 0.78 above 75 Hz) and the solver reproduces that noise
only where it has data to fit.  decimate_intp's DEBIAS_TRACE_CLASSES takes it
down to +1.3 %.
"""

import os

# ------------------------------------------------------------------ layout --

# Relative to the project root; the scripts resolve them with
# utils.data.get_project_root().
DATA_DIR = os.path.join("data", "pohang_shore")
NPY_DIR = os.path.join(DATA_DIR, "numpy")

# Every stage's array, by domain.  A name here does not promise the file
# exists: `str_bl` was deleted once the interpolation was done, and the `freq`
# and `log` entries appear as their stages are run.
ARRAYS = {
    "das": {
        "raw": os.path.join(NPY_DIR, "das_data_raw.npy"),
        "geom": os.path.join(NPY_DIR, "das_data_geom.npy"),
        "freq": os.path.join(NPY_DIR, "das_data_freq.npy"),
        "norm": os.path.join(DATA_DIR, "das_data_norm.npy"),
        "log": os.path.join(DATA_DIR, "das_data_log.npy"),
    },
    "str": {
        "raw": os.path.join(NPY_DIR, "str_data_raw.npy"),
        "bl": os.path.join(NPY_DIR, "str_data_bl.npy"),
        "intp": os.path.join(NPY_DIR, "str_data_intp.npy"),
        "geom": os.path.join(NPY_DIR, "str_data_geom.npy"),
        "freq": os.path.join(NPY_DIR, "str_data_freq.npy"),
        "norm": os.path.join(DATA_DIR, "str_data_norm.npy"),
        "log": os.path.join(DATA_DIR, "str_data_log.npy"),
    },
}

# Companion metadata written next to `raw` and `geom` by the scripts that make
# them: receiver and source coordinates, and the DAS channel grouping.
META = {
    "das_raw": os.path.join(NPY_DIR, "das_data_raw_meta.npz"),
    "das_geom": os.path.join(NPY_DIR, "das_data_geom_meta.npz"),
    "str_raw": os.path.join(NPY_DIR, "str_data_raw_meta.npz"),
}

# ------------------------------------------------------------- acquisition --

N_SHOTS = 551

# The common grid both domains are brought onto at the `geom` stage.
DT_US = 1000
N_SAMPLES = 3500

# DAS.  The SEG-Y header coordinates were not surveyed - they were filled in
# from the streamer node positions - so the real acquisition interval is the
# uniform 0.25 m below, not anything measurable from the headers.  Grouping
# three channels gives exactly 0.75 m with no drift.
DAS_CHANNEL_INTERVAL_M = 0.25
DAS_RAW_CHANNELS = 3175
DAS_STRIDE = 3
DAS_ANCHOR_CHANNEL = 146          # channel of the first streamer node
DAS_TRACES = 1058

# Streamer.  24 recorded nodes 12 or 13 DAS channels apart (3.00 / 3.25 m),
# curvelet-interpolated 4x to 93 traces spanning the same 71.25 m.
STR_NODES = 24
STR_TRACES = 93
STR_TRACE_INTERVAL_M = 0.7745

# Trace spacing at the `geom` stage, by domain - what the f-k axes and the
# alias-onset reference lines are drawn against.
DX_M = {"das": DAS_STRIDE * DAS_CHANNEL_INTERVAL_M, "str": STR_TRACE_INTERVAL_M}

# Sample interval of each stage, in microseconds.  The streamer keeps the
# SEG-Y's own 0.5 ms grid until decimate_intp brings it onto the common 1 ms
# one; the DAS is on 1 ms from the start.  Written down because getting it
# wrong is silent - every frequency axis is fftfreq(nt, dt), so a wrong value
# rescales every corner without complaining.
STAGE_DT_US = {
    "das": {"raw": 1000, "geom": 1000, "freq": 1000, "norm": 1000,
            "log": 1000},
    "str": {"raw": 500, "bl": 500, "intp": 500, "geom": 1000, "freq": 1000,
            "norm": 1000, "log": 1000},
}

# ----------------------------------------------------- band-pass (`freq`) --

# process/freq_filter.py, per domain.  The mask is
#     m(f) = (1 + 2 ** (order * x)) ** (-decay / order)
# with x = f - f_cut for the low-pass and x = f_cut - f for the low-cut, so
# decay == order puts -6 dB at f_cut and the roll-off far from the corner is
# 6 * decay dB per Hz.
#
# The low-pass is not an anti-alias measure at these trace spacings: at 0.75 m
# the 1500 m/s water arrival would only fold above 1000 Hz, twice the 500 Hz
# temporal Nyquist.  Whatever is cut is cut because it is noise.
#
# pad_front: zeros prepended so the circular filter's wrap lands in the pad,
# which is then dropped.  Length and time origin are preserved.
FILTER_PARAMS = {
    "das": dict(
        highpass=True, hp_f_cut=5.0, hp_order=1.0, hp_decay=1.0,
        lowpass=True, lp_f_cut=300.0, lp_order=0.5, lp_decay=0.5,
        zero_dc=False, pad_front=2000,
    ),
    # Low-pass only, no low-cut.  Applied on the 0.5 ms grid - by
    # process/band_limit.py on the curvelet route, by
    # process/interpolate_kernel.py on the sinc route - where it is also the
    # anti-alias filter for the decimation to 1 ms.
    #
    # 300 Hz is the interpolation limit, not a preference: the recorded nodes
    # are 3.00 m apart at 14 of the 23 intervals and 3.25 m at the other 9, and
    # the widest gap decides.  The direct arrival moves out 3.1 samples of
    # 0.5 ms per trace - 1950 m/s - so it folds above v / (2 dx) = 300 Hz, and
    # above that the curvelet solve fits the fold-over instead of the true dip.
    # (The 220 Hz this used to be assumed a 1500 m/s water arrival, which the
    # data does not show.)
    #
    # Leave order and decay at 0.5.  They set the transition width, and a
    # sharper corner costs ringing quickly: at order 1 the impulse response
    # tail above -60 dB goes from 143 ms to 236 ms, at order 2 to 376 ms.
    # `raw -> bl`, before the interpolation, on the 0.5 ms grid.  A separate
    # entry from "str" because the two stages want opposite things: this one is
    # a low-pass and nothing else, "str" below is a low-cut and nothing else.
    #
    # 300 Hz is the interpolation limit, not a preference.  The recorded nodes
    # are 3.00 m apart at 14 of the 23 intervals and 3.25 m at the other 9, and
    # the widest gap decides.  The direct arrival moves out 3.1 samples of
    # 0.5 ms per trace - 1950 m/s - so it folds above v / (2 dx) = 300 Hz, and
    # above that the curvelet solve fits the fold-over instead of the true dip.
    # It doubles as the anti-alias filter for the decimation to 1 ms.
    #
    # Leave order and decay at 0.5.  A sharper corner costs ringing quickly: at
    # order 1 the impulse response tail above -60 dB goes from 143 ms to
    # 236 ms, at order 2 to 376 ms.
    # The low-cut is here, ahead of the interpolation, and not at the far end
    # any more.  Below about 55 Hz the recorded traces are mutually incoherent
    # - adjacent-trace coherence 0.10 at 40 Hz against 0.50 at 75 Hz - so no
    # interpolator can put that band in the gaps, and leaving it in only gives
    # the curvelet solve something it cannot reconstruct to spend coefficients
    # on.  Removing it first costs 3 % of the energy.
    "str_bl": dict(
        highpass=True, hp_f_cut=60.0, hp_order=1.0, hp_decay=1.0,
        lowpass=True, lp_f_cut=300.0, lp_order=0.5, lp_decay=0.5,
        zero_dc=False, pad_front=2000,
    ),

    # `geom -> freq`, after the interpolation, on the 1 ms grid.  Low-cut only:
    # the 300 Hz above is already in the data and applying it twice would just
    # square the response near the corner.
    #
    # 60 Hz because below it the recorded traces are mutually incoherent -
    # adjacent-trace coherence is 0.10 at 40 Hz, 0.41 at 60 Hz and 0.50 at
    # 75 Hz.  Low coherence at low frequency is the signature of noise, not
    # signal: a longer wavelength should correlate better across a 3.10 m gap,
    # not worse.  It costs 3.0 % of the energy, and it also takes the 60 Hz
    # mains tone that sits 3.3 dB above its local background.
    "str": dict(
        highpass=True, hp_f_cut=60.0, hp_order=1.0, hp_decay=1.0,
        lowpass=True, lp_f_cut=300.0, lp_order=0.5, lp_decay=0.5,
        zero_dc=False, pad_front=2000,
    ),
}

# ------------------------------------------- normalisation and gain (`log`) --

# process/rms_normalize.py and process/logenv_process.py.  The RMS both domains
# are scaled to.  The value is arbitrary - the dataset divides by its own
# value_range - but it has to be the same for both: putting the two on a common
# amplitude scale is the entire point, since the instruments report amplitude
# four orders of magnitude apart.  It is also what makes the two log_base values
# below comparable, for the envelope-gain route.
TARGET_RMS = 0.1

# What rms_normalize.py measured and applied, recorded so an output can be
# traced back and so the numbers can be checked without re-reading 9 GB.
#
# INPUT_RMS is the overall RMS of each domain's normalisation input - DAS
# das_data_freq.npy, streamer str_data_geom.npy - over all 551 shots, taken
# as sqrt(sum of squares / count) rather than a mean of per-shot values.  The
# run recomputes it and warns if it has drifted, so these stay honest.
#
# Note that the mean of the per-shot RMS values comes out well below TARGET_RMS
# (about 0.78 of it for the DAS, 0.65 for the streamer) even though the overall
# RMS is exactly TARGET_RMS.  That is Jensen, not an error: the per-shot RMS
# spans 0.18-2.33 on the DAS and 0.10-4.07 on the streamer, and the mean of a
# spread-out set of values sits below its quadratic mean.
INPUT_RMS = {"das": 734.3926758515549, "str": 0.041681997497633266}

# The scalar each domain is multiplied by; derived, not independent.
NORM_SCALE = {tag: TARGET_RMS / r for tag, r in INPUT_RMS.items()}

# The FILTER_PARAMS entries each domain's normalisation input has been through,
# in order.  process/rms_normalize.py records these in the sidecar, so a saved
# array carries its whole filter history rather than one stage of it.
# The streamer is band-passed once, before the interpolation, so its chain
# is the one entry; FILTER_PARAMS["str"] is unused while that holds.
NORM_FILTER_CHAIN = {"das": ("das",), "str": ("str_bl",)}

# ------------------------------------------------------- direct-wave mute --

# process/mute_direct.py.  Everything earlier than the direct arrival is
# noise by construction - nothing can arrive before the fastest path from the
# shot - so it is safe to remove, and worth removing: on the streamer the
# curvelet cascade fills that window with coherent-looking wavetrains built
# out of what was incoherent noise going in (adjacent-trace coherence 0.010
# before the interpolation, 0.379 after), and on the DAS the recorded first
# sample is a spike eight times the record median.
#
# The mute boundary is  t = sqrt(offset^2 + depth_m^2) / velocity_m_s - lead_ms
# with the offset taken from the DAS geometry: source_xy from
# das_data_raw_meta.npz, receiver positions from das_data_geom_meta.npz, and
# the streamer's 93 positions read off the same array through
# streamer_trace_group.
#
# velocity_m_s is deliberately faster than the direct wave.  A faster velocity
# puts the boundary earlier, so it cannot reach a real arrival: the error is
# spent on muting less than it could rather than on cutting into signal.  The
# direct arrival measures about 1950 m/s apparent along the streamer, so
# anything comfortably above that is safe; the run reports how close the
# boundary comes to the energy that follows it.
#
# depth_m matters only near zero offset, where the sail line crosses the fibre
# and the horizontal offset falls to 0.22 m - without it the boundary there is
# t = 0 and nothing is muted.  Leaving it at 0 is the conservative choice.
#
# lead_ms is subtracted on top, and taper_ms is the raised-cosine ramp from
# fully muted to untouched, so the cut does not put a step into the record.
MUTE_PARAMS = dict(
    velocity_m_s=1800.0,
    depth_m=0.0,
    lead_ms=-30.0,
    taper_ms=30.0,
)

# Shots the input RMS is measured over; None reads every shot.
#
# None.  Sampling was a false economy: reading all 551 costs 13 s on the DAS
# array and 1 s on the streamer, against 23 s to write the output, while the
# evenly-spaced subsample is biased high - 64 shots overestimated the streamer
# RMS by 11 % and the DAS by 3 %.  np.linspace always includes the first and
# last shot, both of which happen to be three to four times the survey average,
# so a small sample gives them far more weight than their share.
RMS_SAMPLE_SHOTS = None

# "per_shot" fits a gain for every shot; "average" fits one from the mean
# gather and applies it to all.  The shot moves along the line here, so the
# moveout apex moves with it and an averaged gain smears across that -
# "per_shot" unless there is a reason to preserve shot-to-shot amplitude.
SCALE_MODE = "per_shot"

# utils.process.calculate_logscale() keyword arguments, per domain.
#
# log_base is added to the envelope in the envelope's own units, so it only
# means something relative to the amplitudes it meets - which is why the RMS
# normalisation above has to come first.  Even after it the two domains are not
# identical: an RMS-1 DAS gather has a median envelope near 0.21 and the
# streamer near 0.009, the streamer being the spikier of the two.  Keep
# log_base / median_envelope close between the two and both are compressed
# alike; the run prints that ratio.
LOG_SCALE_PARAMS = {
    "das": dict(log_base=2e-1, smooth_sigma=(5.0, 2.0), eps=1e-8,
                is_1d_envelope=True),
    "str": dict(log_base=1e-2, smooth_sigma=(3.0, 0.0), eps=1e-8,
                is_1d_envelope=True),
    # The pre-interpolation streamer gather - `raw` or `bl`, 24 traces on the
    # 0.5 ms grid.  Its own entry because the gather is a different shape and a
    # different amplitude distribution from the 93-trace `geom` one -
    # smooth_sigma is in samples and traces, and 24 traces is not many to
    # smooth across.
    #
    # Applied before the curvelet solve rather than after, the point is not
    # display: the solve keeps a fixed fraction of the largest coefficients, so
    # a gather whose direct arrival towers over everything below it spends its
    # coefficient budget there and the deep events fall under the threshold.
    # Balancing the amplitudes first puts them on comparable footing.  Undo the
    # same gain afterwards to get back to true amplitude.
    "str_pre": dict(log_base=5e-2, smooth_sigma=(3.0, 0.5), eps=1e-8,
                    is_1d_envelope=True),
}
