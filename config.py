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
        "line": os.path.join(NPY_DIR, "das_data_line.npy"),
        # `line` with the receiver axis decimated 12:1 by process/
        # decimate_receiver.py, 0.25 -> 3.00 m, to the streamer's node
        # interval.  This is the DAS side of the pair the dataset loads.
        "deci": os.path.join(NPY_DIR, "das_data_deci.npy"),
        "geom": os.path.join(NPY_DIR, "das_data_geom.npy"),
        "freq": os.path.join(NPY_DIR, "das_data_freq.npy"),
        "norm": os.path.join(DATA_DIR, "das_data_norm.npy"),
        "log": os.path.join(DATA_DIR, "das_data_log.npy"),
        # Log-scaled and transposed to receiver-major - (n_receivers,
        # n_samples, n_shots) - which is the axis order the training reads,
        # one receiver gather per sample.  Split by whether the receiver
        # overlaps the streamer array: `rg_train` is the 71 m the two
        # instruments share, `rg_infer` the rest of the 793 m of fibre, which
        # has no streamer counterpart and so can only be inferred on.
        # Written by process/logenv_process.py.
        "rg_train": os.path.join(DATA_DIR, "das_data_rg_train.npy"),
        "rg_infer": os.path.join(DATA_DIR, "das_data_rg_infer.npy"),
    },
    "str": {
        "raw": os.path.join(NPY_DIR, "str_data_raw.npy"),
        "line": os.path.join(NPY_DIR, "str_data_line.npy"),
        "bl": os.path.join(NPY_DIR, "str_data_bl.npy"),
        "intp": os.path.join(NPY_DIR, "str_data_intp.npy"),
        "geom": os.path.join(NPY_DIR, "str_data_geom.npy"),
        "freq": os.path.join(NPY_DIR, "str_data_freq.npy"),
        "norm": os.path.join(DATA_DIR, "str_data_norm.npy"),
        "log": os.path.join(DATA_DIR, "str_data_log.npy"),
        # As above.  Every streamer receiver is inside the shared aperture by
        # definition, so there is no `rg_infer` on this side.
        "rg_train": os.path.join(DATA_DIR, "str_data_rg_train.npy"),
    },
}

# Companion metadata written next to `raw` and `geom` by the scripts that make
# them: receiver and source coordinates, and the DAS channel grouping.
META = {
    "das_raw": os.path.join(NPY_DIR, "das_data_raw_meta.npz"),
    "das_geom": os.path.join(NPY_DIR, "das_data_geom_meta.npz"),
    "str_raw": os.path.join(NPY_DIR, "str_data_raw_meta.npz"),
    # Written by process/line_static.py alongside the `line` arrays: the
    # projected source positions the stage moved the data onto, the per-trace
    # shift it applied and the mute boundary it used.  Anything downstream of
    # `line` must take its source coordinates from here rather than from
    # *_raw_meta.npz - see LINE_STATIC_PARAMS.
    "das_line": os.path.join(NPY_DIR, "das_data_line_meta.npz"),
    "str_line": os.path.join(NPY_DIR, "str_data_line_meta.npz"),
    # Written by process/decimate_receiver.py: the block-centre receiver
    # coordinates of the decimated grid and the mute boundary they imply.
    "das_deci": os.path.join(NPY_DIR, "das_data_deci_meta.npz"),
}

# ------------------------------------------------------------- acquisition --

N_SHOTS = 551

# The common grid both domains are brought onto at the `geom` stage.
DT_US = 1000
N_SAMPLES = 3500

# Samples kept in the arrays the dataset loads.  The pipeline still runs at
# N_SAMPLES; process/rms_normalize.py crops on the way out.
#
# 2000 ms covers the target.  With 87 m of water - measured from the
# near-zero-offset arrival on the seafloor DAS, 58 ms at 1500 m/s - a reflector
# 200 m below the seabed arrives at 280 ms (DAS) or 338 ms (streamer) at zero
# offset, and at worst 1684 ms: that is the farthest offset in the survey,
# 2408 m, with the sediment as slow as 1450 m/s.  So 2000 leaves 300 ms of
# margin over the slowest case.
#
# What it drops is 5.9 % of the DAS energy and 0.7 % of the streamer's, none of
# it from the target interval - the streamer has 97.6 % of its energy inside
# the first 800 ms.  2000 is also 125 x 16, so it survives four stride-2
# stages.
#
# Note the mute then takes a larger share of what is left: 28.5 % of the DAS
# record and 37.0 % of the streamer's, against 16.3 % and 21.1 % at 3500.
# Patch sampling has to skip those or a third of the batch is empty.
NORM_SAMPLES = 2000

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
    "das": {"raw": 1000, "line": 1000, "deci": 1000, "geom": 1000,
            "freq": 1000, "norm": 1000, "log": 1000},
    # `line` is 1 ms on both sides: process/line_static.py decimates the
    # streamer's 0.5 ms grid 2:1 on the way through, so the two domains come
    # out of that stage on one time axis.
    "str": {"raw": 500, "line": 1000, "bl": 500, "intp": 500, "geom": 1000,
            "freq": 1000, "norm": 1000, "log": 1000},
}

# ------------------------------------------------- line static (`raw -> line`) --

# process/line_static.py.  The shots were meant to run along one line and did
# not: the boat wandered up to 37 m to the side of the principal axis, 13.9 m
# rms.  Sorting the shots by along-line position - which is what makes a
# receiver gather usable here, taking the 15.2 m shot interval down to 3.3 m -
# then puts neighbouring traces at inconsistent offsets, and the moveout comes
# out ragged rather than smooth.
#
# This stage removes that.  For every source-receiver pair it takes the
# difference between the offset as recorded and the offset the pair would have
# had with the source projected perpendicularly onto the line, converts it to
# time, and shifts the trace by it.  Afterwards the data is what a perfectly
# straight sail line would have recorded, to the accuracy of the single
# velocity below.
#
# How much it matters depends entirely on how close the line comes to the
# receivers, because d(offset)/d(lateral) vanishes at large offset:
#
#     str    24 nodes at along 1240..1312 m, shots span -1059..+1056 m, so the
#            minimum offset is 184 m and the shift is -1.50..+0.04 ms
#            (0.19 ms rms, at most 3 samples of 0.5 ms)
#     das  1058 groups at along 540..1348 m - the line crosses the fibre, the
#            minimum offset is 0.2 m and the shift is -22.49..+7.49 ms
#            (1.19 ms rms, at most 22 samples of 1 ms).  Under 50 m offset the
#            mean correction is 3.9 ms, and between neighbouring shots in
#            sorted order it jumps by 7.4 ms at the 99th percentile.
#
# So it is near-free on the streamer and the whole point on the DAS.
#
# velocity_m_s converts metres of offset error into time and is the one
# approximation in the stage.  The correction is exact for a straight ray at
# this velocity and scales as 1/v for anything else, so it is set to the water
# velocity: the near-offset traces, where the correction is large enough to
# matter, are dominated by the direct water arrival.  This is deliberately not
# MUTE_PARAMS["velocity_m_s"], which is set fast on purpose to keep the mute
# boundary clear of the arrival - using 1800 here would under-correct by 17 %.
# A 10 % velocity error leaves at most 2.2 ms of residual on the worst trace,
# against the mute's 30 ms taper.
#
# mode: "fourier" shifts by a phase ramp on the rfft, so the delay is
# fractional; "integer" rounds to whole samples and zero-fills.  Use "fourier".
# The streamer's 0.19 ms rms is well under half a sample of 0.5 ms, so rounding
# would quantise almost all of it to zero and the stage would do nothing.
#
# pad_samples / taper_samples: the phase ramp is a circular shift, so the
# record gets a tail of zeros to wrap into, preceded by a raised-cosine fade
# from the last sample so the join is continuous.  pad_samples has to exceed
# the largest shift; 256 covers the DAS's 22 by a wide margin.  The pad is
# cropped off afterwards, and no sample of the record itself is touched.
#
# out_dt_us / out_samples put both domains on one time axis, which is the
# other thing this stage does.  The streamer arrives on 0.5 ms and the DAS on
# 1 ms, and nothing downstream wants to keep track of which is which, so the
# streamer is decimated 2:1 here - with FILTER_PARAMS["str_bl"] in front of it
# as the anti-alias filter, since halving the rate halves the Nyquist from
# 1000 to 500 Hz.  The DAS is already on 1 ms and is only cropped.
#
# 2000 samples of 1 ms is the same window NORM_SAMPLES keeps, for the reasons
# written there: it covers the target interval with 300 ms of margin over the
# slowest case, drops 5.9 % of the DAS energy and 0.7 % of the streamer's, and
# is 125 x 16 so it survives four stride-2 stages.
#
# Order matters:
#
#     str   roll -> filter -> decimate 2:1 -> crop -> mute
#     das   roll -> filter ->                 crop -> mute
#
#   * the roll is first, on the input grid, so the streamer's sub-millisecond
#     shift is resolved at 0.5 ms rather than at 1 ms.
#   * the band-pass sits between the roll and the mute in BOTH domains.  Put
#     it on one side of the mute in one domain and the other side in the
#     other, and the two records meet their mute boundary carrying different
#     bands.  line_static.FILTER_KEY picks the FILTER_PARAMS entry per domain.
#   * the crop comes after the roll.  The DAS shift reaches -22.5 ms, so a
#     sample belonging at 1999 ms in the output sits past 2000 ms before the
#     shift, and cropping first would discard it.
#   * the mute is last and is built on the OUTPUT grid.  Applied on the fine
#     grid and then decimated, its onset gets resampled and that corner rings
#     - the mute has a 30 ms raised-cosine taper, but the taper's start is
#     still a corner.
LINE_STATIC_PARAMS = dict(
    velocity_m_s=1500.0,
    mode="fourier",
    pad_samples=256,
    taper_samples=64,
    out_dt_us=1000,
    out_samples=2000,
)

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
        highpass=True, hp_f_cut=20.0, hp_order=1.0, hp_decay=1.0,
        lowpass=True, lp_f_cut=300.0, lp_order=0.5, lp_decay=0.5,
        zero_dc=False, pad_front=2000,
    ),
    # `raw -> bl`, before the interpolation, on the 0.5 ms grid.  Applied by
    # process/band_limit.py on the curvelet route and by
    # process/interpolate_kernel.py on the sinc route, and it doubles as the
    # anti-alias filter for the later decimation to 1 ms -
    # process/decimate_intp.py imports the low-pass corner from band_limit.py
    # so the two cannot drift apart.
    #
    # 250 Hz sits under the interpolation limit with margin.  The recorded
    # nodes are 3.00 m apart at 14 of the 23 intervals and 3.25 m at the other
    # 9, and the widest gap decides; the direct arrival moves out 3.1 samples
    # of 0.5 ms per trace - 1950 m/s, not the 1500 m/s a marine gather is
    # usually assumed to hold - so it folds above v / (2 dx) = 300 Hz, and
    # above that the curvelet solve fits the fold-over instead of the true dip.
    #
    # Leave order and decay at 0.5.  They set the transition width, and a
    # sharper corner costs ringing quickly: at order 1 the impulse response
    # tail above -60 dB goes from 143 ms to 236 ms, at order 2 to 376 ms.
    #
    # The low-cut is back at 5 Hz, so the interpolation sees the whole low end
    # again.  It was 60 Hz for a while, on the measurement that the recorded
    # traces are mutually incoherent below about 55 Hz - adjacent-trace
    # coherence 0.10 at 40 Hz, 0.41 at 60 Hz, 0.50 at 75 Hz.  Low coherence at
    # low frequency is the signature of noise, not signal: a longer wavelength
    # should correlate better across a 3.10 m gap, not worse.  No interpolator
    # can put an incoherent band in the gaps, and cutting it ahead of the solve
    # took the period-4 artefact from 0.69 % to 0.43 % of the trace RMS, at a
    # cost of 3 % of the energy.  At 5 Hz that band - and the artefact with it
    # - is back.
    "str_bl": dict(
        highpass=True, hp_f_cut=20.0, hp_order=1.0, hp_decay=1.0,
        lowpass=True, lp_f_cut=300.0, lp_order=0.5, lp_decay=0.5,
        zero_dc=False, pad_front=2000,
    ),

    # `geom -> freq`, after the interpolation, on the 1 ms grid.
    #
    # Not currently reached: NORM_FILTER_CHAIN takes the streamer straight from
    # `geom` to `norm`, because "str_bl" above already set the band and
    # applying it a second time would square the response near both corners.
    # The entry is kept in step with "str_bl" so that switching the stage back
    # on does not quietly change the band.
    "str": dict(
        highpass=True, hp_f_cut=20.0, hp_order=1.0, hp_decay=1.0,
        lowpass=True, lp_f_cut=300.0, lp_order=0.5, lp_decay=0.5,
        zero_dc=False, pad_front=2000,
    ),
}

# ------------------------------------------------ f-k fan filter (`raw -> bl`) --

# process/band_limit.py, applied after the band-pass and before the mute, on
# the 0.5 ms grid.  utils.process.fk_filter builds the mask from the apparent
# velocity |f / k|:
#     m(v) = (1 + 2 ** (order * x)) ** (-decay / order),  x = v_cut - v
# with is_lowpass False, so slow - steep - events are the ones removed.
#
# What the data actually holds, stacked over 23 shots, 20-250 Hz, k != 0:
#
#     < 1500 m/s        6.6 %      1500-2000 m/s   30.5 %  (mode 1600-1700)
#     2000-4000        16.2 %      > 4000          43.2 %
#
# So there is very little genuinely slow energy to take out.  1400 m/s is set
# where the judgement is unambiguous rather than where the energy is: nothing
# in water propagates slower than about 1500 m/s except interface waves and
# cable noise, while the 1600-1700 m/s mode is a real recorded arrival - it
# measures 17.99 % of the energy with the mute off and 19.42 % with it on, so
# it is not the mute's own 1800 m/s boundary showing up either.  Cutting at
# 1950 would remove that mode and 37 % of the record with it; that is a
# decision about what the arrival is, not a filter setting.
#
# order and decay: x is in m/s, so the roll-off is 6.02 * decay dB per m/s and
# fk_filter's max_clip of 400 m/s puts a floor at -2408 * decay dB.  At 0.025
# the corner runs -6 dB at 1400 to -40 dB at 1174 - 226 m/s of transition -
# and the floor lands at -60 dB just where the clip takes over, which is the
# pairing that makes the two constants agree.  Do not go below about 0.02: at
# 0.010 the clip floor is only -24.6 dB and the stop band leaks.
#
# This mask is symmetric in |v| and so says nothing about which way an event
# dips.  It therefore does not touch the 18.8 % of the record that carries
# dt/dx > 0, a dip this geometry cannot produce - most of which sits at high
# apparent velocity, not in the slow band this removes.
# Off, for now.  It turns an isolated spike into a dipping wedge: a spike that
# lives on one trace is a delta in x, so it is flat in k, and the mask carves
# that flat line along the v_cut ray - the inverse transform spreads along
# +-v_cut instead of staying put.  Measured, the leak is 1.3 % of the spike at
# one trace away and 0.33 % at two, which sounds harmless until the spike is
# 2452 x the local median, as the worst one in shot 0 is: 1.3 % of that is
# still 32 x, a coherent dipping event several times stronger than the real
# ones around it.
#
# Turning the filter off is not the fix, only the immediate one.  A one-trace
# spike is not band-limited in x - it has energy at every wavenumber, well past
# the array's Nyquist - and both this filter and the interpolation that follows
# are defined only for fields that are.  The interpolation is in fact the worse
# of the two: the sinc kernel puts 0.90 of the spike on the adjacent
# interpolated trace and something above 1 % on 29 of the 93.  The real fix is
# to remove the spikes before either runs.
#
# process/interpolate_preview.py builds the mask from this entry regardless of
# `enabled`, so the filtered and unfiltered inputs can be looked at together.
FK_PARAMS = dict(
    enabled=False,
    v_cut=1400.0,
    order=0.050,
    decay=0.050,
    is_lowpass=False,
    # Traces of tapered pad on each side.  The 2-D transform is periodic in x
    # too, so without it the last trace and the first are neighbours and that
    # step smears across the whole f-k plane.  The pad holds the edge trace
    # faded to zero by a raised cosine and is cropped off afterwards.
    trace_pad=16,
)

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
# Measured on the `line` route, AFTER the mute: DAS das_data_deci.npy and
# streamer str_data_line.npy, all 551 shots, 2000 samples, with the
# direct-arrival mute applied first - which is why these differ from the
# pre-mute values the earlier `freq` / `geom` route gave (925.694 and
# 0.055303).  The mute removes 28.5 % of the DAS record and 37.0 % of the
# streamer's, so the DAS came down 47 % and the streamer went UP 10 %: the
# streamer's muted window was quieter than its average, the DAS's louder.
INPUT_RMS = {"das": 487.9856478458612, "str": 0.060958039756119686}

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
#
# One thing to watch on the `line` route.  process/line_static.py applies this
# mute itself, against the projected source positions, because that is the
# geometry its output holds.  A later stage that mutes again from
# *_raw_meta.npz would be using the wrong geometry, and wrong in the dangerous
# direction: the DAS shift is mostly negative, so the arrival has moved up to
# 22.5 ms earlier while a boundary built from the recorded positions has not,
# which puts it inside the first break.  Downstream of `line`, read
# source_xy_projected from META["*_line"] or leave the mute off.
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
# The smallest trace envelope in each domain's whole (shot x receiver) array,
# measured by process/measure_env_min.py on the `norm` stage and passed to
# utils.process.calculate_logscale as `env_min`.
#
# What it is for.  calculate_logscale subtracts data_env_log.min() from the log
# envelope, and that one number decides where its gain peaks:
#
#     peak envelope = e * (env_min + log_base) - log_base
#
# Left to itself it is the minimum of whatever array was handed in, so the gain
# a trace receives depends on the gather it was fitted in.  Measuring it once
# over everything makes every route agree by construction instead of by
# coincidence - a shot gather, a receiver gather, and the transposed
# receiver-major arrays all give the same number for the same sample.
#
# It does NOT change the values as things stand.  Both minima are ten orders of
# magnitude below their log_base, so log10(env_min + log_base) equals
# log10(log_base) to nine decimals - which is exactly what
# logenv_process.PAD_TAIL was already producing by putting zeros in the array.
# The gain map is unchanged; what changes is that it no longer depends on that
# coincidence holding.
#
# Re-measure after anything that changes the samples or the normalisation.
ENV_MIN = {"das": 4.145768201391925e-4, "str": 1.2834000626561065e-4}

LOG_SCALE_PARAMS = {
    "das": dict(log_base=1e-2, smooth_sigma=(5.0, 1.0), eps=1e-8,
                is_1d_envelope=True),
    "str": dict(log_base=5e-3, smooth_sigma=(5.0, 0.0), eps=1e-8,
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
    "str_pre": dict(log_base=1e-3, smooth_sigma=(3.0, 1.0), eps=1e-8,
                is_1d_envelope=True),
}
