"""Interpolate a few shots two ways at once and look at all of it.

Each window is a grid.  Columns are shots, SHOTS_PER_FIGURE of them, and rows
are what was done to each: every stage of the interpolation, once on the input
as it is and once with the f-k fan filter applied first.

    METHOD "curvelet"   24 -> 47 -> 93, cascaded         2 variants x 3 stages
    METHOD "sinc"       24 -> 93, one kernel             2 variants x 2 stages

Reading down a column follows one shot through both variants and every stage;
reading across a row compares the same treatment between shots.  Shots are the
columns because that is the way round that gives each gather the tall narrow
panel a 71 m by 2 s record wants.  Nothing is written.

What to look at
---------------
Within a column every image shares one log-envelope gain and one clip, both
taken from that shot's unfiltered input, so the rows are on the same amplitude
scale: an event that keeps its amplitude and its dip down a column is being
reconstructed, one that fades or bends is not.  The gain is per column because
shots differ in amplitude by more than the stages do.  See DISPLAY for what the
gain is and why it is fitted once rather than per panel.  Every fourth trace of
a 4x panel is one that went in - check those first.  With REPLACE_OBSERVED off,
even those are the solver's own answer, and the misfit printed against each
stage says how far they drifted from the recorded traces.  The kernel is exact
there, so its misfit is zero.

The f-k rows repeat the same layout underneath, and are the alias diagnostic.
The recorded array is not spatially aliased below its own Nyquist, 1 / (2 x
3.098 m) = 0.161 c/m, so the true
wavefield has nothing beyond that line.  Interpolating to 0.77 m moves the
Nyquist out by four but cannot create real energy on the way: whatever appears
outside the dashed line is an artefact, and which panel it appears in says what
made it.

Dips run one way only.  All 551 shots have their offset decreasing
monotonically across the array - the source is always past the far end, at
184 m minimum against a 72 m array - so events arrive earlier at higher trace
index and nothing dips the other way.  Energy in the opposite f-k quadrant is
therefore also an artefact, whatever its wavenumber.

Watch what the fan filter does to spikes.  It is off in the pipeline now
because it turns an isolated one-trace spike into a wedge along +-v_cut - see
config.FK_PARAMS - and these panels are how that gets judged shot by shot.

Edit the settings block below, then run from the repository root:

    python -m process.interpolate_preview
"""

import os
import time

import matplotlib
import numpy as np
from scipy.signal.windows import tukey

import config as C
# Note: importing this forces the Agg backend, so its workers can draw
# headless.  show_figures() below puts an interactive one back.
import process.interpolate_traces as I
import process.intp_sweep as S
import utils.fdct_wrapping as FD
# The f-k fan mask and its application, so the preview and the stage that used
# to run it cannot drift apart.  See fk_variants.
import process.band_limit as B
# The windowed-sinc kernel the "sinc" METHOD uses.
import process.interpolate_kernel as K
# The log-envelope gain the images are drawn in comes from here, so the preview
# and the pipeline stage cannot drift apart.  See DISPLAY.
import process.logenv_process as L
from utils.data import get_project_root

import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------- settings --

# The interpolation's input: 10-200 Hz, muted, 24 traces on the 0.5 ms grid.
# The f-k fan is no longer applied here - fk_variants adds it, one row each way.
INPUT_NPY = C.ARRAYS["str"]["bl"]
DT_MS = C.STAGE_DT_US["str"]["bl"] / 1000.0

# "curvelet" runs the two-stage sparsity solve process/interpolate_traces.py
# runs.  "sinc" runs the windowed-sinc kernel from
# process/interpolate_kernel.py instead, on the same 1 -> 2 -> 4 grids.
#
# Sinc is worth comparing against now that the band is 10-200 Hz.  Band-limited
# interpolation is the unique reconstruction of a field with no energy past the
# array's Nyquist, and the recorded array no longer has any: the widest node
# gap is 3.25 m, so k_Nyq = 0.1538 c/m, while everything that survives the
# 200 Hz low-pass and the 1400 m/s fan cut has k = f / v < 200 / 1400 =
# 0.1429 c/m.  Sparsity-based interpolation earns its cost by reconstructing
# past the Nyquist from dip continuity; with nothing past the Nyquist there is
# nothing for it to earn, and the measured cost is real - the solve puts 2.9 %
# of its output beyond the input Nyquist, doubles the sub-1400 m/s content, and
# lands 16.5 % away from the recorded traces it was fitting.
#
# The kernel is exact by construction on the same terms: it reproduces the
# recorded traces, adds no wavenumbers, and takes seconds rather than 26 min.
METHOD = "sinc"

# Shots to draw.  They fill the columns, SHOTS_PER_FIGURE to a window.
SHOTS = range(0, 551, 50)

# Time window drawn.  None draws the whole record.  A stage costs roughly 20 s
# per shot over the full 4000 samples and scales with the window, so keeping
# this short is what makes the script usable.
WINDOW_MS = (0.0, 2000.0)

# Run the cascade over the whole record rather than the window.  Off runs it
# over the window widened by one block at each end, then crops - the interior
# is identical, and it is several times faster.
RUN_FULL_RECORD = False

# Shots per window, one to a column.  SHOTS is drawn in chunks of this many, so
# a long SHOTS list becomes several windows rather than one unreadable one.
SHOTS_PER_FIGURE = 4

# Inches per panel.  Tall and narrow, because a panel holds 71 m of array
# against 2 s of record and the time axis is the one worth reading.
PANEL_W = 3.4
PANEL_H = 6.4

CLIP_PERC = 99.0

CMAP = "gray"

# "linear" draws true amplitude, "log" the log-envelope gain from
# config.LOG_SCALE_PARAMS[LOG_PARAMS_KEY] - the same transform
# process/logenv_process.py applies, on the 24-trace entry.
#
# The gain is fitted once, on the input panel, and the same map is stretched
# across the trace axis onto the 2x and 4x grids.  Fitting each panel
# separately would give each its own gain and defeat the comparison: the point
# of the three panels is that an event which survives the cascade keeps the
# same amplitude, and that only reads if they share a gain.  The map is smooth
# by construction - Gaussian-smoothed with smooth_sigma - so stretching it is
# not the approximation it sounds like.
#
# calculate_logscale is not scale invariant: log_base is added to the envelope
# in the envelope's own units.  So the panels are put at config.TARGET_RMS
# first, as logenv_process does, and log_base then means what it means there.
DISPLAY = "linear"
LOG_PARAMS_KEY = "str_pre"

# Overrides log_base for the display only, leaving config alone.  None uses the
# config value.
#
# Worth having its own knob: config's "str_pre" is the gain the pipeline would
# apply before the curvelet solve, so turning it up to make a figure legible
# would quietly change what the solve sees.  log_base sets how hard the weak
# amplitudes are lifted - smaller lifts harder - and it is not scale free, so
# the useful range depends on TARGET_RMS.  At rms 0.1, 1e-3 lifts far enough to
# saturate the whole record into noise; try 5e-2 and work down.
LOG_BASE = None

# Zero the curvelet wedges that carry the dip the survey cannot produce.
#
# Every one of the 551 shots has its offset decreasing monotonically across the
# array - the source is always past the far end, 184 m minimum against a 72 m
# array - so an event arrives earlier at higher trace index and its moveout
# dt/dx is negative, without exception.  Positive-dip coefficients can only be
# fitting noise, the aperture edges, or the fold-over of a real event, and the
# solver is free to spend them wherever the data does not pin it down: between
# the recorded traces, which is the whole output.
#
# This patches process/intp_sweep.py from the outside and leaves it alone on
# disk, so the production run is unaffected until the setting is moved there.
DIP_RESTRICT = False

# Dips flatter than this apparent velocity count as flat and are kept whichever
# way they lean.  A wedge is judged by its centroid, so a wedge straddling zero
# lands near zero and survives; without the tolerance the flat wedges would be
# a coin toss and near-offset reflections, which are nearly flat, would be
# clipped on half of them.  The steepest real event is the 1950 m/s direct
# arrival, so anything above a few thousand m/s on the positive side is not a
# dip the survey can make.
DIP_FLAT_V = 12000.0

# Print the measured dip of every wedge and whether it is kept.  Worth leaving
# on the first few times: it is the only check that the calibration below
# agrees with what the transform actually does.
DIP_REPORT = True

# Draw the f-k panels as well.
SHOW_FK = True
FK_PAD_X = 4
FK_F_MAX = 240.0
FK_DB_FLOOR = -70.0

# Show interactively, and optionally save.  A falsy SAVE_DIR skips saving.
SHOW = True
SAVE_DIR = None

# Backends tried, in order, when SHOW is on.  None of them is guaranteed to be
# installed; the first one that imports wins.  Naming one here overrides the
# search.
BACKENDS = ("QtAgg", "Qt5Agg", "TkAgg", "MacOSX")

# ----------------------------------------------------------------- derived --

# The recorded nodes sit every fourth trace of the final 93-trace grid.
DX_IN = C.STR_TRACE_INTERVAL_M * (C.STR_TRACES - 1) / (C.STR_NODES - 1)
APERTURE = DX_IN * (C.STR_NODES - 1)
K_NYQ_IN = 1.0 / (2.0 * DX_IN)

# ---------------------------------------------------------------------------


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(get_project_root(), path)


def show_figures():
    """Whether plt.show() will actually put a window up.

    process.interpolate_traces sets the backend to Agg when it is imported, so
    that its worker processes can draw without a display.  This script is the
    opposite case, so swap an interactive backend back in.  Returns False if
    none of them is installed, which leaves saving as the only way to see
    anything.
    """
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
    print(f"  warning: none of {', '.join(BACKENDS)} is installed, so the "
          f"figures cannot be shown - set SAVE_DIR and look at the files")
    return False


def window_samples(nt):
    """The displayed sample range, and the wider range the cascade is run on."""
    if WINDOW_MS is None:
        return (0, nt), (0, nt)
    a = max(0, int(round(WINDOW_MS[0] / DT_MS)))
    b = min(nt, int(round(WINDOW_MS[1] / DT_MS)))
    if b - a < 2:
        raise SystemExit(f"WINDOW_MS {WINDOW_MS} is empty on a {nt}-sample record")
    if RUN_FULL_RECORD:
        return (a, b), (0, nt)
    pad = S.BLOCK_SAMPLES
    return (a, b), (max(0, a - pad), min(nt, b + pad))


# --------------------------------------------------------- dip restriction --

# Grid the current solve is running on, recorded by the interpolate_grid patch
# so that the build_mask patch - which is only handed coefficients - can work
# out the trace spacing.  Safe as a global because the preview runs the stages
# inline; process/interpolate_traces.py spreads shots over processes, so moving
# this into the pipeline means passing it down instead.
_GRID = {}
_DIPS = {}
_ORIG_INTERPOLATE_GRID = None
_ORIG_BUILD_MASK = None


def wedge_dips(shape, nbscales):
    """Moveout dt/dx of each wedge, in samples per trace, measured.

    One unit coefficient at the centre of a wedge inverts to that wedge's
    curvelet atom, whose 2-D spectrum is by construction confined to the
    wedge's own patch of the (k_t, k_x) plane.  A linear event of moveout p
    transforms onto the line k_x = -p k_t, so the energy centroid gives
    p = -k_x / k_t.

    Measuring beats indexing.  The wedges come in antipodal pairs that carry
    the same dip - the transform is run complex, is_real 0 - and the finer
    scales are numbered by a rounding rule this code does not own, so a
    hand-written table of wedge numbers is exactly the kind of thing that is
    wrong in one scale out of four and hard to notice.
    """
    key = (shape, nbscales)
    if key in _DIPS:
        return _DIPS[key]

    nt, nx = shape
    C = FD.fdct_wrapping(np.zeros(shape), S.IS_REAL_FORWARD, S.FINEST,
                         nbscales, S.NBANGLES_COARSE)
    kt = np.fft.fftfreq(nt)[:, None]
    kx = np.fft.fftfreq(nx)[None, :]

    dips = []
    for s, scale in enumerate(C):
        row = []
        for w, coef in enumerate(scale):
            i, j = coef.shape[0] // 2, coef.shape[1] // 2
            coef[i, j] = 1.0
            atom = FD.ifdct_wrapping(C, S.IS_REAL_INVERSE, nt, nx)
            coef[i, j] = 0.0
            e = np.abs(np.fft.fft2(atom)) ** 2
            tot = e.sum()
            if tot <= 0:
                row.append(0.0)
                continue
            kt_bar = float((e * kt).sum() / tot)
            kx_bar = float((e * kx).sum() / tot)
            row.append(0.0 if abs(kt_bar) < 1e-12 else -kx_bar / kt_bar)
        dips.append(row)

    _DIPS[key] = dips
    return dips


def wedge_keep(shape, nbscales, dx):
    """Which wedges survive, and the tolerance used, for a grid this shape."""
    dips = wedge_dips(shape, nbscales)
    tol = dx / (DIP_FLAT_V * DT_MS * 1e-3)
    keep = [[p <= tol for p in row] for row in dips]
    return keep, dips, tol


def _patched_interpolate_grid(grid, cols, threshold, alias_scales, nbscales):
    _GRID["shape"] = grid.shape
    _GRID["nbscales"] = nbscales
    return _ORIG_INTERPOLATE_GRID(grid, cols, threshold, alias_scales,
                                  nbscales)


def _patched_build_mask(C, threshold, alias_scales):
    """The stock support mask with the wrong-dip wedges zeroed.

    Applied after build_mask rather than inside it, so it lands on every scale
    on its own measured dip and never has to follow the inheritance rule that
    hands a coarse scale's mask up to the finer ones.
    """
    mask = _ORIG_BUILD_MASK(C, threshold, alias_scales)
    shape = _GRID.get("shape")
    # Checked here and not only at install time, so flipping DIP_RESTRICT after
    # the patch is in place actually turns it off - a sweep that runs both ways
    # in one process would otherwise silently restrict every run after the
    # first restricted one.
    if shape is None or not DIP_RESTRICT:
        return mask
    dx = APERTURE / (shape[1] - 2 * S.TRACE_PAD - 1)
    keep, _, _ = wedge_keep(shape, _GRID["nbscales"], dx)
    for s, row in enumerate(keep):
        if s >= len(mask) or mask[s] is None:
            continue
        for w, ok in enumerate(row):
            if not ok and w < len(mask[s]):
                mask[s][w] = np.zeros_like(mask[s][w])
    return mask


def install_dip_restriction():
    """Patch intp_sweep for this process only.  Returns whether it took."""
    global _ORIG_INTERPOLATE_GRID, _ORIG_BUILD_MASK
    if not DIP_RESTRICT:
        return False
    if _ORIG_BUILD_MASK is None:
        _ORIG_INTERPOLATE_GRID = S.interpolate_grid
        _ORIG_BUILD_MASK = S.build_mask
        S.interpolate_grid = _patched_interpolate_grid
        S.build_mask = _patched_build_mask
    return True


def report_dips(n_in):
    """The measured dip of every wedge, per stage, and the verdict on it."""
    nt = S.BLOCK_SAMPLES
    nx = n_in
    for k in range(S.N_STAGES):
        nx = (nx - 1) * S.STAGE_FACTOR + 1
        w = nx + 2 * S.TRACE_PAD
        dx = APERTURE / (nx - 1)
        shape = (nt, w)
        nbscales = S.scales_for(w)
        keep, dips, tol = wedge_keep(shape, nbscales, dx)
        v_tol = dx / (tol * DT_MS * 1e-3) if tol else float("inf")
        print(f"    stage {k + 1}: grid {nt} x {w}, dx {dx:.3f} m, "
              f"{nbscales} scales, tolerance {tol:+.3f} samples/trace "
              f"({v_tol:.0f} m/s apparent)")
        for s, (row, ok) in enumerate(zip(dips, keep)):
            kept = [i + 1 for i, o in enumerate(ok) if o]
            cut = [i + 1 for i, o in enumerate(ok) if not o]
            print(f"      scale {s}: {len(row):>2} wedges, keep {len(kept)}, "
                  f"cut {len(cut)}")
            if len(row) > 1:
                print(f"        dip {' '.join(f'{p:+.2f}' for p in row)}")
                print(f"        cut {cut}")


def sinc_to(traces, n_out):
    """The windowed-sinc kernel straight from `traces` to an n_out grid."""
    n_in = traces.shape[1]
    pos = K.output_positions(n_in, n_out)
    cols, w = K.kernel_weights(pos, n_in)
    return traces @ K.interp_matrix(cols, w, n_in, n_out)


def stages(gather):
    """The input and the output of each stage, as (label, grid, info).

    The two methods reach the panels differently, deliberately.

    The curvelet route cascades: stage 2's input is stage 1's output, because
    the solve should never extrapolate - going 2x at a time keeps every unknown
    trace bracketed by two known ones.

    The kernel route does not cascade.  It never extrapolates either, so it has
    no reason to, and going straight to the final grid is one matrix product
    instead of two.  Measured over 5 shots the two routes differ by 0.3 %, with
    the direct one marginally flatter below 0.7 of the array Nyquist and
    marginally cleaner past it, so the simpler one wins on a tie.  The 2x panel
    is then a separate 24 -> 47 interpolation made only to be looked at - it is
    not on the path to the 4x panel beside it.
    """
    if METHOD not in ("curvelet", "sinc"):
        raise SystemExit(f'METHOD must be "curvelet" or "sinc", not {METHOD!r}')
    n_in = gather.shape[1]
    out = [("1x  input", gather, None)]

    if METHOD == "sinc":
        # One column, not two: the kernel goes straight to the final grid, so
        # an intermediate 2x panel would be a separate picture of nothing that
        # happens on the way.
        step = S.STAGE_FACTOR ** S.N_STAGES
        t0 = time.time()
        y = sinc_to(gather, (n_in - 1) * step + 1)
        cols = np.arange(n_in) * step
        out.append((f"{step}x  direct", y,
                    dict(residual=0.0, seconds=time.time() - t0,
                         misfit=np.linalg.norm(y[:, cols] - gather)
                         / max(np.linalg.norm(gather), 1e-30))))
        return out

    current = gather
    for k in range(S.N_STAGES):
        step = S.STAGE_FACTOR ** (k + 1)
        t0 = time.time()
        current, residual = S.upsample_once(current, I.THRESHOLD,
                                            I.ALIAS_SCALES)
        cols = np.arange(n_in) * step
        misfit = (np.linalg.norm(current[:, cols] - gather)
                  / max(np.linalg.norm(gather), 1e-30))
        out.append((f"{step}x  stage {k + 1}", current,
                    dict(residual=residual, seconds=time.time() - t0,
                         misfit=misfit)))
    return out


def fk_variants(gather):
    """The input as it is, and with the fan filter applied.

    The filter used to live in process/band_limit.py and does not any more -
    see config.FK_PARAMS for why - so it is applied here instead, on the input,
    and both versions are carried through the interpolation side by side.  The
    mask is built from the same config entry either way, so what is drawn is
    what that stage would have done.
    """
    nt, nx = gather.shape
    m = B.build_fk_mask(nt, nx + 2 * B.FK_TRACE_PAD, DT_MS * 1e-3)
    return [("no f-k", gather),
            (f"f-k {B.FK_V_CUT:g} m/s",
             B.fk_apply(gather, m, B.FK_TRACE_PAD))]


def fk(gather, dx):
    """|F(f, k)| in dB, both axes shifted, k in cycles/m."""
    nt, nx = gather.shape
    w = np.outer(tukey(nt, 0.1), tukey(nx, 0.2))
    F = np.fft.rfft(gather * w, axis=0)
    F = np.fft.fftshift(np.fft.fft(F, n=nx * FK_PAD_X, axis=1), axes=1)
    amp = np.abs(F)
    f = np.fft.rfftfreq(nt, DT_MS * 1e-3)
    k = np.fft.fftshift(np.fft.fftfreq(nx * FK_PAD_X, dx))
    return f, k, 20 * np.log10(amp / max(amp.max(), 1e-30) + 1e-12)


def stretch(gain, ntr):
    """A gain map on n_in traces, linearly resampled onto ntr traces.

    Both grids span the same aperture with their first and last columns on the
    array ends, so this is a plain stretch of the trace axis.
    """
    n_in = gain.shape[1]
    if ntr == n_in:
        return gain
    xi = np.linspace(0.0, n_in - 1, ntr)
    lo = np.floor(xi).astype(int)
    hi = np.minimum(lo + 1, n_in - 1)
    w = (xi - lo)[None, :]
    return gain[:, lo] * (1.0 - w) + gain[:, hi] * w


def log_params():
    """config's entry, with the LOG_BASE override applied."""
    lsp = dict(C.LOG_SCALE_PARAMS[LOG_PARAMS_KEY])
    if LOG_BASE is not None:
        lsp["log_base"] = LOG_BASE
    return lsp


def display_gain(reference):
    """(scalar, gain map, label) for what the image panels are drawn in.

    The map is fitted on `reference` - the input panel - and stretched onto the
    denser grids by stretch(), so all three panels share one gain.
    """
    if DISPLAY == "linear":
        return 1.0, None, "true amplitude"
    if DISPLAY != "log":
        raise SystemExit(f'DISPLAY must be "linear" or "log", not {DISPLAY!r}')
    lsp = log_params()
    rms = float(np.sqrt(np.mean(reference ** 2)))
    norm = C.TARGET_RMS / rms if rms > 0 else 1.0
    gain = L.logscale(reference * norm, lsp)
    label = (f"log envelope, base {lsp['log_base']:g}, "
             f"sigma {lsp['smooth_sigma']}, at rms {C.TARGET_RMS:g}")
    return norm, gain, label


def fk_metrics(gather, dx):
    """Two energy fractions that say how much of a panel cannot be real.

    A linear event t = p x transforms to the line k = -f p, so with every
    moveout in this survey negative, all real energy sits at k > 0 for f > 0.
    Energy at k < 0 carries a dip the geometry cannot produce, and energy past
    the input Nyquist is beyond what 3.10 m sampling recorded.  Both are
    measured against the panel's own total, so the three panels compare.
    """
    nt, nx = gather.shape
    w = np.outer(tukey(nt, 0.1), tukey(nx, 0.2))
    F = np.fft.rfft(gather * w, axis=0)
    F = np.fft.fftshift(np.fft.fft(F, n=nx * FK_PAD_X, axis=1), axes=1)
    e = np.abs(F) ** 2
    f = np.fft.rfftfreq(nt, DT_MS * 1e-3)
    k = np.fft.fftshift(np.fft.fftfreq(nx * FK_PAD_X, dx))
    band = (f >= 20.0) & (f <= FK_F_MAX)
    e = e[band]
    tot = e.sum()
    if tot <= 0:
        return 0.0, 0.0
    return (float(e[:, np.abs(k) > K_NYQ_IN].sum() / tot),
            float(e[:, k < 0].sum() / tot))


def figure(shot, variants, view):
    """One window per shot: a column for every f-k variant and stage.

    The columns are the flattened product - no f-k 1x, no f-k 4x, f-k 1x,
    f-k 4x for the kernel, and the same with three stages for the curvelet -
    so every treatment of the shot stands beside every other, and the f-k
    spectra repeat the same columns underneath.

    Each panel is PANEL_W by PANEL_H inches, tall and narrow, because a panel
    holds 71 m of array against 2 s of record and the time axis is the one
    worth reading.

    Every image shares one gain and one clip, both taken from the unfiltered
    input, or the columns would not be comparable.
    """
    a, b = view
    cols = [(f"{vlabel}  {name}", g)
            for vlabel, panels in variants
            for name, g, _ in panels]
    ncol = len(cols)
    nrow = 2 if SHOW_FK else 1
    fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                             figsize=(PANEL_W * ncol, PANEL_H * nrow))

    norm, gain, gain_label = display_gain(variants[0][1][0][1])

    def shown(g):
        y = g * norm
        return y if gain is None else y * stretch(gain, g.shape[1])

    clip = float(np.percentile(np.abs(shown(variants[0][1][0][1])),
                               CLIP_PERC)) or 1.0
    img_ref = spec_ref = None

    for col, (label, g) in enumerate(cols):
        ntr = g.shape[1]
        dx = APERTURE / (ntr - 1)

        ax = axes[0][col]
        ax.imshow(shown(g), cmap=CMAP, vmin=-clip, vmax=clip, aspect="auto",
                  interpolation="nearest",
                  extent=[0, APERTURE, b * DT_MS, a * DT_MS])
        ax.set_title(f"{label}\n{ntr} traces @ {dx:.2f} m", fontsize=10)
        ax.set_xlabel("distance [m]", fontsize=9)
        if col == 0:
            ax.set_ylabel("time [ms]")
        else:
            ax.tick_params(labelleft=False)
        ax.tick_params(labelsize=8)
        # Linked pan and zoom across every image panel, both axes.  The grids
        # hold different numbers of traces but span the same 71.25 m, and x is
        # drawn in metres, so a distance range means the same thing on all.
        if img_ref is None:
            img_ref = ax
        else:
            ax.sharex(img_ref)
            ax.sharey(img_ref)

        if not SHOW_FK:
            continue
        # On the linear gather: the log gain varies in t and x, so it smears
        # the spectrum and would make the aliasing check useless.
        f, k, db = fk(g, dx)
        ax = axes[1][col]
        ax.imshow(db, aspect="auto", cmap="magma", vmin=FK_DB_FLOOR, vmax=0.0,
                  extent=[k[0], k[-1], f[0], f[-1]], origin="lower")
        for sgn in (-1, 1):
            ax.axvline(sgn * K_NYQ_IN, color="c", ls="--", lw=1)
        ax.set_xlim(-4 * K_NYQ_IN, 4 * K_NYQ_IN)
        ax.set_ylim(0, FK_F_MAX)
        ax.set_xlabel("wavenumber [cycles/m]", fontsize=9)
        if col == 0:
            ax.set_ylabel("frequency [Hz]")
        else:
            ax.tick_params(labelleft=False)
        ax.tick_params(labelsize=8)
        if spec_ref is None:
            spec_ref = ax
        else:
            ax.sharex(spec_ref)
            ax.sharey(spec_ref)

    if METHOD == "curvelet":
        how = (f"curvelet cascade, threshold {I.THRESHOLD:.0%}, alias scales "
               f"{I.ALIAS_SCALES}, {I.N_ITER} iterations, replace observed "
               f"{S.REPLACE_OBSERVED}, "
               + (f"dip restricted to <= {DIP_FLAT_V:g} m/s apparent"
                  if DIP_RESTRICT else "all dips"))
    else:
        how = (f"{K.KERNEL} kernel, half width {K.HALF_WIDTH}, "
               f"Kaiser beta {K.KAISER_BETA:g}")
    fig.suptitle(f"shot {shot}   {a * DT_MS:.0f}-{b * DT_MS:.0f} ms   {how}\n"
                 f"images: {gain_label}, one gain and clip for the window; "
                 f"dashed = input Nyquist, outside it is artefact")
    # Leave room for the two-line suptitle, which tight_layout does not
    # account for and would otherwise overlap the top row of panel titles.
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0 - 0.6 / (PANEL_H * nrow)))
    return fig


def main():
    I.configure()
    show = show_figures()
    if not show and not SAVE_DIR:
        raise SystemExit("nothing to do: SHOW is off (or has no backend) and "
                         "SAVE_DIR is not set")
    in_path = resolve(INPUT_NPY)
    if not os.path.isfile(in_path):
        raise SystemExit(f"not found: {in_path}")
    data = np.load(in_path, mmap_mode="r")
    n_shots, nt, n_in = data.shape
    bad = [s for s in SHOTS if not 0 <= s < n_shots]
    if bad:
        raise SystemExit(f"shots out of range 0..{n_shots - 1}: {bad}")
    if n_in != C.STR_NODES:
        print(f"  note: {n_in} input traces, config says {C.STR_NODES}")

    view, run = window_samples(nt)
    final = (n_in - 1) * S.STAGE_FACTOR ** S.N_STAGES + 1
    print(f"{INPUT_NPY}")
    print(f"  {data.shape} @ {DT_MS:g} ms, aperture {APERTURE:.2f} m, "
          f"input spacing {DX_IN:.3f} m, input Nyquist {K_NYQ_IN:.4f} c/m")
    print(f"  {S.N_STAGES} stages x {S.STAGE_FACTOR}: {n_in} -> "
          + " -> ".join(str((n_in - 1) * S.STAGE_FACTOR ** (k + 1) + 1)
                        for k in range(S.N_STAGES))
          + f"  (final grid {APERTURE / (final - 1):.3f} m)")
    if METHOD == "curvelet":
        print(f"  method: curvelet, threshold {I.THRESHOLD:.0%}, alias scales "
              f"{I.ALIAS_SCALES}, {I.N_ITER} iterations, blocks of "
              f"{S.BLOCK_SAMPLES} samples")
    else:
        print(f"  method: {K.KERNEL} kernel, half width {K.HALF_WIDTH}, "
              f"Kaiser beta {K.KAISER_BETA:g}")
    print(f"  display: {DISPLAY}, cmap {CMAP}"
          + (f", {LOG_PARAMS_KEY} {log_params()}, tail pad {L.PAD_TAIL}"
             + ("" if LOG_BASE is None else "  (log_base overridden here)")
             if DISPLAY == "log" else ""))
    if install_dip_restriction():
        print(f"  dip restriction ON: keeping dt/dx <= 0, flat band to "
              f"{DIP_FLAT_V:g} m/s apparent (intp_sweep patched in this "
              f"process only)")
        if DIP_REPORT:
            report_dips(n_in)
    else:
        print("  dip restriction off")
    print(f"  drawing samples {view[0]}-{view[1]} "
          f"({view[0] * DT_MS:.0f}-{view[1] * DT_MS:.0f} ms), "
          f"solving over {run[0]}-{run[1]}")

    out_dir = None
    if SAVE_DIR:
        out_dir = resolve(SAVE_DIR)
        os.makedirs(out_dir, exist_ok=True)

    for n, shot in enumerate(SHOTS, 1):
        print(f"\n  [{n}/{len(SHOTS)}] shot {shot}", flush=True)
        g = np.asarray(data[shot, run[0]:run[1]], dtype=np.float64)
        lo, hi = view[0] - run[0], view[1] - run[0]

        variants = []
        for vlabel, gv in fk_variants(g):
            print(f"    {vlabel}", flush=True)
            shown = []
            for name, arr, info in stages(gv):
                cut = arr[lo:hi]
                alias, wrong = fk_metrics(cut, APERTURE / (arr.shape[1] - 1))
                line = (f"      {name:<14} {arr.shape[1]:>3} traces @ "
                        f"{APERTURE / (arr.shape[1] - 1):.2f} m   "
                        f"rms {np.sqrt(np.mean(cut ** 2)):.4g}   "
                        f"past Nyquist {alias:5.1%}   wrong dip {wrong:5.1%}")
                if info is not None:
                    line += (f"   residual {info['residual']:.4f}"
                             f"   misfit at the recorded traces "
                             f"{info['misfit']:.1%}   {info['seconds']:.0f}s")
                print(line, flush=True)
                shown.append((name, cut, info))
            variants.append((vlabel, shown))

        fig = figure(shot, variants, view)
        if out_dir:
            p = os.path.join(out_dir, f"interpolate_preview_{shot:04d}.png")
            fig.savefig(p, dpi=130)
            print(f"        wrote {p}", flush=True)
        if show:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
