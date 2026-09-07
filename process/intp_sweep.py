"""Curvelet trace interpolation: hyperparameter sweep over ALIAS_SCALES x THRESHOLD.

Upsamples the streamer gathers in the trace direction by STAGE_FACTOR, N_STAGES
times over, and shows for each combination of the two mask parameters what came
out and what its f-k spectrum looks like.  Only the final result is stored and
plotted; the intermediate stages are just steps on the way.

Why a cascade rather than one big jump: at 2x each stage has to fill a single
gap between two recorded traces, which is a far better-posed problem than
filling three at once, and the second stage starts from an already denser grid.
24 -> 47 -> 93 traces, i.e. 3.12 -> 1.56 -> 0.78 m.

Reading the f-k panels
----------------------
The recorded data are not spatially aliased, so the true wavefield has no
energy beyond the *input* spatial Nyquist (0.160 cycles/m at the 3.12 m group
interval).  Interpolating to a finer grid raises the Nyquist by the total
factor but cannot create real energy above the old one.  Any amplitude between
the two dashed lines in the output spectra is therefore an artefact - that band
is the sweep's main diagnostic, alongside the residual each setting prints.

Two modes:

  "upsample"  the product: the recorded traces go every STAGE_FACTOR-th column
              of a wider grid and the gaps are solved for.  No ground truth, so
              quality is judged from the f-k panels, the images, and the
              residual on the observed traces.
  "validate"  keeps every (STAGE_FACTOR ** N_STAGES)-th recorded trace, runs the
              same cascade, and scores against the traces held out.  The only
              mode that yields an SNR, so it is the one to trust when ranking
              settings - but it reconstructs from an array 4x coarser than the
              one you actually have, so it is a harder problem than the real
              job.

Runtime is the constraint: every CG iteration costs two curvelet transforms,
multiplied by N_STAGES stages x 9 settings x SHOT_COUNT shots x the number of
time blocks.  Run once with SHOT_COUNT = 1 to time it.

Edit the settings block below, then run from the repository root:

    python -m process.intp_sweep
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal.windows import tukey

from utils.data import get_project_root
from utils.fdct_wrapping import fdct_wrapping, ifdct_wrapping

# ---------------------------------------------------------------- settings --

# (n_shots, n_samples, n_channels) input.  Relative paths resolve against the
# project root.
# The 220 Hz band-limited streamer array, still on the SEG-Y's 0.5 ms grid and
# its 24 channels.  Interpolating the unfiltered data does not work: above the
# 240 Hz alias onset the solver fits the fold-over rather than the true dip,
# and the replica survives in the output.
#
# Named `_bl` rather than `_freq` because `_freq` now means the band-pass
# applied on the common 1 ms geometry grid, at the far end of the pipeline
# (das_data_geom -> das_data_freq).  This one is a preparatory step for the
# interpolation and comes much earlier.  The file does not currently exist -
# regenerate it with a band-limiting pass over str_data_raw.npy if the
# interpolation has to be re-run.
INPUT_NPY = os.path.join("data", "pohang_shore", "numpy", "str_data_bl.npy")

# Directory for the saved arrays and figures; created if missing.
OUT_DIR = os.path.join("data", "pohang_shore", "intp_sweep")

# Acquisition, for the f-k axes.
DT_MS = 0.5
DX_M = 3.1196

# "upsample" or "validate"; see the docstring.
MODE = "upsample"

# Trace upsampling: STAGE_FACTOR per stage, N_STAGES stages.  2 x 2 takes the
# 24 recorded traces to 47 and then 93 - every unknown trace stays bracketed by
# two known ones, and nothing is extrapolated past the end of the array.
STAGE_FACTOR = 2
N_STAGES = 2

# Put the recorded traces back into the output of each stage, so they pass
# through the cascade unchanged.  Left off: the solver's own values at the
# observed positions differ from the measurements, so slamming the data back in
# makes the interpolated traces step away from their neighbours, and that
# trace-to-trace discontinuity shows up as striping across the whole f-k plane
# and as visible vertical banding in the gather.  Measured worse on every
# threshold tried, filtered and unfiltered alike.
REPLACE_OBSERVED = False

# Shots: SHOT_COUNT gathers starting at SHOT_START, every SHOT_STEP.
SHOT_START = 5
SHOT_STEP = 50
SHOT_COUNT = 1

# Time window of the record to process, in samples.  The full record is 8000
# samples (4 s at 0.5 ms); 2000 samples is the first 1 s.
T_START = 0
T_SAMPLES = 2000

# The gather is far taller than it is wide, while the curvelet transform
# assumes a roughly isotropic grid.  Processing in time blocks keeps each
# transform closer to square.  Blocks overlap by BLOCK_OVERLAP of their length
# and are recombined with a half-shifted Hann taper.
BLOCK_SAMPLES = 400
BLOCK_OVERLAP = 0.5

# The transform is FFT-based, so the trace axis is periodic: what leaves the
# right-hand edge of the array comes back in at the left.  With a live event
# running right up to the last trace that wrap is a step discontinuity, and it
# smears across the whole spectrum.
#
# TRACE_PAD columns are added on each side, holding the edge trace faded out to
# zero over TRACE_TAPER of the pad by a raised cosine.  The pad columns are
# treated as observed, so the solver has to follow the decay and the field is
# continuous where it wraps; they are cropped off before anything is returned,
# so the padding never reaches the output.  Mirror padding would also close the
# discontinuity, but it reverses the dip of every event, and a dip-selective
# transform then represents the mirror image as real signal.
TRACE_PAD = 16
TRACE_TAPER = 1.0

# The swept parameters.  ALIAS_SCALES cannot usefully exceed the scale count
# actually used at a stage: there it means "threshold every scale directly,
# inherit nothing", which is the baseline the others are measured against.
# Chosen on the 220 Hz filtered data; put more values back to re-sweep.
# 2 % is deliberately absent: that mask is loose enough that the solver simply
# reproduces the zero-filled input - the recorded traces are matched to 6 %
# because nothing was changed, the gaps stay visible in the image and the
# +-0.32 c/m replicas survive intact in the f-k.  It scores best on data
# fidelity by not interpolating.
ALIAS_SCALES_LIST = [2]
THRESHOLD_LIST = [0.05]

# Curvelet transform.  NBSCALES is an upper bound: it is clamped per stage to
# what that stage's grid width can carry.  fdct_wrapping starts at M = n/3 and
# halves it once per scale, so the narrow axis sets the limit - 47 traces
# support 3 scales, 93 support 4.  Eight scales, the value intp_CG.m used on
# its 792x800 images, would collapse to floor(4*M) = 0 here.
NBSCALES = 4
NBANGLES_COARSE = 16
FINEST = 1
# intp_CG.m pairs a forward with is_real=0 to an inverse with is_real=1, which
# is not an adjoint pair: measured on a 400x93 block, that combination has a
# 31 % round-trip error (0->0 and 1->1 both give 7e-16) and fails the adjoint
# identity by a factor 0.67, so the CG descends on something that is not the
# gradient of its objective.  Set IS_REAL_INVERSE = 1 to reproduce the MATLAB.
IS_REAL_FORWARD = 0
IS_REAL_INVERSE = 0

# CG.  N_ITER is halved from the 20 of intp_CG.m to keep the sweep tractable;
# raise it once a setting has been chosen.
MU = 0.0
ALPHA = 1.0
N_ITER = 10

# f-k display.
PAD_X = 8
TAPER_T = 0.1
TAPER_X = 0.2
DB_FLOOR = -60.0
F_MAX = 250.0

# Image display.
CLIP_PERC = 99.0

# Recompute even if a saved result is already on disk.
FORCE = True

# ---------------------------------------------------------------------------

TOTAL_FACTOR = STAGE_FACTOR ** N_STAGES


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(get_project_root(), path)


# ------------------------------------------------------------------ scales --

def max_scales(nx):
    """Largest scale count whose coarsest wedge grid stays as wide as the one
    intp_CG.m worked with (floor(4*M) >= 8), i.e. 2^(n-1) <= nx/6."""
    return max(1, int(np.floor(1 + np.log2(max(nx, 6) / 6.0))))


def scales_for(nx):
    """NBSCALES clamped to what a grid this wide can carry."""
    return max(1, min(NBSCALES, max_scales(nx)))


def stage_widths(nx_in):
    """Grid width after each cascade stage."""
    widths = []
    nx = nx_in
    for _ in range(N_STAGES):
        nx = (nx - 1) * STAGE_FACTOR + 1
        widths.append(nx)
    return widths


# ------------------------------------------------------------ interpolation --

def build_mask(C, threshold, alias_scales):
    """Support mask per scale/wedge.

    The coarsest `alias_scales` scales are thresholded at `threshold` times the
    largest coefficient magnitude in that scale.  Finer scales inherit an
    upsampled copy of the next coarser scale's mask, which carries the
    unaliased dip information up into the bands where thresholding the
    decimated data would pick up the aliases themselves.
    """
    cutoff = [threshold * max(np.abs(w).max() for w in scale) for scale in C]

    mask = [None] * len(C)
    for s in range(min(alias_scales, len(C))):
        mask[s] = [(np.abs(w) > cutoff[s]).astype(float) for w in C[s]]

    for s in range(alias_scales, len(C)):
        l1, l2 = len(C[s]), len(C[s - 1])
        # MATLAB: n = uint8(l1/l2); a = w/n with n a uint8, so the division
        # rounds instead of erroring on odd w.
        n = max(1, int(np.floor(l1 / l2 + 0.5)))
        mask[s] = [imresize_nearest(
            mask[s - 1][max(1, min(l2, int(np.floor(w / n + 0.5)))) - 1],
            C[s][w - 1].shape) for w in range(1, l1 + 1)]
    return mask


def _nearest_axis(n_out, n_in):
    """MATLAB imresize(...,'nearest') index map for one axis, 0-based."""
    u = (np.arange(1, n_out + 1) - 0.5) * n_in / n_out + 0.5
    return np.clip(np.floor(u + 0.5), 1, n_in).astype(int) - 1


def imresize_nearest(mask, shape):
    return mask[np.ix_(_nearest_axis(shape[0], mask.shape[0]),
                       _nearest_axis(shape[1], mask.shape[1]))]


def apply_mask(C, mask):
    return [[c * m for c, m in zip(cs, ms)] for cs, ms in zip(C, mask)]


def gradient(Ct, mask, sampled, nt, nx, cols, nbscales):
    """grad = 2 mu Ct - mask .* F(S* (d - S F* (mask .* Ct)))."""
    m = ifdct_wrapping(apply_mask(Ct, mask), IS_REAL_INVERSE, nt, nx)
    misfit = sampled - np.real(m[:, cols])

    null = np.zeros((nt, nx))
    null[:, cols] = misfit
    Cnull = fdct_wrapping(null, IS_REAL_FORWARD, FINEST, nbscales,
                          NBANGLES_COARSE)
    grad = [[2 * MU * t - n * mk for t, n, mk in zip(ts, ns, ms)]
            for ts, ns, ms in zip(Ct, Cnull, mask)]
    return grad, misfit


def coeff_dot(A, B):
    return sum(complex(np.sum(np.conj(a) * b))
               for cs, bs in zip(A, B) for a, b in zip(cs, bs))


def interpolate_grid(grid, cols, threshold, alias_scales, nbscales):
    """Fill the unobserved columns of `grid`.

    `grid` already holds the known traces at `cols` and zeros elsewhere.
    Returns the reconstruction and the relative residual on `cols`.
    """
    nt, nx = grid.shape
    sampled = grid[:, cols]
    norm = np.linalg.norm(sampled)

    Ci = fdct_wrapping(grid, IS_REAL_FORWARD, FINEST, nbscales, NBANGLES_COARSE)
    mask = build_mask(Ci, threshold, alias_scales)

    Ct = [[np.zeros_like(w) for w in scale] for scale in Ci]
    grad_prev, misfit = gradient(Ct, mask, sampled, nt, nx, cols, nbscales)
    direction = [[-g for g in gs] for gs in grad_prev]
    Ct = [[t + ALPHA * d for t, d in zip(ts, ds)]
          for ts, ds in zip(Ct, direction)]

    for _ in range(N_ITER):
        grad, misfit = gradient(Ct, mask, sampled, nt, nx, cols, nbscales)
        num = coeff_dot([[g - gp for g, gp in zip(gs, gps)]
                         for gs, gps in zip(grad, grad_prev)], grad)
        den = coeff_dot(grad_prev, grad_prev)
        beta = num / den if den != 0 else 0.0
        direction = [[-g + beta * d for g, d in zip(gs, ds)]
                     for gs, ds in zip(grad, direction)]
        Ct = [[t + ALPHA * d for t, d in zip(ts, ds)]
              for ts, ds in zip(Ct, direction)]
        grad_prev = grad

    out = np.real(ifdct_wrapping(Ct, IS_REAL_INVERSE, nt, nx))
    residual = float(np.linalg.norm(misfit) / norm) if norm > 0 else np.nan
    return out, residual


# ------------------------------------------------------------------ blocks --

def half_shifted_hann(n):
    """Hann sampled half a grid step off: w[i] = sin^2(pi (i + 0.5) / n).

    The symmetric Hann is exactly zero at i = 0 and i = n-1, which left the
    first and last sample of the record with zero total weight and forced them
    to zero.  Shifting by half a sample keeps both ends nonzero, and at 50 %
    overlap consecutive windows still sum to exactly one, since
    sin^2(x) + sin^2(x + pi/2) = 1.
    """
    return np.sin(np.pi * (np.arange(n) + 0.5) / n) ** 2


def block_starts(n_samples):
    step = max(1, int(round(BLOCK_SAMPLES * (1.0 - BLOCK_OVERLAP))))
    if n_samples <= BLOCK_SAMPLES:
        return [0], n_samples
    starts = list(range(0, n_samples - BLOCK_SAMPLES + 1, step))
    if starts[-1] + BLOCK_SAMPLES < n_samples:
        starts.append(n_samples - BLOCK_SAMPLES)
    return starts, BLOCK_SAMPLES


def interpolate_gather(grid, cols, threshold, alias_scales, nbscales):
    """Blocked interpolation over the time axis, recombined by weighted average.

    Each block is solved independently - so each gets its own support mask -
    and the outputs are averaged with the half-shifted Hann as weights,
    normalised by the weight sum.  The normalisation is what makes the
    single-covered samples at the two ends come out at full amplitude.
    """
    nt, nx = grid.shape
    starts, blen = block_starts(nt)
    if len(starts) == 1:
        return interpolate_grid(grid, cols, threshold, alias_scales, nbscales)

    acc = np.zeros((nt, nx))
    wsum = np.zeros((nt, 1))
    taper = half_shifted_hann(blen)[:, None]
    residuals = []
    for s in starts:
        out, res = interpolate_grid(grid[s:s + blen], cols, threshold,
                                    alias_scales, nbscales)
        acc[s:s + blen] += out * taper
        wsum[s:s + blen] += taper
        residuals.append(res)
    return acc / np.maximum(wsum, 1e-12), float(np.mean(residuals))


# ----------------------------------------------------------------- cascade --

def pad_traces(grid, cols):
    """Extend the trace axis so the periodic transform does not wrap the two
    aperture edges into each other.

    The pad holds the outermost observed trace faded to zero by a raised
    cosine, and counts as observed, so the solution is pinned to that decay
    instead of being free to grow where nothing constrains it.  Returns the
    padded grid, the shifted observed columns, and the pad width.
    """
    if TRACE_PAD <= 0:
        return grid, cols, 0

    p = int(TRACE_PAD)
    t = np.arange(1, p + 1) / p
    w = 0.5 * (1.0 + np.cos(np.pi * np.clip(t / max(TRACE_TAPER, 1e-9), 0, 1)))

    left = grid[:, cols[0]][:, None] * w[::-1][None, :]     # 0 outward, ~1 inward
    right = grid[:, cols[-1]][:, None] * w[None, :]         # ~1 inward, 0 outward
    padded = np.concatenate([left, grid, right], axis=1)

    nx = grid.shape[1]
    new_cols = np.concatenate([np.arange(p), cols + p,
                               np.arange(nx + p, nx + 2 * p)])
    return padded, new_cols, p


def upsample_once(traces, threshold, alias_scales):
    """One STAGE_FACTOR-fold stage: known traces in, denser grid out."""
    nt, nx = traces.shape
    nx_out = (nx - 1) * STAGE_FACTOR + 1
    cols = np.arange(nx) * STAGE_FACTOR
    grid = np.zeros((nt, nx_out))
    grid[:, cols] = traces

    grid, pad_cols, pad = pad_traces(grid, cols)
    nbscales = scales_for(grid.shape[1])
    out, residual = interpolate_gather(grid, pad_cols, threshold, alias_scales,
                                       nbscales)
    if pad:
        out = out[:, pad:pad + nx_out]
    if REPLACE_OBSERVED:
        out[:, cols] = traces
    return out, residual


def cascade(traces, threshold, alias_scales):
    """Run every stage; return the final grid and the per-stage residuals."""
    residuals = []
    current = traces
    for _ in range(N_STAGES):
        current, res = upsample_once(current, threshold, alias_scales)
        residuals.append(res)
    return current, residuals


def prepare(gather):
    """Split a recorded gather into (input traces, reference or None).

    In "validate" the reference is the span the cascade can reach, so the
    reconstruction and the reference line up column for column.
    """
    if MODE == "upsample":
        return gather, None
    if MODE == "validate":
        keep = np.arange(0, gather.shape[1], TOTAL_FACTOR)
        return gather[:, keep], gather[:, :keep[-1] + 1]
    raise SystemExit(f'MODE must be "upsample" or "validate", not {MODE!r}')


def display_grid(traces, nx_out):
    """The input traces on the final grid with the gaps left at zero, for the
    image and f-k panels."""
    nt, nx = traces.shape
    cols = np.arange(nx) * TOTAL_FACTOR
    keep = cols < nx_out
    grid = np.zeros((nt, nx_out))
    grid[:, cols[keep]] = traces[:, keep]
    return grid


# ------------------------------------------------------------------- f-k ----

def fk_amplitude(gather, dx, pad_x):
    """|F(f, k)| with both axes shifted, f from -Nyquist to +Nyquist."""
    nt, nx = gather.shape
    w = np.outer(tukey(nt, TAPER_T), tukey(nx, TAPER_X))
    F = np.fft.fft(gather * w, axis=0)
    F = np.fft.fft(F, n=nx * pad_x, axis=1)
    amp = np.abs(np.fft.fftshift(F, axes=(0, 1)))
    f = np.fft.fftshift(np.fft.fftfreq(nt, d=DT_MS / 1000.0))
    k = np.fft.fftshift(np.fft.fftfreq(nx * pad_x, d=dx))
    return f, k, amp


def draw_fk(ax, stack, dx, k_nyq_in, title):
    """`stack` is (n_shots, nt, nx).  The gathers have different moveout, so
    the average is taken on |F| after the transform - averaging the gathers
    first would let events at different positions cancel."""
    f = k = amp = None
    for gather in stack:
        f, k, a = fk_amplitude(gather, dx, PAD_X)
        amp = a if amp is None else amp + a
    amp /= len(stack)

    db = 20 * np.log10(np.maximum(amp, 1e-30) / amp.max())
    im = ax.pcolormesh(k, f, db, cmap="turbo", vmin=DB_FLOOR, vmax=0.0,
                       shading="auto")
    k_nyq_out = 1.0 / (2.0 * dx)
    for sign in (-1, 1):
        ax.axvline(sign * k_nyq_in, color="white", ls="--", lw=1.0)
        if abs(k_nyq_out - k_nyq_in) > 1e-9:
            ax.axvline(sign * k_nyq_out, color="white", ls=":", lw=1.0,
                       alpha=0.7)
    ax.set_ylim(-F_MAX, F_MAX)
    ax.set_xlim(-k_nyq_out, k_nyq_out)
    ax.set_title(title, fontsize=9)
    return im


# ---------------------------------------------------------------- plotting --

def plot_gathers(shot, input_grid, results, out_dir):
    """One figure per shot: the input next to the 3 x 3 sweep of final results."""
    n_a, n_t = len(ALIAS_SCALES_LIST), len(THRESHOLD_LIST)
    fig, axes = plt.subplots(n_a, n_t + 1, figsize=(3.0 * (n_t + 1), 3.4 * n_a),
                             squeeze=False)
    clip = np.percentile(np.abs(input_grid), CLIP_PERC)

    for i, alias in enumerate(ALIAS_SCALES_LIST):
        axes[i][0].imshow(input_grid, cmap="gray", vmin=-clip, vmax=clip,
                          aspect="auto", interpolation="nearest")
        axes[i][0].set_title("input (gaps zeroed)" if i == 0 else "", fontsize=9)
        axes[i][0].set_ylabel(f"alias={alias}", fontsize=9)
        for j, thr in enumerate(THRESHOLD_LIST):
            ax = axes[i][j + 1]
            ax.imshow(results[(alias, thr)], cmap="gray", vmin=-clip, vmax=clip,
                      aspect="auto", interpolation="nearest")
            ax.set_title(f"thr={thr:.0%}" if i == 0 else "", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[i][0].set_xticks([])
        axes[i][0].set_yticks([])

    fig.suptitle(f"shot {shot} - curvelet interpolation x{TOTAL_FACTOR} "
                 f"({N_STAGES} x {STAGE_FACTOR})")
    fig.tight_layout()
    path = os.path.join(out_dir, f"gathers_shot{shot:04d}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_fk_grid(inputs, results, out_dir):
    """One figure: input spectrum plus the 3 x 3 sweep, each averaged over the
    processed shots in the |F| domain."""
    n_a, n_t = len(ALIAS_SCALES_LIST), len(THRESHOLD_LIST)
    k_nyq_in = 1.0 / (2.0 * DX_M)
    dx_out = DX_M / TOTAL_FACTOR

    fig, axes = plt.subplots(n_a, n_t + 1, figsize=(3.4 * (n_t + 1), 3.4 * n_a),
                             squeeze=False)
    im = None
    for i, alias in enumerate(ALIAS_SCALES_LIST):
        draw_fk(axes[i][0], inputs, dx_out, k_nyq_in,
                "input (gaps zeroed)" if i == 0 else "")
        axes[i][0].set_ylabel(f"alias={alias}\nfrequency [Hz]", fontsize=9)
        for j, thr in enumerate(THRESHOLD_LIST):
            im = draw_fk(axes[i][j + 1], results[(alias, thr)], dx_out,
                         k_nyq_in, f"thr={thr:.0%}" if i == 0 else "")
    for ax in axes[-1]:
        ax.set_xlabel("k [cycles/m]", fontsize=9)

    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), label="dB re peak",
                     shrink=0.6)
    fig.suptitle(f"f-k after x{TOTAL_FACTOR} interpolation - dashed: input "
                 f"Nyquist ({k_nyq_in:.3f} c/m), dotted: output Nyquist "
                 f"({1 / (2 * dx_out):.3f} c/m).\n"
                 "Energy between them is interpolation artefact.")
    path = os.path.join(out_dir, "fk_sweep.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# -------------------------------------------------------------------- main --

def snr_db(signal, noise):
    ps = float(np.sum(signal ** 2))
    pn = float(np.sum(noise ** 2))
    return 10.0 * np.log10(ps / pn) if pn > 0 else np.inf


def fmt_hms(seconds):
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


def report_stages(nx_in):
    """Print the cascade plan and refuse widths NBSCALES cannot carry."""
    nx = nx_in
    for stage in range(N_STAGES):
        nx_out = (nx - 1) * STAGE_FACTOR + 1
        nx_solve = nx_out + 2 * max(0, int(TRACE_PAD))
        nbs = scales_for(nx_solve)
        m = (nx_solve / 3.0) / (2 ** (nbs - 1))
        f4 = int(np.floor(4 * m))
        dx = DX_M / (STAGE_FACTOR ** (stage + 1))
        print(f"  stage {stage + 1}: {nx:3d} -> {nx_out:3d} traces "
              f"({dx:.3f} m), solved on {nx_solve} with {TRACE_PAD} pad each "
              f"side, {nbs} scales, coarsest floor(4*M) = {f4}")
        if f4 < 4:
            raise SystemExit(
                f"stage {stage + 1} cannot run: {nx_out} traces at {nbs} "
                f"scales collapses to floor(4*M) = {f4}. Lower NBSCALES.")
        if nbs < NBSCALES:
            print(f"      (NBSCALES = {NBSCALES} clamped to {nbs} for this width)")
        over = [a for a in ALIAS_SCALES_LIST if a > nbs]
        if over:
            print(f"      NOTE: ALIAS_SCALES {over} exceed {nbs} here and "
                  f"behave identically to {nbs}")
        nx = nx_out
    return nx


def main():
    path = resolve(INPUT_NPY)
    if not os.path.isfile(path):
        raise SystemExit(f"not found: {path}")
    out_dir = resolve(OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    data = np.load(path, mmap_mode="r")
    n_shots, n_samples, n_traces = data.shape
    shots = [s for s in range(SHOT_START, n_shots, SHOT_STEP)][:SHOT_COUNT]
    t_stop = min(T_START + T_SAMPLES, n_samples)

    print(f"{path}\n  {n_shots} shots x {n_samples} samples x {n_traces} traces")
    print(f"  mode '{MODE}', x{TOTAL_FACTOR} in {N_STAGES} stages of "
          f"{STAGE_FACTOR}, shots {shots}")
    print(f"  samples {T_START}..{t_stop}, blocks of {BLOCK_SAMPLES} "
          f"overlapping {BLOCK_OVERLAP:.0%}")
    print(f"  sweep: alias {ALIAS_SCALES_LIST} x threshold {THRESHOLD_LIST}")

    save_path = os.path.join(out_dir, f"sweep_{MODE}_x{TOTAL_FACTOR}.npz")
    if os.path.isfile(save_path) and not FORCE:
        print(f"  loading cached {save_path} (set FORCE = True to recompute)")
        cached = np.load(save_path)
        results = {(int(key.split("|")[0]), float(key.split("|")[1])): cached[key]
                   for key in cached.files if key != "inputs"}
        inputs = cached["inputs"]
    else:
        first = np.asarray(data[shots[0], T_START:t_stop], dtype=float)
        traces0, _ = prepare(first)
        nx_final = report_stages(traces0.shape[1])
        n_blocks = len(block_starts(traces0.shape[0])[0])

        inputs = np.zeros((len(shots), traces0.shape[0], nx_final))
        results = {(a, t): np.zeros_like(inputs)
                   for a in ALIAS_SCALES_LIST for t in THRESHOLD_LIST}

        settings = [(a, t) for a in ALIAS_SCALES_LIST for t in THRESHOLD_LIST]
        n_units = len(shots) * len(settings)
        print(f"  {traces0.shape[0]} samples, {n_blocks} blocks per stage, "
              f"{n_units} shot-setting units "
              f"({n_units * N_STAGES * n_blocks} block solves)")

        stats = {s: {"time": [], "residual": [], "snr": []} for s in settings}
        t0 = time.time()
        done = 0
        for si, shot in enumerate(shots):
            gather = np.asarray(data[shot, T_START:t_stop], dtype=float)
            traces, reference = prepare(gather)
            inputs[si] = display_grid(traces, nx_final)

            for alias, thr in settings:
                tic = time.time()
                out, residuals = cascade(traces, thr, alias)
                dt_unit = time.time() - tic
                results[(alias, thr)][si] = out

                stats[(alias, thr)]["time"].append(dt_unit)
                stats[(alias, thr)]["residual"].append(residuals[-1])
                line = (f"  shot {shot:4d} alias={alias} thr={thr:5.0%}: "
                        f"{dt_unit:6.1f} s "
                        f"({dt_unit / (N_STAGES * n_blocks):5.2f} s/block), "
                        f"residual per stage "
                        + " ".join(f"{r:.4f}" for r in residuals))
                if reference is not None:
                    snr = snr_db(reference, out - reference)
                    stats[(alias, thr)]["snr"].append(snr)
                    line += f", SNR {snr:6.2f} dB"
                print(line)

                done += 1
                elapsed = time.time() - t0
                print(f"      [{done}/{n_units}] elapsed {fmt_hms(elapsed)}, "
                      f"eta {fmt_hms(elapsed / done * (n_units - done))}")

        print(f"\n  compute finished in {fmt_hms(time.time() - t0)}")
        has_snr = bool(stats[settings[0]]["snr"])
        print(f"  {'alias':>5} {'thr':>5} {'s/shot':>8} {'residual':>9}"
              + (f" {'SNR dB':>8}" if has_snr else ""))
        for setting in settings:
            st = stats[setting]
            line = (f"  {setting[0]:>5} {setting[1]:>5.0%} "
                    f"{np.mean(st['time']):>8.1f} "
                    f"{np.mean(st['residual']):>9.4f}")
            if st["snr"]:
                line += f" {np.mean(st['snr']):>8.2f}"
            print(line)

        np.savez_compressed(
            save_path, inputs=inputs,
            **{f"{a}|{t}": v for (a, t), v in results.items()})
        print(f"  wrote {save_path}")

    t_plot = time.time()
    for si, shot in enumerate(shots):
        p = plot_gathers(shot, inputs[si],
                         {k: v[si] for k, v in results.items()}, out_dir)
        print(f"  wrote {p}")
    p = plot_fk_grid(inputs, results, out_dir)
    print(f"  wrote {p}")
    print(f"  plotting took {fmt_hms(time.time() - t_plot)}")


if __name__ == "__main__":
    main()
