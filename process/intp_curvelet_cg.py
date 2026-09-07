"""Curvelet-domain trace interpolation by nonlinear conjugate gradient.

Port of `hj_code/curvelet_interpolation/intp_CG.m`.  Every second trace is
missing; the algorithm finds curvelet coefficients whose inverse transform
matches the recorded traces, restricted to a support mask estimated from the
decimated data itself.

    minimise  ||S F* (C .* mask) - d||^2 + mu ||C||^2

with S the sampling operator (take the even columns), F* the inverse curvelet
transform, and d the recorded traces.  Solved by Polak-Ribiere CG on the
curvelet coefficients: the first step is steepest descent, then
`N_ITER` conjugate steps.

The anti-aliasing idea is in the mask.  Aliasing grows with frequency, so the
support is thresholded honestly only on the `ALIAS_SCALES` coarsest scales,
where the decimated data are still unaliased; finer scales inherit their mask
by nearest-neighbour upsampling from the next coarser scale, which carries the
unaliased dip information up into the aliased band.

Edit the settings block below, then run from the repository root:

    python -m process.intp_curvelet_cg


A note on `is_real`, carried over from the MATLAB
-------------------------------------------------
intp_CG.m calls the forward transform with is_real=0 but the inverse with
is_real=1 (except in the initial steepest-descent step, which uses 0).  Those
are not an adjoint pair: with is_real=1 the inverse reads C{j}{l} and
C{j}{l+nbangles/2} as the real and imaginary parts of one complex curvelet,
which is not how the is_real=0 forward laid them out.  The defaults below
reproduce the MATLAB exactly, so the numbers match what you already have.  Set
IS_REAL_INVERSE = 0 to make the transform pair consistent, which is what the
CG derivation above actually assumes - worth comparing the two.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from utils.data import get_project_root
from utils.fdct_wrapping import fdct_wrapping, ifdct_wrapping

# ---------------------------------------------------------------- settings --

# Input: (n_shots, n_time, n_trace) array of fully sampled gathers, used both
# as the source of the decimated input and as the reference for the SNR.
# A relative path resolves against the project root.
INPUT_NPY = os.path.join("data", "pohang_shore", "numpy", "str_data_raw.npy")

# Where the interpolated volume is written; None to skip saving.
OUTPUT_NPY = os.path.join("data", "pohang_shore", "numpy", "str_data_intp.npy")

# Shots to process: every STEP-th from START to STOP (STOP = None -> all).
START, STOP, STEP = 0, 4, 1

# Curvelet transform parameters (MATLAB scale0 / angle).
NBSCALES = 8
NBANGLES_COARSE = 16
FINEST = 1                 # 1: curvelets at the finest scale, 2: wavelets

# is_real for the two transforms.  (0, 1) reproduces intp_CG.m; (0, 0) makes
# them a consistent adjoint pair.  See the note at the end of the docstring.
IS_REAL_FORWARD = 0
IS_REAL_INVERSE = 1

# Support mask: keep coefficients above this fraction of the per-scale maximum
# (MATLAB thres/100, e.g. 5 -> 0.05).
THRESHOLD = 0.05

# Number of coarsest scales whose mask is thresholded directly.  Above this,
# the mask is upsampled from the next coarser scale (MATLAB `alias`).
ALIAS_SCALES = 5

# CG: mu is the Tikhonov weight, alpha the step length, N_ITER the number of
# conjugate steps after the initial steepest-descent step.
MU = 0.0
ALPHA = 1.0
N_ITER = 20

# Decimation: keep every DECIMATE-th trace.  The MATLAB keeps columns
# 2,4,6,... of a 1-based array, i.e. index 1,3,5,... 0-based, so the offset is
# DECIMATE-1.
DECIMATE = 2

# Print the SNR of each shot against the reference as it finishes.
REPORT_SNR = True

# Plot decimated / interpolated / reference for this shot; None to skip.
PLOT_SHOT = 0
CLIP_PERC = 99.0

# ---------------------------------------------------------------------------


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(get_project_root(), path)


def snr_db(signal, noise):
    """MATLAB's snr(signal, noise) = 10 log10(P_signal / P_noise), in dB."""
    ps = float(np.sum(np.abs(signal) ** 2))
    pn = float(np.sum(np.abs(noise) ** 2))
    return 10.0 * np.log10(ps / pn) if pn > 0 else np.inf


def _nearest_axis(n_out, n_in):
    """MATLAB imresize(...,'nearest') index map for one axis.

    Output pixel i (1-based) maps to the continuous source coordinate
    u = (i - 0.5) * n_in / n_out + 0.5 and takes round(u), clamped to the
    input range.  Returned 0-based.
    """
    u = (np.arange(1, n_out + 1) - 0.5) * n_in / n_out + 0.5
    src = np.floor(u + 0.5)                       # round, halves away from zero
    return np.clip(src, 1, n_in).astype(int) - 1


def imresize_nearest(mask, shape):
    """Nearest-neighbour resize matching MATLAB's imresize(...,'nearest')."""
    out_h, out_w = shape
    in_h, in_w = mask.shape
    return mask[np.ix_(_nearest_axis(out_h, in_h), _nearest_axis(out_w, in_w))]


def zeros_like_coeffs(C):
    return [[np.zeros_like(w) for w in scale] for scale in C]


def build_mask(C):
    """Support mask per scale/wedge.

    The coarsest ALIAS_SCALES scales are thresholded at THRESHOLD times the
    largest coefficient magnitude in that scale; finer scales inherit an
    upsampled copy of the next coarser scale's mask.
    """
    cutoff = [THRESHOLD * max(np.abs(w).max() for w in scale) for scale in C]

    mask = [None] * len(C)
    for s in range(min(ALIAS_SCALES, len(C))):
        mask[s] = [(np.abs(w) > cutoff[s]).astype(float) for w in C[s]]

    for s in range(ALIAS_SCALES, len(C)):
        l1, l2 = len(C[s]), len(C[s - 1])
        # MATLAB: n = uint8(l1/l2); a = w/n with n a uint8, so the division
        # rounds (half away from zero) instead of erroring on odd w.
        n = int(np.floor(l1 / l2 + 0.5))
        mask[s] = []
        for w in range(1, l1 + 1):
            a = max(1, min(l2, int(np.floor(w / n + 0.5)) if n else w))
            mask[s].append(imresize_nearest(mask[s - 1][a - 1], C[s][w - 1].shape))
    return mask


def apply_mask(C, mask):
    return [[c * m for c, m in zip(cs, ms)] for cs, ms in zip(C, mask)]


def coeff_axpy(A, alpha, B):
    """A + alpha * B, coefficient-wise."""
    return [[a + alpha * b for a, b in zip(cs, bs)] for cs, bs in zip(A, B)]


def gradient(Ct, mask, sampled, nt, nx, offset):
    """Gradient of the objective at Ct, and the modelled traces.

    grad = 2 mu Ct - mask .* F(S* (d - S F* (mask .* Ct)))
    """
    m = ifdct_wrapping(apply_mask(Ct, mask), IS_REAL_INVERSE, nt, nx)
    modelled = np.real(m[:, offset::DECIMATE])
    misfit = sampled - modelled

    null = np.zeros((nt, nx))
    null[:, offset::DECIMATE] = misfit
    Cnull = fdct_wrapping(null, IS_REAL_FORWARD, FINEST, NBSCALES, NBANGLES_COARSE)

    grad = [[2 * MU * t - n * mk
             for t, n, mk in zip(ts, ns, ms)]
            for ts, ns, ms in zip(Ct, Cnull, mask)]
    return grad, misfit


def coeff_dot(A, B):
    """sum(conj(A) .* B) over every coefficient, as MATLAB's
    sum(conj(...) .* ..., 'all') accumulated across the cell array."""
    return sum(complex(np.sum(np.conj(a) * b)) for cs, bs in zip(A, B)
               for a, b in zip(cs, bs))


def interpolate(gather):
    """Interpolate one (n_time, n_trace) gather whose traces are decimated."""
    nt, nx = gather.shape
    offset = DECIMATE - 1

    decimated = np.zeros_like(gather)
    decimated[:, offset::DECIMATE] = gather[:, offset::DECIMATE]
    sampled = gather[:, offset::DECIMATE]

    Ci = fdct_wrapping(decimated, IS_REAL_FORWARD, FINEST, NBSCALES,
                       NBANGLES_COARSE)
    mask = build_mask(Ci)

    # Initial steepest-descent step from Ct = 0.
    Ct = zeros_like_coeffs(Ci)
    grad_prev, _ = gradient(Ct, mask, sampled, nt, nx, offset)
    direction = [[-g for g in gs] for gs in grad_prev]
    Ct = coeff_axpy(Ct, ALPHA, direction)

    for _ in range(N_ITER):
        grad, _ = gradient(Ct, mask, sampled, nt, nx, offset)

        # Polak-Ribiere beta, exactly as the MATLAB forms it.
        num = coeff_dot([[g - gp for g, gp in zip(gs, gps)]
                         for gs, gps in zip(grad, grad_prev)], grad)
        den = coeff_dot(grad_prev, grad_prev)
        beta = num / den if den != 0 else 0.0

        direction = [[-g + beta * d for g, d in zip(gs, ds)]
                     for gs, ds in zip(grad, direction)]
        Ct = coeff_axpy(Ct, ALPHA, direction)
        grad_prev = grad

    out = ifdct_wrapping(Ct, IS_REAL_INVERSE, nt, nx)
    return np.real(out), decimated


def show(decimated, interpolated, reference):
    clip = np.percentile(np.abs(reference), CLIP_PERC)
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    for ax, img, title in zip(
            axes,
            [decimated, interpolated, reference],
            ["decimated input", "interpolated", "reference"]):
        ax.imshow(np.real(img), cmap="gray", vmin=-clip, vmax=clip,
                  aspect="auto", interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("trace")
    axes[0].set_ylabel("sample")
    fig.tight_layout()
    plt.show()


def main():
    path = resolve(INPUT_NPY)
    if not os.path.isfile(path):
        raise SystemExit(f"not found: {path}")

    data = np.load(path, mmap_mode="r")
    if data.ndim != 3:
        raise SystemExit(f"expected (shots, samples, traces), got {data.shape}")
    n_shots, nt, nx = data.shape
    print(f"{path}\n  {n_shots} shots x {nt} samples x {nx} traces")

    stop = n_shots if STOP is None else min(STOP, n_shots)
    indices = list(range(START, stop, STEP))
    print(f"  interpolating {len(indices)} shots, "
          f"{NBSCALES} scales / {NBANGLES_COARSE} angles, "
          f"threshold {THRESHOLD:g}, {ALIAS_SCALES} unaliased scales, "
          f"{N_ITER} CG iterations")

    out = np.zeros((len(indices), nt, nx), dtype=np.float32)
    snrs = []
    shown = None
    for k, idx in enumerate(indices):
        reference = np.asarray(data[idx], dtype=float)
        interpolated, decimated = interpolate(reference)
        out[k] = interpolated.astype(np.float32)

        if REPORT_SNR:
            s = snr_db(reference, interpolated - reference)
            s0 = snr_db(reference, decimated - reference)
            snrs.append(s)
            print(f"  shot {idx:4d}: {s0:7.2f} dB decimated -> {s:7.2f} dB interpolated")
        if idx == PLOT_SHOT:
            shown = (decimated, interpolated, reference)

    if snrs:
        print(f"  mean SNR over {len(snrs)} shots: {np.mean(snrs):.2f} dB")

    if OUTPUT_NPY:
        out_path = resolve(OUTPUT_NPY)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        np.save(out_path, out)
        print(f"wrote {out_path}  shape {out.shape}")

    if shown is not None:
        show(*shown)


if __name__ == "__main__":
    main()
