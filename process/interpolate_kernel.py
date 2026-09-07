"""Trace interpolation of the streamer by a fixed kernel, straight from SEG-Y.

    numpy/str_data_raw.npy   (551, 8000, 24) @ 0.5 ms, 3.10 m
    numpy/str_data_geom.npy  (551, 3500, 93) @ 1 ms,   0.77 m

Replaces both the curvelet cascade and the linear interpolation that came
before it.  The cascade was set aside because the recorded traces are mutually
incoherent below about 55 Hz, so nothing can reconstruct that band in the gaps.
Linear was set aside because of what follows.

Why not linear
--------------
Linear interpolation of a dipping event is a comb filter.  Averaging two
neighbours that are Delta-t apart with weights (1-w, w) has response
|1 - w + w exp(2 pi i f Delta-t)|, which at the half-way trace is
|cos(pi f Delta-t)| - a notch at 1 / (2 Delta-t) and a roll-off that starts at
DC.  The direct arrival here moves out 3.1 samples of 0.5 ms per trace, an
apparent velocity near 1950 m/s, so Delta-t = 1.6 ms and the notch lands at
312 Hz.  Measured on the interpolated traces that is -1.2 dB at 100 Hz, -5.5 dB
at 200 Hz and -9.0 dB at 240 Hz, while the recorded traces every 4th column
keep their amplitude untouched - a period-4 amplitude step on the strongest
event in the gather, with the notch distorting the waveform on top of it.

Note that 1 / (2 Delta-t) equals v / (2 dx), the frequency above which that
event aliases on the recorded array: the notch and the alias onset are the same
number, necessarily, since Delta-t = dx / v.

Why sinc fixes it
-----------------
Below the alias onset the wavefield is band-limited in wavenumber, and a
band-limited signal is exactly recoverable at any position by sinc
interpolation - for every dip at once, with no dip estimation anywhere.  The
cos roll-off is an artefact of truncating the kernel to two taps, not something
inherent to interpolating.  A windowed sinc of 2 * HALF_WIDTH taps is flat to
within a fraction of a dB across the band instead; the run prints the achieved
response so the window can be judged rather than assumed.

The premise is that nothing survives above the alias onset.  The steepest event
measured is ~1930 m/s, alias onset 312 Hz, against the 320 Hz low-pass below -
so the top few Hz sit marginally over the line, where the filter is already
past -6 dB and falling. Lower LP_F_CUT in config.FILTER_PARAMS["str"] if that
margin is not wanted.

Both kernels reproduce the recorded traces exactly: at those positions the
kernel argument is zero for one tap and an integer for the rest, and sinc is 1
and 0 there.  23 of the 93 outputs are recorded traces, copied.

Edit the settings block below, then run from the repository root:

    python -m process.interpolate_kernel
"""

import os
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config as C  # noqa: E402
from utils.data import create_memmap, get_project_root, npy_shape  # noqa: E402
from utils.process import f_filter, f_filtering  # noqa: E402

# ---------------------------------------------------------------- settings --

INPUT_NPY = C.ARRAYS["str"]["raw"]
OUTPUT_NPY = C.ARRAYS["str"]["geom"]

# "sinc" or "linear".  linear is kept so the two can be compared; the response
# table the run prints is the comparison.
KERNEL = "sinc"

# Windowed-sinc kernel: 2 * HALF_WIDTH taps, Kaiser-windowed.  Wider is flatter
# but reaches further past the array ends, and there are only 24 traces to
# reach into.  Ignored when KERNEL == "linear".
# hw6/beta6 is flat to within 0.05 % out to 0.75 of the input Nyquist and
# 0.963 at 0.9, against 0.383 and 0.156 for linear.  Wider is flatter still -
# hw8/beta4 holds 1.00 to 0.75 and 0.80 at 0.9 - but 2*hw taps reach hw traces
# past each end of a 24-trace array, so hw8 leaves only a third of the array
# free of edge clamping and hw6 leaves half.  The run prints both the interior
# and the edge response.
HALF_WIDTH = 6
KAISER_BETA = 6.0

# Sample interval of the input, in microseconds, and of the output.
DT_US = 500
OUTPUT_DT_US = C.DT_US

OUT_TRACES = C.STR_TRACES
OUT_SAMPLES = C.N_SAMPLES

# Zeros prepended before the circular low-pass and dropped afterwards, so the
# time origin is unchanged.  In input-rate samples.
PAD_FRONT = 2000

# QC figure; None to skip.
FIGURE_PATH = os.path.join(C.DATA_DIR, "interpolate_kernel.png")
FIGURE_SHOT = 275
FIGURE_SAMPLES = 2000
CLIP_PERC = 99.0

# Apparent velocities to report the interpolation response against.  The first
# is the measured direct-arrival moveout.
VELOCITIES = (1950.0, 1500.0)

# ----------------------------------------------------------------- derived --

_F = C.FILTER_PARAMS["str_bl"]
LOWPASS = _F["lowpass"]
LP_F_CUT = _F["lp_f_cut"]
LP_ORDER = _F["lp_order"]
LP_DECAY = _F["lp_decay"]
HIGHPASS = _F["highpass"]
HP_F_CUT = _F["hp_f_cut"]
HP_ORDER = _F["hp_order"]
HP_DECAY = _F["hp_decay"]

DX_IN_M = C.STR_TRACE_INTERVAL_M * (OUT_TRACES - 1) / (C.STR_NODES - 1)

# ---------------------------------------------------------------------------


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(get_project_root(), path)


def output_positions(n_in, n_out):
    """Where each output trace sits, in units of input trace index."""
    return np.arange(n_out) * (n_in - 1) / (n_out - 1)


def kernel_weights(pos, n_in):
    """Tap indices and weights for each output position.

    Returns (cols, w), both (n_out, n_tap).  Taps that fall off either end are
    clamped to the edge trace and their weights fold onto it, which is gentler
    than zeroing - the alternative puts a step at the array edge.  Weights are
    renormalised to sum to one so a constant is reproduced exactly everywhere,
    edges included.
    """
    if KERNEL == "linear":
        i0 = np.clip(np.floor(pos).astype(int), 0, n_in - 2)
        cols = np.stack([i0, i0 + 1], axis=1)
        frac = (pos - i0)[:, None]
        w = np.concatenate([1.0 - frac, frac], axis=1)
    elif KERNEL == "sinc":
        i0 = np.floor(pos).astype(int)
        taps = np.arange(-HALF_WIDTH + 1, HALF_WIDTH + 1)
        cols = i0[:, None] + taps[None, :]
        x = pos[:, None] - cols
        # Kaiser window evaluated on the kernel argument, zero outside.
        r = np.clip(np.abs(x) / HALF_WIDTH, 0.0, 1.0)
        win = np.i0(KAISER_BETA * np.sqrt(np.maximum(0.0, 1.0 - r ** 2)))
        win /= np.i0(KAISER_BETA)
        w = np.sinc(x) * win
        w[np.abs(x) > HALF_WIDTH] = 0.0
    else:
        raise SystemExit(f'KERNEL must be "sinc" or "linear", not {KERNEL!r}')

    cols = np.clip(cols, 0, n_in - 1)
    w = w / w.sum(axis=1, keepdims=True)
    return cols, w


def interp_matrix(cols, w, n_in, n_out):
    """(n_in, n_out) matrix M with y = g @ M.

    Built once; the per-shot work is then a single small matrix product.
    Clamped taps land on the same column more than once, so accumulate.
    """
    M = np.zeros((n_in, n_out))
    for o in range(n_out):
        np.add.at(M[:, o], cols[o], w[o])
    return M


def plane_wave_response(cols, w, pos, o, frac_k):
    """|H| at k = frac_k * input Nyquist, for output trace o.

    The phase across a tap offset of (cols - pos) traces is
    2 pi k dx (cols - pos), and k = frac_k / (2 dx), so the dx cancels and it
    is pi * frac_k * (cols - pos).  Getting that factor wrong is easy and the
    result looks plausible either way, hence spelling it out.
    """
    return abs(np.sum(w[o] * np.exp(1j * np.pi * frac_k * (cols[o] - pos[o]))))


def report_kernel(cols, w, M, pos, n_in):
    print(f"  kernel {KERNEL}"
          + (f", {2 * HALF_WIDTH} taps, Kaiser beta {KAISER_BETA:g}"
             if KERNEL == "sinc" else ", 2 taps"))

    # Recorded positions must come back untouched: the matrix column there has
    # to be a unit vector.  Checked on the matrix rather than the raw weights
    # because clamped taps fold several of them onto the same column.
    exact = np.flatnonzero(np.isclose(pos, np.round(pos)))
    err = 0.0
    for o in exact:
        e = np.zeros(n_in)
        e[int(round(pos[o]))] = 1.0
        err = max(err, float(np.abs(M[:, o] - e).max()))
    print(f"    {len(exact)} outputs land on a recorded trace and are copied "
          f"exactly (worst error {err:.1e})")

    # Response at the half-way output trace, where any kernel is weakest.
    # Two of them: one well inside the array, and the worst-clamped one near
    # the end, since a wide kernel cannot reach past the last recorded trace.
    frac = pos - np.floor(pos)
    half = np.isclose(frac, 0.5)
    inside = half & (pos > HALF_WIDTH) & (pos < n_in - 1 - HALF_WIDTH)
    o_in = int(np.flatnonzero(inside)[0]) if inside.any() else int(
        np.flatnonzero(half)[len(np.flatnonzero(half)) // 2])
    o_edge = int(np.flatnonzero(half)[0])
    n_clean = int(inside.sum())

    print(f"    response at half-way traces, dx = {DX_IN_M:.3f} m "
          f"({n_clean} of {int(half.sum())} are clear of edge clamping):")
    print("      k/Nyq  " + "".join(f"{v:>9.0f} m/s" for v in VELOCITIES)
          + f"{'interior':>12}{'edge':>10}")
    k_nyq = 1.0 / (2.0 * DX_IN_M)
    for frac_k in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
        hi = plane_wave_response(cols, w, pos, o_in, frac_k)
        he = plane_wave_response(cols, w, pos, o_edge, frac_k)
        freqs = "".join(f"{v * frac_k * k_nyq:>9.0f} Hz" for v in VELOCITIES)
        print(f"      {frac_k:>5.2f}  {freqs}{hi:>12.4f}{he:>10.4f}")
    print("      (a plane wave at wavenumber k is an event of apparent "
          "velocity v at frequency v*k;")
    print("       |H| = 0 at Nyquist is forced - the half-way sample of an "
          "alternating series is zero)")


def build_mask(nt, dt):
    mask = np.ones(nt)
    if LOWPASS:
        mask *= f_filter(nt, dt, LP_F_CUT, LP_ORDER, LP_DECAY, is_lowpass=True)
    if HIGHPASS:
        mask *= f_filter(nt, dt, HP_F_CUT, HP_ORDER, HP_DECAY, is_lowpass=False)
    return mask


def figure(before, after, dt_in, dt_out, path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    clip = np.percentile(np.abs(before), CLIP_PERC)
    axes[0].imshow(before[:FIGURE_SAMPLES], cmap="gray", vmin=-clip, vmax=clip,
                   aspect="auto", interpolation="nearest")
    axes[0].set_title(f"shot {FIGURE_SHOT} in: {before.shape[1]} traces "
                      f"@ {dt_in * 1e3:g} ms")
    axes[0].set_xlabel("trace")
    axes[0].set_ylabel("sample")
    axes[1].imshow(after[:FIGURE_SAMPLES // 2], cmap="gray", vmin=-clip,
                   vmax=clip, aspect="auto", interpolation="nearest")
    axes[1].set_title(f"out: {after.shape[1]} traces @ {dt_out * 1e3:g} ms "
                      f"({KERNEL})")
    axes[1].set_xlabel("trace")

    for g, dt, lab in ((before, dt_in, "in"), (after, dt_out, "out")):
        sp = np.abs(np.fft.rfft(g, axis=0)).mean(axis=1)
        fr = np.fft.rfftfreq(g.shape[0], dt)
        axes[2].plot(fr, 20 * np.log10(np.maximum(sp / sp.max(), 1e-12)),
                     label=lab)
    if LOWPASS:
        axes[2].axvline(LP_F_CUT, color="k", ls="--", lw=1,
                        label=f"{LP_F_CUT:g} Hz")
    axes[2].axvline(VELOCITIES[0] / (2 * DX_IN_M), color="r", ls=":", lw=1,
                    label=f"alias onset {VELOCITIES[0] / (2 * DX_IN_M):.0f} Hz")
    axes[2].set_xlim(0, 0.5 / dt_in)
    axes[2].set_ylim(-100, 5)
    axes[2].set_xlabel("frequency [Hz]")
    axes[2].set_ylabel("dB re peak")
    axes[2].set_title("mean trace spectrum")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    in_path = resolve(INPUT_NPY)
    if not os.path.isfile(in_path):
        raise SystemExit(f"not found: {in_path}")
    out_path = resolve(OUTPUT_NPY)

    data = np.load(in_path, mmap_mode="r")
    n_shots, nt, n_in = data.shape
    dt = DT_US * 1e-6
    dt_out = OUTPUT_DT_US * 1e-6

    if OUTPUT_DT_US % DT_US:
        raise SystemExit(f"OUTPUT_DT_US {OUTPUT_DT_US} is not a whole multiple "
                         f"of DT_US {DT_US}")
    q = OUTPUT_DT_US // DT_US
    need = OUT_SAMPLES * q
    if need > nt:
        raise SystemExit(f"{OUT_SAMPLES} output samples need {need} input "
                         f"samples, the record has {nt}")

    print(f"{INPUT_NPY}\n  {n_shots} shots x {nt} samples @ {DT_US / 1000:g} ms "
          f"x {n_in} traces @ {DX_IN_M:.3f} m")
    print(f"  {n_in} -> {OUT_TRACES} traces @ {C.STR_TRACE_INTERVAL_M:g} m")
    print("  low-pass " + (f"{LP_F_CUT:g} Hz (order {LP_ORDER:g}, "
                           f"decay {LP_DECAY:g})" if LOWPASS else "off")
          + ", low-cut " + (f"{HP_F_CUT:g} Hz" if HIGHPASS else "off"))
    print(f"  keep the first {need} samples, decimate {q}x "
          f"-> {OUT_SAMPLES} @ {OUTPUT_DT_US / 1000:g} ms")

    pos = output_positions(n_in, OUT_TRACES)
    cols, w = kernel_weights(pos, n_in)
    M = interp_matrix(cols, w, n_in, OUT_TRACES)
    report_kernel(cols, w, M, pos, n_in)

    nt_pad = nt + PAD_FRONT
    keep = slice(PAD_FRONT, nt_pad)
    mask = build_mask(nt_pad, dt) if (LOWPASS or HIGHPASS) else None

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if os.path.isfile(out_path):
        prev = npy_shape(out_path)
        print(f"\n  overwriting {OUTPUT_NPY} (was {prev})")
        del prev
    out = create_memmap(out_path, (n_shots, OUT_SAMPLES, OUT_TRACES))
    print(f"  output {out.shape}, {out.nbytes / 1e6:.0f} MB")

    pad = np.zeros((PAD_FRONT, OUT_TRACES))
    before0 = after0 = None
    t0 = time.time()
    for i in range(n_shots):
        g = np.asarray(data[i], dtype=np.float64)
        y = g @ M
        if mask is not None:
            y = f_filtering(np.concatenate([pad, y], axis=0), mask)[keep]
        y = y[:need:q]
        out[i] = y.astype(np.float32)
        if i == FIGURE_SHOT:
            before0, after0 = g, np.asarray(y)
        if (i + 1) % 100 == 0 or i + 1 == n_shots:
            print(f"    {i + 1}/{n_shots} shots  {time.time() - t0:.0f}s elapsed")
    out.flush()

    print(f"\n  wrote {out_path}  shape {out.shape}")
    print(f"  amplitude range [{out[0].min():.4g}, {out[0].max():.4g}] (shot 0)")

    if FIGURE_PATH and before0 is not None:
        p = resolve(FIGURE_PATH)
        figure(before0, after0, dt, dt_out, p)
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
