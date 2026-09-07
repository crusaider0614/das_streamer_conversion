"""Re-filter the interpolated traces and resample them onto the DAS time grid.

Takes `str_data_intp.npy` (551 x 7000 x 93 at 0.5 ms) and writes
`str_data_geom.npy` (551 x 3500 x 93 at 1 ms) - the same 3500 samples the
DAS array uses.

The low-pass has to be applied again, not just carried over from
`process/freq_filter.py`.  The curvelet solve reconstructs from a thresholded
coefficient support, and nothing in that support constrains the temporal band,
so the interpolated gathers come back with energy where the input had none:
measured on shot 0, the mean trace spectrum sits at -65 dB at 240 Hz going in
and -25 dB coming out, and stays around -30 dB well past 300 Hz.  Decimating
without removing that would fold everything from 500 to 1000 Hz straight back
into the signal band.

The filter values are imported from `process/band_limit.py` - the stage that
band-limited the streamer before the interpolation - rather than repeated here,
so "the same low-pass" stays true if that corner is retuned.  Not from
freq_filter.py: that one is the band-pass at the far end of the pipeline, its
corner is per-domain and has nothing to do with what constrained this data.

Edit the settings block below, then run from the repository root:

    python -m process.decimate_intp
"""

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from process.band_limit import LP_DECAY, LP_F_CUT, LP_ORDER  # noqa: E402
from utils.data import create_memmap, get_project_root  # noqa: E402
from scipy.ndimage import uniform_filter1d  # noqa: E402

from utils.process import f_filter, f_filtering  # noqa: E402
from utils import mute as MU  # noqa: E402

# ---------------------------------------------------------------- settings --

INPUT_NPY = os.path.join("data", "pohang_shore", "numpy", "str_data_intp.npy")
OUTPUT_NPY = os.path.join("data", "pohang_shore", "numpy", "str_data_geom.npy")

# Sample interval of the input and of the output, in microseconds.
DT_US = 500
OUTPUT_DT_US = 1000

# Where to apply the low-pass.  Only before: anti-aliasing is by definition a
# pre-decimation operation, and once energy has folded no later filter can
# separate it again.  A second pass after decimation could only remove what the
# first left between 220 and 500 Hz, which is already -30 dB at 230 Hz and
# -120 dB by 240 Hz - nothing.  It is not free either: the response squares, so
# the transition narrows (-40 dB moves from 233 Hz to 226 Hz) and the impulse
# response spreads about 13 % further, 71 ms to 80 ms above -40 dB.  No gain,
# a little more ringing.
FILTER_BEFORE = True
FILTER_AFTER = False

# Zeros added at each end before each circular filter, and removed after, so
# the record length is preserved.  Given in input-rate samples; the
# post-decimation pass uses the same duration at the coarser rate.  Unlike
# freq_filter.py - which pads only the front because it is also trimming the
# tail - nothing is cropped here, so the padding is symmetric.
PAD = 1000

# Correct the period-4 amplitude stripe the cascade leaves behind.
#
# The curvelet solve treats observed and unobserved positions differently: the
# misfit term pins the solution to the data where there is data, while the
# sparsity threshold costs amplitude where there is none.  Measured over 138
# shots, the recorded positions come out about 4 % above a 9-trace running mean
# and both interpolated classes about 1.5 % below it - a 6 % step repeating
# every 4 traces, which the envelope gain downstream makes plainly visible.
#
# The correction is one scalar per trace class - recorded (idx % 4 == 0),
# stage-1 interpolated (idx % 4 == 2), stage-2 interpolated (odd) - taken as
# the median log departure from a running mean over DEBIAS_SHOTS shots and
# normalised to leave the overall amplitude unchanged.  Three numbers for the
# whole survey: it removes the periodic component and cannot touch anything
# aperiodic, which is the point.  Genuine trace-to-trace variation, including
# the recorded array's own ~20 % receiver imbalance, survives untouched.
#
# STAGE_FACTOR and N_STAGES must match what process/interpolate_traces.py ran
# with, since they are what define the classes.
DEBIAS_TRACE_CLASSES = True
DEBIAS_SHOTS = 128
DEBIAS_SMOOTH_TRACES = 9
STAGE_FACTOR = 2
N_STAGES = 2

# Mute ahead of the direct arrival on the way out.  The low-pass above and the
# curvelet solve before it both spread energy in time, so the window is swept
# again after each of them; process/band_limit.py already swept it once ahead
# of the interpolation.
MUTE_AFTER = True

# Print the mean trace spectrum before and after, at these frequencies.
REPORT_HZ = [100, 200, 220, 240, 260, 300, 400, 600, 900]
REPORT_SHOT = 0

# Before/after QC figure; None to skip.
FIGURE_PATH = os.path.join("data", "pohang_shore", "decimate_intp.png")
FIGURE_SAMPLES = 2000
CLIP_PERC = 99.0

# ---------------------------------------------------------------------------


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(get_project_root(), path)


def spectrum_db(gather, dt):
    sp = np.abs(np.fft.rfft(gather, axis=0)).mean(axis=1)
    f = np.fft.rfftfreq(gather.shape[0], dt)
    return f, 20 * np.log10(np.maximum(sp / sp.max(), 1e-12))


def report_response(dt_in, dt_out):
    """The combined effect of the two passes, on a fine frequency grid."""
    n = 1 << 16
    f = np.abs(np.fft.fftfreq(n, dt_in))
    m = np.ones(n)
    if FILTER_BEFORE:
        m *= f_filter(n, dt_in, LP_F_CUT, LP_ORDER, LP_DECAY, is_lowpass=True)
    if FILTER_AFTER:
        # Same corner, evaluated on the coarser grid; below the new Nyquist the
        # shape is identical, so the two simply multiply.
        m *= f_filter(n, dt_in, LP_F_CUT, LP_ORDER, LP_DECAY, is_lowpass=True)

    half = n // 2
    ff, mm = f[:half], m[:half]
    passes = int(FILTER_BEFORE) + int(FILTER_AFTER)
    print(f"  combined response over {passes} pass(es):")
    for p in (180, 200, 210, 220, 230, 240, 260):
        i = int(np.argmin(np.abs(ff - p)))
        print(f"    {p:>5d} Hz: {20 * np.log10(max(mm[i], 1e-300)):8.2f} dB")
    db = 20 * np.log10(np.maximum(mm, 1e-300))
    for target in (-3.0, -40.0):
        past = np.where((ff > LP_F_CUT * 0.5) & (db < target))[0]
        print(f"    crosses {target:6.0f} dB at "
              f"{ff[past[0]] if len(past) else float('nan'):7.1f} Hz")


def report_line(label, gather, dt):
    f, db = spectrum_db(gather, dt)
    nyq = 0.5 / dt
    cells = []
    for p in REPORT_HZ:
        cells.append(f"{db[int(np.argmin(np.abs(f - p)))]:7.1f}" if p < nyq
                     else "      -")
    print(f"    {label:<22}{' '.join(cells)}")


def figure(before, after, dt_in, dt_out, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    clip = np.percentile(np.abs(before), CLIP_PERC)
    axes[0].imshow(before[:FIGURE_SAMPLES], cmap="gray", vmin=-clip, vmax=clip,
                   aspect="auto", interpolation="nearest")
    axes[0].set_title(f"before: {dt_in * 1e3:g} ms")
    axes[1].imshow(after[:FIGURE_SAMPLES // 2], cmap="gray", vmin=-clip,
                   vmax=clip, aspect="auto", interpolation="nearest")
    axes[1].set_title(f"after: filtered + {dt_out * 1e3:g} ms")
    for ax in axes[:2]:
        ax.set_xlabel("trace")
    axes[0].set_ylabel("sample")

    for g, dt, lab in ((before, dt_in, "before"), (after, dt_out, "after")):
        f, db = spectrum_db(g, dt)
        axes[2].plot(f, db, label=lab)
    axes[2].axvline(LP_F_CUT, color="k", ls="--", lw=1, label=f"{LP_F_CUT:g} Hz")
    axes[2].axvline(0.5 / dt_out, color="r", ls=":", lw=1,
                    label=f"new Nyquist {0.5 / dt_out:g} Hz")
    axes[2].set_xlim(0, 1.0 / dt_in / 2); axes[2].set_ylim(-100, 5)
    axes[2].set_xlabel("frequency [Hz]"); axes[2].set_ylabel("dB re peak")
    axes[2].set_title("mean trace spectrum"); axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def trace_classes(n_traces):
    """Which cascade stage introduced each output trace.

    0 is a recorded position, k is a trace the k-th stage created.  With
    STAGE_FACTOR 2 and N_STAGES 2 that is: every 4th trace recorded, the
    halfway traces from stage 1, the odd traces from stage 2.
    """
    j = np.arange(n_traces)
    cls = np.full(n_traces, N_STAGES)
    for k in range(1, N_STAGES + 1):
        cls[j % (STAGE_FACTOR ** k) == 0] = N_STAGES - k
    return cls


def measure_class_gains(data, cls):
    """One gain per trace class, from the median departure from a running mean.

    Log domain, so the statistic is a ratio and the median is insensitive to
    the handful of shots where one trace blows up.  Traces within a half-window
    of either end are left out - the running mean is not centred there.
    """
    n_shots, _, n_traces = data.shape
    shots = (np.arange(n_shots) if DEBIAS_SHOTS is None or DEBIAS_SHOTS >= n_shots
             else np.unique(np.linspace(0, n_shots - 1, DEBIAS_SHOTS).astype(int)))

    dev = np.zeros((len(shots), n_traces))
    for k, s in enumerate(shots):
        r = np.sqrt(np.mean(np.asarray(data[s], dtype=np.float64) ** 2, axis=0))
        lr = np.log(np.maximum(r, 1e-30))
        dev[k] = lr - uniform_filter1d(lr, size=DEBIAS_SMOOTH_TRACES,
                                       mode="nearest")
    med = np.median(dev, axis=0)

    edge = DEBIAS_SMOOTH_TRACES // 2 + 1
    interior = np.zeros(n_traces, dtype=bool)
    interior[edge:n_traces - edge] = True

    gains = np.ones(n_traces)
    print(f"\n  trace-class debias, median over {len(shots)} shots")
    for c in range(N_STAGES + 1):
        sel = interior & (cls == c)
        if not sel.any():
            continue
        m = float(np.median(med[sel]))
        gains[cls == c] = np.exp(-m)
        label = "recorded" if c == 0 else f"stage-{c} interpolated"
        print(f"    class {c} {label:<22} {sel.sum():>3} traces  "
              f"{100 * (np.exp(m) - 1):+6.2f} %  -> gain {np.exp(-m):.4f}")

    # Keep the survey's overall amplitude where it was: only the pattern moves.
    gains /= np.exp(np.mean(np.log(gains)))
    print(f"    normalised gains "
          + ", ".join(f"{g:.4f}" for g in np.unique(gains)))
    return gains


def main():
    in_path = resolve(INPUT_NPY)
    if not os.path.isfile(in_path):
        raise SystemExit(f"not found: {in_path}")
    out_path = resolve(OUTPUT_NPY)

    done_path = os.path.splitext(in_path)[0] + ".done.npy"
    if os.path.isfile(done_path):
        done = np.load(done_path)
        if not done.all():
            raise SystemExit(
                f"{int(done.sum())} of {done.size} shots are interpolated; "
                f"finish process.interpolate_traces first")

    data = np.load(in_path, mmap_mode="r")
    n_shots, nt, n_traces = data.shape
    dt = DT_US * 1e-6

    if OUTPUT_DT_US % DT_US:
        raise SystemExit(f"OUTPUT_DT_US {OUTPUT_DT_US} is not a whole multiple "
                         f"of DT_US {DT_US}")
    q = OUTPUT_DT_US // DT_US
    nt_out = len(range(0, nt, q))

    print(f"{in_path}\n  {n_shots} shots x {nt} samples @ {DT_US / 1000:g} ms "
          f"x {n_traces} traces")
    print(f"  low-pass {LP_F_CUT:g} Hz (order {LP_ORDER:g}, decay {LP_DECAY:g}), "
          f"same values as band_limit.py")
    steps = []
    if FILTER_BEFORE:
        steps.append(f"filter on {nt + 2 * PAD} @ {DT_US / 1000:g} ms")
    steps.append(f"decimate {q}x -> {nt_out}")
    if FILTER_AFTER:
        steps.append(f"filter on {nt_out + 2 * (PAD // q)} @ "
                     f"{OUTPUT_DT_US / 1000:g} ms")
    print(f"  pad {PAD} each end -> " + " -> ".join(steps))
    report_response(dt, OUTPUT_DT_US * 1e-6)

    dt_out = OUTPUT_DT_US * 1e-6
    pad_out = PAD // q

    mask_in = f_filter(nt + 2 * PAD, dt, LP_F_CUT, LP_ORDER, LP_DECAY,
                       is_lowpass=True) if FILTER_BEFORE else None
    mask_out = f_filter(nt_out + 2 * pad_out, dt_out, LP_F_CUT, LP_ORDER,
                        LP_DECAY, is_lowpass=True) if FILTER_AFTER else None
    pad_hi = np.zeros((PAD, n_traces))
    pad_lo = np.zeros((pad_out, n_traces))

    t_mute = None
    if MUTE_AFTER:
        t_mute, _ = MU.boundary_ms(n_traces)
        print(MU.summary(t_mute, nt_out, OUTPUT_DT_US / 1000.0))

    gains = None
    if DEBIAS_TRACE_CLASSES:
        gains = measure_class_gains(data, trace_classes(n_traces))[None, :]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    out = create_memmap(out_path, (n_shots, nt_out, n_traces))

    before0 = after0 = None
    for i in range(n_shots):
        g = np.asarray(data[i], dtype=np.float64)

        y = g
        if mask_in is not None:
            y = f_filtering(np.concatenate([pad_hi, y, pad_hi], axis=0),
                            mask_in)[PAD:PAD + nt]
        y = y[::q]
        if mask_out is not None:
            y = f_filtering(np.concatenate([pad_lo, y, pad_lo], axis=0),
                            mask_out)[pad_out:pad_out + nt_out]

        if gains is not None:
            y = y * gains
        if t_mute is not None:
            y = y * MU.weights(t_mute[i], nt_out, OUTPUT_DT_US / 1000.0)

        out[i] = y.astype(np.float32)
        if i == REPORT_SHOT:
            before0, after0 = g, np.asarray(y)
        if (i + 1) % 100 == 0 or i + 1 == n_shots:
            print(f"    {i + 1}/{n_shots} shots")
    out.flush()

    print(f"\n  wrote {out_path}")
    print(f"  shape {out.shape}, {out.nbytes / 1e6:.0f} MB")

    if before0 is not None:
        print(f"\n  mean trace spectrum, shot {REPORT_SHOT} "
              f"(dB re peak, at {' '.join(f'{p:>6d}' for p in REPORT_HZ)} Hz)")
        report_line("before", before0, dt)
        report_line("after", after0, OUTPUT_DT_US * 1e-6)

        if FIGURE_PATH:
            p = resolve(FIGURE_PATH)
            figure(before0, after0, dt, OUTPUT_DT_US * 1e-6, p)
            print(f"  wrote {p}")


if __name__ == "__main__":
    main()
