"""Band-pass an array on the common geometry grid.

Applies the project's `utils.process.f_filter` masks - a low-cut and a
low-pass - to the time axis.  As configured it is the DAS `_geom -> _freq`
stage:

    numpy/das_data_geom.npy  (551, 3500, 1058) @ 1 ms
    numpy/das_data_freq.npy  (551, 3500, 1058) @ 1 ms

Length and time origin are preserved: the only samples removed are the ones
padded on to keep the circular filter from wrapping.  Set DOMAIN = "str" to do
the streamer side instead; the paths, the sample interval, the trace spacing
and the corners all come from `config.py`, so nothing else has to change.

Two things this script does, independently
------------------------------------------
PREVIEW filters PREVIEW_SHOTS only, holds the result in memory and draws it:
the mask, the mean spectrum before and after, and each shot as
before / after / removed.  Nothing is written.  This is the setting-the-corners
mode; it is off now that they are set.

WRITE_ALL filters every shot and streams it into OUTPUT_NPY.  This is what runs
by default.  Turning both on previews first and then writes.

What the low-pass is for here
-----------------------------
It is no longer an anti-alias measure.  That was its job in the streamer's
pre-interpolation pass, where the 3.12 m group interval put the spatial Nyquist
at 0.160 cycles/m and the 1500 m/s water arrival folded over above 240 Hz.  On
the 0.75 m DAS grid the spatial Nyquist is 0.667 cycles/m and the same arrival
would only alias above 1000 Hz - twice the 500 Hz temporal Nyquist, so it
cannot.  Whatever is cut here is cut because it is noise.

Choosing ORDER and DECAY
------------------------
The mask is  m(f) = (1 + 2 ** (order * x)) ** (-decay / order),  x = f - F_CUT
for the low-pass and x = F_CUT - f for the low-cut.  Two consequences worth
keeping in mind:

  * m(F_CUT) = 2 ** (-decay / order), so decay == order puts the -6 dB point
    exactly at F_CUT.
  * far from the corner the roll-off is 6 * decay dB per Hz, so `decay` alone
    fixes the transition width.

`decay = 5` therefore means 30 dB per Hz: a 3 Hz transition band, which on a
3.5 s record is a dozen frequency bins and rings audibly in time.  If the
"after" panel grows a wavetrain behind every strong arrival, widen the
transition by lowering DECAY before blaming the data.

Edit the settings block below, then run from the repository root:

    python -m process.freq_filter
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np

import config as C
from utils.data import create_memmap, get_project_root, npy_shape
from utils.process import f_filter, f_filtering
from utils import mute as MU

# ---------------------------------------------------------------- settings --

# Which side to filter: "das" or "str".  The paths, the sample interval, the
# trace spacing and the corners all follow from it - see config.ARRAYS,
# config.DX_M and config.FILTER_PARAMS.  Change the corners there, not here.
DOMAIN = "str"

# Filter a few shots and draw them.  Writes nothing.  Turn this back on to
# re-tune the corners; it costs a few seconds and shows what WRITE_ALL is about
# to do to all 551.
PREVIEW = True
PREVIEW_SHOTS = range(0, 551, 50)

# Filter every shot and write the output.  The full pass over the 1058-channel
# DAS array reads 8.2 GB and writes 8.2 GB back.
WRITE_ALL = False

# Mute ahead of the direct arrival on the way out, the filter having just
# spread energy in time.
MUTE_AFTER = True

# Resample the time axis after filtering.  None keeps the input grid.  Must be
# a whole multiple of the input's sample interval.
OUTPUT_DT_US = None

# Apparent velocities whose alias onset is worth marking on the response plot.
VELOCITIES = [1500.0, 2000.0]

# Print the amplitude response at a set of probe frequencies.
REPORT_RESPONSE = True
PROBE_HZ = [1, 2, 3, 5, 8, 10, 20, 50, 100, 150, 200, 220, 250, 300, 400, 500]

# What the preview panels show.  TRACES = None draws every trace; on the 1058
# channel DAS array a slice is easier to read.  SAMPLES caps the time axis;
# None draws the whole record, which is what you want when checking the tail
# for wrap-around the padding failed to absorb.
TRACES = None
SAMPLES = None
CLIP_PERC = 99.0
SHOW_DIFFERENCE = True

# Show the preview interactively, and optionally save it.  Both only matter
# when PREVIEW is on; a falsy SAVE_DIR skips saving.
SHOW = True
SAVE_DIR = None

# ----------------------------------------------------------------- derived --
# Unpacked from config so DOMAIN is the only thing to change when switching
# sides.  DT_US in particular has to match the array: the frequency axis is
# fftfreq(nt, DT_US), so a wrong value silently rescales every corner.

INPUT_NPY = C.ARRAYS[DOMAIN]["raw"]
OUTPUT_NPY = C.ARRAYS[DOMAIN]["freq"]
DT_US = C.DT_US
DX_M = C.DX_M[DOMAIN]

_FILTER = C.FILTER_PARAMS[DOMAIN]
HIGHPASS = _FILTER["highpass"]
HP_F_CUT = _FILTER["hp_f_cut"]
HP_ORDER = _FILTER["hp_order"]
HP_DECAY = _FILTER["hp_decay"]
LOWPASS = _FILTER["lowpass"]
LP_F_CUT = _FILTER["lp_f_cut"]
LP_ORDER = _FILTER["lp_order"]
LP_DECAY = _FILTER["lp_decay"]
ZERO_DC = _FILTER["zero_dc"]
PAD_FRONT = _FILTER["pad_front"]

# ---------------------------------------------------------------------------


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(get_project_root(), path)


def build_mask(nt, dt):
    """The combined low-cut x low-pass mask, on the fftfreq grid."""
    mask = np.ones(nt)
    if LOWPASS:
        mask *= f_filter(nt, dt, LP_F_CUT, LP_ORDER, LP_DECAY, is_lowpass=True)
    if HIGHPASS:
        mask *= f_filter(nt, dt, HP_F_CUT, HP_ORDER, HP_DECAY, is_lowpass=False)
    if ZERO_DC:
        mask = mask.copy()
        mask[0] = 0.0
    return mask


def report(mask, nt, dt):
    f = np.abs(np.fft.fftfreq(nt, dt))
    half = nt // 2
    print("  amplitude response:")
    for p in PROBE_HZ:
        i = int(np.argmin(np.abs(f[:half] - p)))
        db = 20 * np.log10(max(mask[i], 1e-300))
        print(f"    {p:>5.0f} Hz: {mask[i]:9.6f}  ({db:8.2f} dB)")

    ff, db = f[:half], 20 * np.log10(np.maximum(mask[:half], 1e-300))
    if LOWPASS:
        for target in (-3.0, -40.0, -80.0):
            past = np.where((ff > LP_F_CUT * 0.5) & (db < target))[0]
            edge = ff[past[0]] if len(past) else float("nan")
            print(f"  low-pass crosses {target:6.0f} dB at {edge:7.1f} Hz")
    if HIGHPASS:
        for target in (-3.0, -40.0):
            below = np.where((ff < HP_F_CUT * 2.0) & (db < target))[0]
            edge = ff[below[-1]] if len(below) else float("nan")
            print(f"  low-cut  crosses {target:6.0f} dB at {edge:7.1f} Hz")

    k_nyq = 1.0 / (2.0 * DX_M)
    f_nyq = 0.5 / dt
    print(f"  temporal Nyquist {f_nyq:.0f} Hz at dt = {dt * 1000:g} ms")
    print(f"  spatial Nyquist {k_nyq:.4f} c/m at dx = {DX_M:g} m; alias onset:")
    for v in VELOCITIES:
        onset = v * k_nyq
        note = "  (above temporal Nyquist - cannot alias)" if onset > f_nyq else ""
        print(f"    v = {v:6.0f} m/s -> {onset:6.1f} Hz{note}")


def filter_gather(gather, mask, keep, q):
    """Pad, filter, drop the pad, decimate.  Returns the unfiltered gather put
    through the same steps alongside the filtered one, so the two line up
    sample for sample and their difference is what the filter removed."""
    pad = np.zeros((PAD_FRONT, gather.shape[1]), dtype=np.float64)
    padded = np.concatenate([pad, gather], axis=0)
    after = f_filtering(padded, mask, is_zeroout=False)[keep]
    before = padded[keep]
    return before[::q], after[::q]


def response_figure(mask, nt, dt, spectra):
    """Filter response, and the mean trace spectrum before and after."""
    f = np.abs(np.fft.fftfreq(nt, dt))
    half = nt // 2
    order = np.argsort(f[:half])
    f_nyq = 0.5 / dt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(f[:half][order],
                 20 * np.log10(np.maximum(mask[:half][order], 1e-300)))
    if LOWPASS:
        axes[0].axvline(LP_F_CUT, color="k", ls="--", lw=1,
                        label=f"low-pass {LP_F_CUT:g} Hz")
    if HIGHPASS:
        axes[0].axvline(HP_F_CUT, color="b", ls="--", lw=1,
                        label=f"low-cut {HP_F_CUT:g} Hz")
    for v in VELOCITIES:
        onset = v / (2 * DX_M)
        if onset <= f_nyq:
            axes[0].axvline(onset, color="r", ls=":", lw=1,
                            label=f"{v:g} m/s alias onset")
    axes[0].set_xlim(0, f_nyq)
    axes[0].set_ylim(-100, 5)
    axes[0].set_xlabel("frequency [Hz]")
    axes[0].set_ylabel("dB")
    axes[0].set_title("filter response")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    for lab, (fr, sp) in spectra.items():
        axes[1].plot(fr, sp, label=lab)
    axes[1].set_xlim(0, f_nyq)
    axes[1].set_ylim(-100, 5)
    axes[1].set_xlabel("frequency [Hz]")
    axes[1].set_ylabel("dB re peak")
    axes[1].set_title(f"mean trace spectrum, {len(PREVIEW_SHOTS)} shots averaged")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return fig


def gather_figure(panels):
    """Rows of before / after / removed, one row per shot."""
    n_col = 3 if SHOW_DIFFERENCE else 2
    fig, axes = plt.subplots(len(panels), n_col,
                             figsize=(4.2 * n_col, 4.6 * len(panels)),
                             squeeze=False)
    for row, (shot, before, after) in enumerate(panels):
        clip = np.percentile(np.abs(before), CLIP_PERC)
        cols = [(before, "before"), (after, "after")]
        if SHOW_DIFFERENCE:
            cols.append((before - after, "removed"))
        for col, (img, name) in enumerate(cols):
            ax = axes[row][col]
            ax.imshow(img, cmap="gray", vmin=-clip, vmax=clip,
                      aspect="auto", interpolation="nearest")
            ax.set_title(f"shot {shot} {name}", fontsize=9)
            ax.set_xlabel("trace")
            if col == 0:
                ax.set_ylabel("sample")
    low_cut = f"{HP_F_CUT:g} Hz" if HIGHPASS else "off"
    low_pass = f"{LP_F_CUT:g} Hz" if LOWPASS else "off"
    fig.suptitle(f"low-cut {low_cut}, low-pass {low_pass}"
                 f"  (panels clipped at {CLIP_PERC:g}% of 'before')")
    fig.tight_layout()
    return fig


def preview(data, mask, keep, q, dt, nt_out):
    """Filter PREVIEW_SHOTS, draw them, write nothing."""
    n_shots = data.shape[0]
    bad = [s for s in PREVIEW_SHOTS if not 0 <= s < n_shots]
    if bad:
        raise SystemExit(f"preview shots out of range 0..{n_shots - 1}: {bad}")

    dt_out = dt * q
    fr = np.fft.rfftfreq(nt_out, dt_out)
    acc_before = np.zeros(len(fr))
    acc_after = np.zeros(len(fr))
    panels = []
    ns = nt_out if SAMPLES is None else min(SAMPLES, nt_out)
    tr = slice(None) if TRACES is None else TRACES

    print(f"\n  preview: {len(PREVIEW_SHOTS)} shots, nothing written")
    for shot in PREVIEW_SHOTS:
        before, after = filter_gather(np.asarray(data[shot], dtype=np.float64),
                                      mask, keep, q)
        acc_before += np.abs(np.fft.rfft(before, axis=0)).mean(axis=1)
        acc_after += np.abs(np.fft.rfft(after, axis=0)).mean(axis=1)

        rms_b = float(np.sqrt(np.mean(before ** 2)))
        rms_a = float(np.sqrt(np.mean(after ** 2)))
        change = 20 * np.log10(max(rms_a, 1e-300) / max(rms_b, 1e-300))
        print(f"    shot {shot:>4}: rms {rms_b:.4g} -> {rms_a:.4g} "
              f"({change:+.2f} dB)")

        panels.append((shot, before[:ns, tr], after[:ns, tr]))

    peak = max(acc_before.max(), 1e-300)
    spectra = {
        "before": (fr, 20 * np.log10(np.maximum(acc_before / peak, 1e-12))),
        "after": (fr, 20 * np.log10(np.maximum(acc_after / peak, 1e-12))),
    }

    figs = [("response", response_figure(mask, len(mask), dt, spectra)),
            ("gathers", gather_figure(panels))]

    if SAVE_DIR:
        out_dir = resolve(SAVE_DIR)
        os.makedirs(out_dir, exist_ok=True)
        for name, fig in figs:
            p = os.path.join(out_dir, f"freq_filter_{name}.png")
            fig.savefig(p, dpi=130)
            print(f"  wrote {p}")

    if SHOW:
        plt.show()
    else:
        for _, fig in figs:
            plt.close(fig)


def write_all(data, mask, keep, q, nt_out):
    """Filter every shot into OUTPUT_NPY."""
    n_shots, _, n_traces = data.shape
    out_path = resolve(OUTPUT_NPY)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    if os.path.isfile(out_path):
        prev = npy_shape(out_path)
        print(f"\n  overwriting {out_path} (was {prev})")
        del prev

    out = create_memmap(out_path, (n_shots, nt_out, n_traces))
    print(f"  output {out.shape}, {out.nbytes / 1e9:.2f} GB")

    t_mute = None
    if MUTE_AFTER:
        dt_out_ms = (OUTPUT_DT_US or DT_US) / 1000.0
        t_mute, _ = MU.boundary_ms(n_traces)
        print(MU.summary(t_mute, nt_out, dt_out_ms))

    t0 = time.time()
    for i in range(n_shots):
        _, after = filter_gather(np.asarray(data[i], dtype=np.float64),
                                 mask, keep, q)
        if t_mute is not None:
            after = after * MU.weights(t_mute[i], nt_out, dt_out_ms)
        out[i] = after.astype(np.float32)
        if (i + 1) % 50 == 0 or i + 1 == n_shots:
            el = time.time() - t0
            print(f"    {i + 1}/{n_shots} shots  {el:.0f}s elapsed, "
                  f"eta {el / (i + 1) * (n_shots - i - 1):.0f}s")
    out.flush()
    print(f"\n  wrote {out_path}")


def main():
    in_path = resolve(INPUT_NPY)
    if not os.path.isfile(in_path):
        raise SystemExit(f"not found: {in_path}")

    data = np.load(in_path, mmap_mode="r")
    if data.ndim != 3:
        raise SystemExit(f"expected (shots, samples, traces), got {data.shape}")
    n_shots, nt, n_traces = data.shape
    dt = DT_US * 1e-6

    nt_pad = nt + PAD_FRONT
    keep = slice(PAD_FRONT, nt_pad)

    q = 1
    if OUTPUT_DT_US is not None:
        if OUTPUT_DT_US % DT_US:
            raise SystemExit(f"OUTPUT_DT_US {OUTPUT_DT_US} is not a whole "
                             f"multiple of DT_US {DT_US}")
        q = OUTPUT_DT_US // DT_US
    nt_out = len(range(0, keep.stop - keep.start, q))

    low_cut = (f"{HP_F_CUT:g} Hz (order {HP_ORDER:g}, decay {HP_DECAY:g})"
               if HIGHPASS else "off")
    low_pass = (f"{LP_F_CUT:g} Hz (order {LP_ORDER:g}, decay {LP_DECAY:g})"
                if LOWPASS else "off")

    print(f"{INPUT_NPY}\n  {n_shots} shots x {nt} samples @ {DT_US / 1000:g} ms "
          f"x {n_traces} traces @ {DX_M:g} m")
    print(f"  pad {PAD_FRONT} in front -> filter on {nt_pad} -> "
          f"drop the pad -> {nt_out} samples")
    print(f"  low-cut  {low_cut}")
    print(f"  low-pass {low_pass}")
    print(f"  zero DC  {ZERO_DC}")
    if q > 1:
        print(f"  decimating {q}x after filtering -> {OUTPUT_DT_US / 1000:g} ms")

    mask = build_mask(nt_pad, dt)
    if REPORT_RESPONSE:
        report(mask, nt_pad, dt)

    if PREVIEW:
        preview(data, mask, keep, q, dt, nt_out)

    if WRITE_ALL:
        write_all(data, mask, keep, q, nt_out)
    else:
        print(f"\n  WRITE_ALL is off - {OUTPUT_NPY} not touched")


if __name__ == "__main__":
    main()
