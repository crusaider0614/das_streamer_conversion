"""Try a band-pass on the decimated arrays and look at what it does.

A preview, not a stage: nothing is written to disk.  Pick a domain, set the
low-cut and low-pass corners, run it, and look at the gathers and spectra
before and after.  When the corners are settled, copy them into
`process/freq_filter.py` (or whatever writes the filtered array) to apply them
for real.

    das  numpy/das_data_geom.npy  (551, 3500, 1058) @ 1 ms, 0.75 m
    str  numpy/str_data_geom.npy  (551, 3500,   93) @ 1 ms, 0.77 m

Why the low-pass means something different here
-----------------------------------------------
In `freq_filter.py` the 220 Hz low-pass was there to stop the 3.12 m streamer
sampling from folding steep events over during the interpolation.  That reason
is gone: at 0.75 m the spatial Nyquist is 0.667 cycles/m, so the 1500 m/s water
arrival would only alias above 1000 Hz - twice the 500 Hz temporal Nyquist, so
it cannot.  Whatever low-pass is applied now is about noise, not aliasing.

The two domains also arrive in different states.  The streamer has already been
through the 220 Hz low-pass twice (once before the interpolation, once in
`decimate_intp.py`), so it has nothing left up there to remove.  The DAS array
has never been filtered at all: it comes straight off the SEG-Y through a
3-channel average, which is a mild spatial low-pass and nothing in time.  Expect
the two to want different corners, and expect the DAS spectrum to be the
informative one.

The mask
--------
Both sides use `utils.process.f_filter`:

    m(f) = (1 + 2 ** (order * x)) ** (-decay / order)

with x = f - F_CUT for the low-pass and x = F_CUT - f for the low-cut.  Two
consequences worth keeping in mind:

  * m(F_CUT) = 2 ** (-decay / order), so decay == order puts -6 dB at F_CUT.
  * far from the corner the roll-off is 6 * decay dB per Hz, so `decay` alone
    fixes the transition width - decay = 1 is 6 dB/Hz, decay = 0.5 is 3 dB/Hz.

A sharp corner rings.  On a 3.5 s record a 3 Hz transition band is about ten
frequency bins, which shows up in the gather as a wavetrain trailing every
strong arrival; if the "after" panel looks ripply, widen the transition by
lowering DECAY before blaming the data.

Edit the settings block below, then run from the repository root:

    python -m process.freq_filter_preview
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from utils.data import get_project_root
from utils.process import f_filter, f_filtering

# ---------------------------------------------------------------- settings --

# Which array to look at: "das" or "str".
DOMAIN = "das"

# Per-domain input path and trace spacing.  Relative paths resolve against the
# project root.
SOURCES = {
    "das": (os.path.join("data", "pohang_shore", "numpy", "das_data_geom.npy"),
            0.75),
    "str": (os.path.join("data", "pohang_shore", "numpy",
                         "str_data_geom.npy"), 0.7745),
}

# Sample interval of both arrays, in microseconds.
DT_US = 1000

# Shots to preview.  A few spread over the survey show more than a few next to
# each other.  Keep the list short - rendering the 1058-trace DAS panels costs
# more than filtering them.
SHOTS = (0, 275, 550)

# Low-cut.  Swell noise and the DAS instrument's low-frequency drift live down
# here; 3-5 Hz is the usual starting point for marine data.
HIGHPASS = True
HP_F_CUT = 5.0
HP_ORDER = 1.0
HP_DECAY = 1.0

# Low-pass.  No longer an anti-alias measure (see above) - set it where the
# signal actually ends, which the "before" spectrum will show.
LOWPASS = True
LP_F_CUT = 220.0
LP_ORDER = 0.5
LP_DECAY = 0.5

# Zero the DC bin outright.  Cheap way to remove a constant trace offset that
# a finite-width low-cut leaves behind.
ZERO_DC = True

# f_filtering transforms the whole trace at once, so the filter is circular:
# without room the tail of the record wraps into its head.  PAD_FRONT zeros are
# prepended to absorb it and the same number are dropped afterwards, so the
# time origin is unchanged.  Make it a few times the impulse response length -
# 1000 samples is 1 s, ample for a 3 Hz transition band.
PAD_FRONT = 1000

# What the gather panels show.  TRACES = None draws every trace; on the 1058
# channel DAS array a slice is easier to read.  SAMPLES caps the time axis;
# None draws the whole record.
TRACES = None
SAMPLES = None
CLIP_PERC = 99.0

# Difference panel next to before/after.  It is the only place a small change
# is visible once both panels are clipped the same way.
SHOW_DIFFERENCE = True

# Print the amplitude response at these frequencies.
REPORT_RESPONSE = True
PROBE_HZ = [1, 2, 3, 5, 8, 10, 20, 50, 100, 150, 200, 220, 250, 300, 400, 500]

# Show the figures interactively, and optionally save them.  SAVE_DIR = None
# skips saving.
SHOW = True
SAVE_DIR = None

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


def report(mask, nt, dt, dx):
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

    k_nyq = 1.0 / (2.0 * dx)
    f_nyq = 0.5 / dt
    print(f"  temporal Nyquist {f_nyq:.0f} Hz at dt = {dt * 1000:g} ms")
    print(f"  spatial Nyquist {k_nyq:.4f} c/m at dx = {dx:g} m; alias onset:")
    for v in (1500.0, 2000.0):
        onset = v * k_nyq
        note = "  (above temporal Nyquist - cannot alias)" if onset > f_nyq else ""
        print(f"    v = {v:6.0f} m/s -> {onset:6.1f} Hz{note}")


def filter_shot(gather, mask, keep):
    pad = np.zeros((PAD_FRONT, gather.shape[1]), dtype=np.float64)
    padded = np.concatenate([pad, gather], axis=0)
    return f_filtering(padded, mask, is_zeroout=False)[keep]


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
    axes[1].set_title(f"mean trace spectrum, {DOMAIN}, "
                      f"{len(SHOTS)} shots averaged")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return fig


def gather_figure(panels):
    """Rows of before / after / difference, one row per shot."""
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
    fig.suptitle(f"{DOMAIN}: low-cut {low_cut}, low-pass {low_pass}"
                 f"  (panels clipped at {CLIP_PERC:g}% of 'before')")
    fig.tight_layout()
    return fig


def main():
    if DOMAIN not in SOURCES:
        raise SystemExit(f"DOMAIN must be one of {sorted(SOURCES)}, got {DOMAIN}")
    in_rel, dx = SOURCES[DOMAIN]
    in_path = resolve(in_rel)
    if not os.path.isfile(in_path):
        raise SystemExit(f"not found: {in_path}")

    data = np.load(in_path, mmap_mode="r")
    if data.ndim != 3:
        raise SystemExit(f"expected (shots, samples, traces), got {data.shape}")
    n_shots, nt, n_traces = data.shape
    dt = DT_US * 1e-6

    bad = [s for s in SHOTS if not 0 <= s < n_shots]
    if bad:
        raise SystemExit(f"shots out of range 0..{n_shots - 1}: {bad}")

    nt_pad = nt + PAD_FRONT
    keep = slice(PAD_FRONT, nt_pad)
    tr = slice(None) if TRACES is None else TRACES
    ns = nt if SAMPLES is None else min(SAMPLES, nt)

    low_cut = (f"{HP_F_CUT:g} Hz (order {HP_ORDER:g}, decay {HP_DECAY:g})"
               if HIGHPASS else "off")
    low_pass = (f"{LP_F_CUT:g} Hz (order {LP_ORDER:g}, decay {LP_DECAY:g})"
                if LOWPASS else "off")

    print(f"{in_rel}\n  {n_shots} shots x {nt} samples @ {DT_US / 1000:g} ms "
          f"x {n_traces} traces @ {dx:g} m")
    print(f"  pad {PAD_FRONT} in front -> filter on {nt_pad} -> drop the pad")
    print(f"  low-cut  {low_cut}")
    print(f"  low-pass {low_pass}")
    print(f"  zero DC  {ZERO_DC}")

    mask = build_mask(nt_pad, dt)
    if REPORT_RESPONSE:
        report(mask, nt_pad, dt, dx)

    fr = np.fft.rfftfreq(nt, dt)
    acc_before = np.zeros(len(fr))
    acc_after = np.zeros(len(fr))
    panels = []

    print(f"\n  filtering {len(SHOTS)} shots")
    for shot in SHOTS:
        before = np.asarray(data[shot], dtype=np.float64)
        after = filter_shot(before, mask, keep)

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

    figs = [("response", response_figure(mask, nt_pad, dt, spectra)),
            ("gathers", gather_figure(panels))]

    if SAVE_DIR is not None:
        out_dir = resolve(SAVE_DIR)
        os.makedirs(out_dir, exist_ok=True)
        for name, fig in figs:
            p = os.path.join(out_dir, f"freq_filter_preview_{DOMAIN}_{name}.png")
            fig.savefig(p, dpi=130)
            print(f"  wrote {p}")

    if SHOW:
        plt.show()
    else:
        for _, fig in figs:
            plt.close(fig)


if __name__ == "__main__":
    main()
