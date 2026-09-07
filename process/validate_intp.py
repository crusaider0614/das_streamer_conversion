"""Is the trace interpolation any good?  Two tests, neither of them cosmetic.

The periodic stripe at the recorded positions says the reconstruction differs
from the recording, but not by how much or whether what it puts in the gaps is
right.  These are the two measurements that answer that.

Test A - hold out recorded traces and score against them
--------------------------------------------------------
Drop every other recorded trace, rebuild it from the ones that are left, and
compare against what was actually recorded.  This is the only test that yields
an SNR, because it is the only one with ground truth.

Run at two decimations:

    2x   12 traces at 6.24 m -> 23 at 3.12 m, scored on the 11 held out
    4x    6 traces at 12.5 m -> 21 at 3.12 m, scored on the 15 held out

Matching the bandwidth is not optional
--------------------------------------
process/band_limit.py cuts at 220 Hz because that is what keeps the 3.12 m
recorded array unaliased: the 1500 m/s water arrival folds over above
v / (2 dx) = 240 Hz.  Decimating to 6.24 m for a hold-out moves that onset to
120 Hz, and to 60 Hz at 12.5 m.  Feeding the solver 220 Hz of data on those
grids breaks the very condition the method needs, and it will fail for that
reason alone - which says nothing about how it does on the array we actually
have.

So each hold-out is low-passed to BASE_LP_HZ / factor first, and the reference
is filtered the same way.  The test then sits at the same point relative to its
own alias onset as the real job does to its own: a scale model rather than a
harder problem.  Set MATCH_BANDWIDTH = False to see what the mismatch costs.

Test B - f-k leakage, measured on the real output
-------------------------------------------------
No ground truth needed, and it runs on the production array rather than a
degraded copy.  The recorded data are not spatially aliased, so the true
wavefield has no energy beyond the *input* spatial Nyquist, 0.160 cycles/m at
the 3.12 m group interval.  Interpolating to 0.78 m raises the Nyquist by 4 but
cannot create real energy above the old one.  Whatever sits between the two is
an artefact, and its share of the total is a direct quality number for the
array as it actually is.

Edit the settings block below, then run from the repository root:

    python -m process.validate_intp
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np

import config as C
import process.interpolate_traces as I
import process.intp_sweep as S
from utils.data import get_project_root
from utils.process import f_filter, f_filtering

# ---------------------------------------------------------------- settings --

# Test A reads the recorded gathers; Test B reads the finished interpolation.
INPUT_NPY = C.ARRAYS["str"]["bl"]
INTP_NPY = C.ARRAYS["str"]["intp"]

# Shots to test.  Each 2x hold-out costs one cascade stage, each 4x costs two,
# so a shot is roughly a minute for both.
SHOTS = (0, 275, 550)

# Hold-out decimations to run.  2 is the honest one; 4 matches intp_sweep's
# "validate" mode and is included for comparison.
FACTORS = (2, 4)

# Low-pass each hold-out to BASE_LP_HZ / factor, so its input sits the same
# distance from its own alias onset as the real job does from 240 Hz.  Without
# this the test measures aliasing, not interpolation.
MATCH_BANDWIDTH = True
BASE_LP_HZ = 220.0
LP_ORDER = 0.5
LP_DECAY = 0.5
DT_US = 500

# Run Test B.  Reads SHOTS from the finished array; no solving, so it is quick.
FK_TEST = True

# Trace spacing of the recorded array and of the interpolated one.
DX_IN_M = 3.1196
DX_OUT_M = C.STR_TRACE_INTERVAL_M

CLIP_PERC = 99.0

# Show interactively, and optionally save.  A falsy SAVE_DIR skips saving.
SHOW = True
SAVE_DIR = None

# ---------------------------------------------------------------------------


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(get_project_root(), path)


def band_limit(gather, f_cut):
    """Low-pass along time, padded so the circular filter does not wrap."""
    nt, nx = gather.shape
    pad = np.zeros((nt // 4, nx))
    y = np.concatenate([pad, gather, pad], axis=0)
    mask = f_filter(y.shape[0], DT_US * 1e-6, f_cut, LP_ORDER, LP_DECAY,
                    is_lowpass=True)
    return f_filtering(y, mask)[pad.shape[0]:pad.shape[0] + nt]


def holdout(gather, factor):
    """Reconstruct `gather` from every `factor`-th trace; return (recon, truth).

    The cascade reaches from the first kept trace to the last, so both arrays
    are trimmed to that span and line up column for column.
    """
    if MATCH_BANDWIDTH:
        gather = band_limit(gather, BASE_LP_HZ / factor)
    keep = np.arange(0, gather.shape[1], factor)
    S.STAGE_FACTOR = 2
    S.N_STAGES = int(round(np.log2(factor)))
    if 2 ** S.N_STAGES != factor:
        raise SystemExit(f"factor {factor} is not a power of the stage factor 2")
    recon, _ = S.cascade(gather[:, keep], I.THRESHOLD, I.ALIAS_SCALES)
    truth = gather[:, :keep[-1] + 1]
    return recon[:, :truth.shape[1]], truth, keep[keep < truth.shape[1]]


def score(recon, truth, kept_cols):
    """SNR over the held-out traces, and over the kept ones for reference."""
    held = np.setdiff1d(np.arange(truth.shape[1]), kept_cols)
    out = {
        "held_snr": S.snr_db(truth[:, held], recon[:, held] - truth[:, held]),
        "kept_snr": S.snr_db(truth[:, kept_cols],
                             recon[:, kept_cols] - truth[:, kept_cols]),
        "n_held": len(held),
        "held": held,
    }
    per = np.array([S.snr_db(truth[:, j], recon[:, j] - truth[:, j])
                    for j in held])
    out["per_trace"] = per
    a, b = truth[:, held].ravel(), recon[:, held].ravel()
    out["corr"] = float(np.corrcoef(a, b)[0, 1])
    out["ampl"] = float(np.sqrt(np.mean(b ** 2) / np.mean(a ** 2)))
    return out


def fk_leakage(gather, dx_out, dx_in):
    """Share of the f-k amplitude sitting above the input spatial Nyquist."""
    S.DT_MS = 0.5
    f, k, amp = S.fk_amplitude(gather, dx_out, 1)
    k_nyq_in = 1.0 / (2.0 * dx_in)
    above = np.abs(k) > k_nyq_in
    return float(amp[:, above].sum() / max(amp.sum(), 1e-30)), f, k, amp, k_nyq_in


def holdout_figure(results):
    n = len(results)
    fig, axes = plt.subplots(n, 4, figsize=(18, 4.6 * n), squeeze=False)
    for row, (shot, factor, recon, truth, sc) in enumerate(results):
        err = recon - truth
        clip = np.percentile(np.abs(truth), CLIP_PERC)
        for col, (img, name) in enumerate(((truth, "recorded"),
                                           (recon, "reconstructed"),
                                           (err, "error"))):
            ax = axes[row][col]
            ax.imshow(img, cmap="gray", vmin=-clip, vmax=clip, aspect="auto",
                      interpolation="nearest")
            ax.set_title(f"shot {shot} {factor}x: {name}", fontsize=9)
            ax.set_xlabel("trace")
            if col == 0:
                ax.set_ylabel("sample")
        ax = axes[row][3]
        ax.plot(sc["held"], sc["per_trace"], marker="o", ms=3, lw=0.8)
        ax.axhline(sc["held_snr"], color="r", lw=1,
                   label=f"overall {sc['held_snr']:.1f} dB")
        ax.set_xlabel("held-out trace")
        ax.set_ylabel("SNR [dB]")
        ax.set_title(f"shot {shot} {factor}x: per-trace SNR", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("hold-out test: rebuild recorded traces and score against them")
    fig.tight_layout()
    return fig


def fk_figure(panels):
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5),
                             squeeze=False)
    for col, (shot, f, k, amp, k_nyq_in, frac) in enumerate(panels):
        ax = axes[0][col]
        db = 20 * np.log10(amp / amp.max() + 1e-12)
        ax.imshow(db, aspect="auto", cmap="magma", vmin=-80, vmax=0,
                  extent=[k[0], k[-1], f[0], f[-1]], origin="lower")
        for s in (-1, 1):
            ax.axvline(s * k_nyq_in, color="c", ls="--", lw=1)
        ax.set_ylim(0, 250)
        ax.set_xlabel("wavenumber [cycles/m]")
        ax.set_ylabel("frequency [Hz]")
        ax.set_title(f"shot {shot}: {100 * frac:.1f} % beyond the input Nyquist",
                     fontsize=9)
    fig.suptitle("f-k of the interpolated array; dashed = input spatial Nyquist, "
                 "anything outside is an artefact")
    fig.tight_layout()
    return fig


def main():
    I.configure()
    in_path = resolve(INPUT_NPY)
    if not os.path.isfile(in_path):
        raise SystemExit(f"not found: {in_path}")
    data = np.load(in_path, mmap_mode="r")
    print(f"{INPUT_NPY}  {data.shape}")
    print(f"  threshold {I.THRESHOLD}, alias scales {I.ALIAS_SCALES}, "
          f"iterations {I.N_ITER}, replace observed {I.REPLACE_OBSERVED}")
    if MATCH_BANDWIDTH:
        print(f"  bandwidth matched: each {'/'.join(str(f) for f in FACTORS)}x "
              f"hold-out low-passed to {BASE_LP_HZ:g}/factor Hz "
              + ", ".join(f"{f}x -> {BASE_LP_HZ / f:.0f} Hz" for f in FACTORS))
    else:
        print(f"  bandwidth NOT matched - the hold-outs are aliased and the "
              f"numbers mean little")

    figs = []
    results = []
    print(f"\nTest A - hold out and score")
    print(f"{'shot':>6}{'factor':>8}{'held':>6}{'held SNR':>11}"
          f"{'kept SNR':>11}{'corr':>8}{'ampl':>8}{'s':>7}")
    for shot in SHOTS:
        g = np.asarray(data[shot], dtype=np.float64)
        for factor in FACTORS:
            t0 = time.time()
            recon, truth, kept = holdout(g, factor)
            sc = score(recon, truth, kept)
            print(f"{shot:>6}{factor:>7}x{sc['n_held']:>6}"
                  f"{sc['held_snr']:>10.2f}dB{sc['kept_snr']:>10.2f}dB"
                  f"{sc['corr']:>8.3f}{sc['ampl']:>8.3f}"
                  f"{time.time() - t0:>7.0f}")
            results.append((shot, factor, recon, truth, sc))
    figs.append(("holdout", holdout_figure(results)))

    for factor in FACTORS:
        sel = [r[4]["held_snr"] for r in results if r[1] == factor]
        amp = [r[4]["ampl"] for r in results if r[1] == factor]
        print(f"  {factor}x mean over {len(sel)} shots: "
              f"{np.mean(sel):.2f} dB, amplitude ratio {np.mean(amp):.3f}")

    if FK_TEST:
        intp_path = resolve(INTP_NPY)
        if not os.path.isfile(intp_path):
            print(f"\nTest B skipped - not found: {INTP_NPY}")
        else:
            intp = np.load(intp_path, mmap_mode="r")
            print(f"\nTest B - f-k leakage on {INTP_NPY}  {intp.shape}")
            print(f"  input spatial Nyquist {1 / (2 * DX_IN_M):.4f} c/m "
                  f"at {DX_IN_M:g} m; output grid {DX_OUT_M:g} m")
            panels = []
            for shot in SHOTS:
                g = np.asarray(intp[shot], dtype=np.float64)
                frac, f, k, amp, k_nyq = fk_leakage(g, DX_OUT_M, DX_IN_M)
                print(f"    shot {shot:>4}: {100 * frac:6.2f} % of the f-k "
                      f"amplitude lies beyond it")
                panels.append((shot, f, k, amp, k_nyq, frac))
            figs.append(("fk", fk_figure(panels)))

    if SAVE_DIR:
        out_dir = resolve(SAVE_DIR)
        os.makedirs(out_dir, exist_ok=True)
        for name, fig in figs:
            p = os.path.join(out_dir, f"validate_intp_{name}.png")
            fig.savefig(p, dpi=130)
            print(f"  wrote {p}")

    if SHOW:
        plt.show()
    else:
        for _, fig in figs:
            plt.close(fig)


if __name__ == "__main__":
    main()
