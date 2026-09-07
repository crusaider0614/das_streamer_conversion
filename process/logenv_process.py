"""RMS normalisation and log-envelope scaling: the last stage before the dataset.

Takes the band-passed arrays and writes the pair `module/dataset_pohang_shore.py`
loads:

    numpy/das_data_freq.npy  (551, 3500, 1058)  ->  das_data_log.npy
    numpy/str_data_freq.npy  (551, 3500,   93)  ->  str_data_log.npy

Two steps, in this order:

  1. RMS normalise.  One scalar per domain, `TARGET_RMS / rms(input)`, applied
     to every shot.  Relative amplitudes between shots and between traces are
     untouched - only the unit changes.
  2. Log-envelope gain.  `utils.process.calculate_logscale` builds a smoothly
     varying scale from the trace envelope and the data is multiplied by it, so
     a gather shows the late, weak arrivals alongside the early, strong ones
     instead of only the direct wave.  Fitted on a tail-padded gather - the
     envelope is circular, see PAD_TAIL.

Both follow `process_data.py` in the 4d-noise-attenuation project, which
normalises to a common RMS (NORMALIZED_RMS) before fitting the envelope gain
and saves the gain and its parameters alongside the data.

Why the normalisation has to come first
---------------------------------------
`calculate_logscale` is not scale invariant.  Its `log_base` is added to the
envelope in the envelope's own units, so it only means something relative to
the amplitudes it meets - the same number applied to two differently scaled
arrays compresses them by different amounts.

That matters here more than usual, because the two domains arrive nowhere near
each other: the DAS array has an RMS around 1.6e+03 and the streamer around
1.1e-01, four orders of magnitude apart, purely from how each instrument
reports amplitude.  Normalising both to TARGET_RMS first puts `log_base` on a
common footing, which for an unpaired translation matters directly: if one
domain comes out visibly more compressed than the other, that difference is a
cue the network can learn instead of the physics.

It does not make the two identical.  The envelope median of an RMS-1 gather is
about 0.21 for the DAS and 0.009 for the streamer - the streamer gather is the
spikier of the two, most of its energy in a few strong arrivals - so LOG_BASE
stays per-domain.  The run reports base/median for each; keep those two ratios
close and the two domains are being compressed alike.

SCALE_MODE
----------
"per_shot" fits a separate gain for every shot.  The shot moves along the line
here, so the moveout apex moves with it (trace ~470 at shot 0, ~380 at shot
550); one gain map averaged over shots would smear across that.  This is the
default and it matches what this project did before.

"average" fits a single gain from the mean gather and applies it to all shots,
the way 4d-noise-attenuation does - correct there because its geometry is fixed.
It preserves relative amplitude between shots, which "per_shot" does not.

Parameters and measurements are written next to the output as
`<name>_log_meta.json` (settings, and the numbers the run measured) and
`<name>_log_meta.npz` (per-shot arrays, and the shared gain map in "average"
mode).  Nothing downstream reads them; they are there so a saved array can be
traced back to what produced it.

Edit the settings block below, then run from the repository root:

    python -m process.logenv_process
"""

import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np

import config as C
from utils.data import create_memmap, get_project_root, npy_shape
from utils.process import calculate_logscale, envelope_1d

# ---------------------------------------------------------------- settings --

# Domains to process, in order.  The input, the output and every parameter
# come from config - see config.ARRAYS, config.TARGET_RMS and
# config.LOG_SCALE_PARAMS.  Change the values there, not here.
#
# "str" is left out while str_data_freq.npy does not exist - a missing input
# aborts the run, so listing it would kill the das preview on the way past.
# Put it back once the streamer band-pass has been written, or set
# INPUT_STAGE = "geom" to read both sides pre-filter.
DOMAINS = ("str",)

# Which stage to read.  "freq" is the band-passed array; use "geom" for a
# domain whose band-pass has not been run.
INPUT_STAGE = "norm"

# Draw a few shots - raw, RMS-normalised, log-scaled, and the gain - and show
# them.  Costs one extra read of those shots and writes nothing.
PREVIEW = True
PREVIEW_SHOTS = (0, 275, 550)
SAMPLES = None
CLIP_PERC = 99.0

# Filter every shot and write the outputs.
WRITE_ALL = False

# Show the preview interactively, and optionally save it.  A falsy SAVE_DIR
# skips saving.
SHOW = True
SAVE_DIR = None

# Write the sidecar parameter/measurement files.  Independent of WRITE_ALL -
# a preview that settled on a norm_scale and a log_base is worth recording on
# its own - so it is off while nothing is being kept.
WRITE_META = False

# Zeros appended to the end of each gather before the gain is fitted, and
# dropped afterwards.  0 disables it.
#
# calculate_logscale takes the envelope with a Hilbert transform built from an
# FFT along time, so it is circular: the end of the record and its start are
# neighbours, and the step between them - the last live samples against the
# first - spreads a false envelope across both.  The same reason
# process/freq_filter.py pads, except that one pads the front because it also
# trims the tail; here nothing is trimmed, so the pad goes on the end where
# the discontinuity is.
#
# 2000 samples is 2 s on the 1 ms grid, comfortably longer than the envelope
# smoothing reaches.  The gain in the pad region is discarded, so the only
# cost is the extra FFT length.
PAD_TAIL = 2000

# ----------------------------------------------------------------- derived --

TARGET_RMS = C.TARGET_RMS
RMS_SAMPLE_SHOTS = C.RMS_SAMPLE_SHOTS
SCALE_MODE = C.SCALE_MODE

# (tag, input, output) per domain.  Inputs sit in numpy/ with the rest of the
# working arrays; outputs go to the pohang_shore root, where the dataset looks
# for them.
JOBS = tuple((tag, C.ARRAYS[tag][INPUT_STAGE], C.ARRAYS[tag]["log"])
             for tag in DOMAINS)

# ---------------------------------------------------------------------------


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(get_project_root(), path)


def sample_indices(n_shots):
    if RMS_SAMPLE_SHOTS is None or RMS_SAMPLE_SHOTS >= n_shots:
        return np.arange(n_shots)
    return np.unique(np.linspace(0, n_shots - 1, RMS_SAMPLE_SHOTS).astype(int))


def measure_rms(data, idx):
    """Overall RMS over the sampled shots, plus the per-shot values.

    Accumulated as a sum of squares rather than a mean of means so the result
    is the true RMS of the sampled block even though it is read shot by shot.
    """
    total, count = 0.0, 0
    per_shot = np.zeros(len(idx))
    for k, i in enumerate(idx):
        g = np.asarray(data[i], dtype=np.float64)
        ss = float(np.sum(g ** 2))
        total += ss
        count += g.size
        per_shot[k] = np.sqrt(ss / g.size)
    return float(np.sqrt(total / count)), per_shot


def average_gather(data, idx):
    """Mean gather over the sampled shots, for SCALE_MODE == "average"."""
    acc = np.zeros(data.shape[1:], dtype=np.float64)
    for i in idx:
        acc += np.asarray(data[i], dtype=np.float64)
    return acc / len(idx)


def logscale(gather, lsp):
    """calculate_logscale on a tail-padded gather, with the pad dropped.

    See PAD_TAIL: the envelope is circular, so the record has to be separated
    from its own start before the gain is fitted.

    The pad also becomes the array's minimum envelope, and calculate_logscale
    subtracts that minimum from the log envelope.  With zeros there the
    minimum is exactly log10(log_base), against log10(min_envelope + log_base)
    without the pad.  Measured on shot 275 of each domain that moves the
    minimum by 4.7e-05 in log10 units on the DAS and 4.2e-06 on the streamer -
    log_base dominates the smallest envelopes either way, so the gain map is
    unchanged for any practical purpose.
    """
    if not PAD_TAIL:
        return calculate_logscale(gather, **lsp)
    nt = gather.shape[0]
    pad = np.zeros((PAD_TAIL, gather.shape[1]), dtype=gather.dtype)
    padded = np.concatenate([gather, pad], axis=0)
    return calculate_logscale(padded, **lsp)[:nt]


def stage_figure(tag, shot, nor, log, scale, shared_scale, lsp):
    """One shot: the input, the log-scaled result, and the gain between them.

    No separate panel for the un-normalised input.  The RMS step is a single
    scalar and every panel is percentile-clipped on its own, so it would come
    out pixel-identical to the normalised one; norm_scale is on the console.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    cols = ((nor, f"input, RMS -> {TARGET_RMS:g}", "seismic"),
            (log, "log envelope", "seismic"),
            (scale, "gain" + (" (shared)" if shared_scale else ""), "viridis"))
    for col, (img, name, cmap) in enumerate(cols):
        ax = axes[col]
        if cmap == "seismic":
            clip = np.percentile(np.abs(img), CLIP_PERC)
            clip = clip if clip > 0 else 1.0
            ax.imshow(img, cmap=cmap, vmin=-clip, vmax=clip, aspect="auto",
                      interpolation="nearest")
        else:
            ax.imshow(img, cmap=cmap, aspect="auto", interpolation="nearest")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("trace")
    axes[0].set_ylabel("sample")
    fig.suptitle(f"{tag} shot {shot}  -  base {lsp['log_base']:g}, "
                 f"sigma {lsp['smooth_sigma']}, {SCALE_MODE}")
    fig.tight_layout()
    return fig


def preview(tag, data, norm_scale, lsp, shared_scale):
    """One figure per shot, drawn and shown before the next is built.

    plt.show() blocks until the window is closed, so the shots arrive one at a
    time rather than all at once - and only one gather is held in memory.
    """
    n_shots = data.shape[0]
    bad = [s for s in PREVIEW_SHOTS if not 0 <= s < n_shots]
    if bad:
        raise SystemExit(f"preview shots out of range 0..{n_shots - 1}: {bad}")

    ns = data.shape[1] if SAMPLES is None else min(SAMPLES, data.shape[1])
    out_dir = None
    if SAVE_DIR:
        out_dir = resolve(SAVE_DIR)
        os.makedirs(out_dir, exist_ok=True)

    print(f"    preview: {len(PREVIEW_SHOTS)} shots, one window at a time")
    for n, shot in enumerate(PREVIEW_SHOTS, 1):
        nor = np.asarray(data[shot], dtype=np.float32) * norm_scale
        scale = shared_scale if shared_scale is not None else logscale(nor, lsp)
        log = nor * scale
        print(f"      [{n}/{len(PREVIEW_SHOTS)}] shot {shot:>4}: "
              f"rms {np.sqrt(np.mean(nor ** 2)):.4g} "
              f"-> {np.sqrt(np.mean(log ** 2)):.4g}, "
              f"|log| max {np.abs(log).max():.4g}, "
              f"gain {scale.min():.3g}..{scale.max():.3g}")

        fig = stage_figure(tag, shot, nor[:ns], log[:ns], scale[:ns],
                           shared_scale is not None, lsp)
        if out_dir:
            p = os.path.join(out_dir, f"logenv_process_{tag}_shot{shot:04d}.png")
            fig.savefig(p, dpi=130)
            print(f"        wrote {p}")
        if SHOW:
            plt.show()
        plt.close(fig)


def write_all(data, out_rel, norm_scale, lsp, shared_scale):
    """Stream every shot through the two steps into out_rel."""
    out_path = resolve(out_rel)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if os.path.isfile(out_path):
        prev = npy_shape(out_path)
        print(f"    overwriting {out_rel} (was {prev})")
        del prev

    n_shots = data.shape[0]
    out = create_memmap(out_path, data.shape)
    print(f"    -> {out_rel}  {out.nbytes / 1e9:.2f} GB")

    rms_out = np.zeros(n_shots)
    lo, hi = np.inf, -np.inf
    t0 = time.time()
    for i in range(n_shots):
        nor = np.asarray(data[i], dtype=np.float32) * norm_scale
        scale = shared_scale if shared_scale is not None else logscale(nor, lsp)
        log = nor * scale
        out[i] = log
        rms_out[i] = np.sqrt(np.mean(log.astype(np.float64) ** 2))
        lo = min(lo, float(log.min()))
        hi = max(hi, float(log.max()))
        if (i + 1) % 50 == 0 or i + 1 == n_shots:
            el = time.time() - t0
            print(f"      {i + 1}/{n_shots} shots  {el:.0f}s elapsed, "
                  f"eta {el / (i + 1) * (n_shots - i - 1):.0f}s")
    out.flush()
    print(f"    amplitude range [{lo:.4g}, {hi:.4g}], "
          f"rms {rms_out.mean():.4g}")
    return dict(amplitude_min=lo, amplitude_max=hi,
                value_range_max_abs=max(abs(lo), abs(hi))), rms_out


def write_meta(out_rel, tag, params, info, arrays):
    stem = resolve(os.path.splitext(out_rel)[0])
    with open(stem + "_meta.json", "w", encoding="utf-8") as f:
        json.dump({"tag": tag, "parameters": params, "measured": info}, f,
                  indent=2, sort_keys=True)
    np.savez(stem + "_meta.npz", **arrays)
    print(f"    wrote {os.path.basename(stem)}_meta.json / _meta.npz")


def main():
    if SCALE_MODE not in ("per_shot", "average"):
        raise SystemExit(f"SCALE_MODE must be 'per_shot' or 'average', "
                         f"got {SCALE_MODE}")

    for tag, in_rel, out_rel in JOBS:
        in_path = resolve(in_rel)
        if not os.path.isfile(in_path):
            raise SystemExit(f"not found: {in_path}")

        data = np.load(in_path, mmap_mode="r")
        n_shots, n_samp, n_trace = data.shape
        lsp = C.LOG_SCALE_PARAMS[tag]
        idx = sample_indices(n_shots)
        print(f"{tag}: {in_rel}  {data.shape}")

        # --- step 1: RMS normalisation
        rms_in, rms_in_per_shot = measure_rms(data, idx)
        norm_scale = TARGET_RMS / rms_in
        print(f"    rms {rms_in:.6g} over {len(idx)} shots "
              f"-> {TARGET_RMS:g}  (scale {norm_scale:.6g})")

        # --- step 2: log-envelope gain
        shared_scale = None
        if SCALE_MODE == "average":
            shared_scale = logscale(average_gather(data, idx) * norm_scale, lsp)
            print(f"    shared gain from the mean of {len(idx)} shots, "
                  f"range [{shared_scale.min():.4g}, {shared_scale.max():.4g}]")

        probe = np.asarray(data[idx[0]], dtype=np.float32) * norm_scale
        env_median = float(np.median(envelope_1d(probe)))
        log_base = lsp["log_base"]
        print(f"    log base {log_base:g} vs median envelope {env_median:.4g} "
              f"-> base/median {log_base / env_median:.3g}")

        if PREVIEW:
            preview(tag, data, norm_scale, lsp, shared_scale)

        # Everything that goes in the sidecar except the output-side numbers is
        # known by now, so the record is written whether or not WRITE_ALL runs:
        # a preview that settled on a norm_scale and a log_base is exactly the
        # thing worth keeping, and losing it because nothing was written would
        # defeat the point.
        params = dict(
            input=in_rel.replace("\\", "/"),
            output=out_rel.replace("\\", "/"),
            shape=list(data.shape),
            target_rms=TARGET_RMS,
            rms_sample_shots=RMS_SAMPLE_SHOTS,
            scale_mode=SCALE_MODE,
            pad_tail=PAD_TAIL,
            **{k: (list(v) if isinstance(v, tuple) else v)
               for k, v in lsp.items()},
        )
        info = dict(
            rms_input=rms_in,
            norm_scale=norm_scale,
            rms_shots_measured=len(idx),
            envelope_median_normalised=env_median,
            base_over_envelope_median=log_base / env_median,
            output_written=bool(WRITE_ALL),
        )
        arrays = dict(rms_input_per_shot=rms_in_per_shot,
                      rms_input_shot_index=idx)
        if shared_scale is not None:
            arrays["shared_scale"] = shared_scale

        if WRITE_ALL:
            out_info, rms_out_per_shot = write_all(data, out_rel, norm_scale,
                                                   lsp, shared_scale)
            info.update(
                rms_output_mean=float(rms_out_per_shot.mean()),
                rms_output_min=float(rms_out_per_shot.min()),
                rms_output_max=float(rms_out_per_shot.max()),
                **out_info,
            )
            arrays["rms_output_per_shot"] = rms_out_per_shot
        else:
            print(f"    WRITE_ALL is off - {out_rel} not touched")

        if WRITE_META:
            write_meta(out_rel, tag, params, info, arrays)

        print()


if __name__ == "__main__":
    main()
