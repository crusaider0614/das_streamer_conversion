"""RMS normalisation only - the last stage before the dataset.

    numpy/das_data_freq.npy  (551, 3500, 1058)  ->  das_data_norm.npy
    numpy/str_data_geom.npy  (551, 3500,   93)  ->  str_data_norm.npy

One scalar per domain, `TARGET_RMS / rms(input)`, applied to every shot.
Relative amplitudes between shots, traces and samples are untouched; only the
unit changes.  That is the whole stage.

No envelope gain.  process/logenv_process.py is the version that also applies
calculate_logscale; this one exists because the two domains have to be brought
onto a common amplitude scale for an unpaired translation - the DAS array has
an RMS around 1.6e+03 and the streamer around 1.1e-01, four orders of magnitude
apart, purely from how each instrument reports amplitude - and that is a
separate question from whether their dynamic range should be compressed.

Where the streamer input comes from
-----------------------------------
str_data_geom.npy, not str_data_freq.npy: whichever route produced `geom` -
the curvelet cascade or process/interpolate_kernel.py - has already applied its
low-pass on the 0.5 ms grid, where it doubles as the anti-alias filter for the
decimation to 1 ms.  A separate `freq` stage on the streamer side would be a
second pass with nothing left to remove, squaring the response near the corner
for no gain.  The DAS side does have one, since its low-cut and low-pass are
applied by process/freq_filter.py.

Parameters and measurements go next to each output as `<name>_meta.json` and
`<name>_meta.npz`, written whether or not the arrays are.

Edit the settings block below, then run from the repository root:

    python -m process.rms_normalize
"""

import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np

import config as C
import utils.mute as MU
from utils.data import create_memmap, get_project_root, npy_shape

# ---------------------------------------------------------------- settings --

# (tag, input stage, output).  Both sides come off the `line` route now: the
# DAS through process/decimate_receiver.py, which brought its receiver axis to
# the streamer's 3.00 m, and the streamer straight from `line`.  Both are
# already band-passed to 20-300 Hz and already 2000 samples at 1 ms.
JOBS = (
    ("das", "deci", C.ARRAYS["das"]["norm"]),
    ("str", "line", C.ARRAYS["str"]["norm"]),
)

# Mute ahead of the direct arrival before measuring and applying the RMS.
#
# Here rather than earlier, and in this order, for two reasons.  Neither
# `line` nor `deci` mutes - the mute edge is a step along the receiver axis,
# which is not band-limited, so a filter with reach along that axis rings
# across it and drags the direct arrival into the muted window (measured:
# 5.36 % mean, 48.3 % p99 for the k-sinc).  Muting after both filters are
# done avoids that entirely.
#
# And the mute comes before the RMS because it removes 28-37 % of the record;
# a scale measured with that energy still in it would not put the two domains
# on the common amplitude scale this stage exists for.
#
# The boundary is read from each domain's sidecar - config.META["<tag>_<stage>"]
# - so what is applied is the boundary the geometry stage computed, on the
# projected source positions the samples were rolled onto.
APPLY_MUTE = True

# Draw a few shots and show them.  Writes nothing.
PREVIEW = True
PREVIEW_SHOTS = (0, 275, 550)
SAMPLES = None
CLIP_PERC = 99.0

# Normalise every shot and write the outputs.
WRITE_ALL = True

# Show the preview interactively, and optionally save it.
SHOW = False
SAVE_DIR = C.DATA_DIR

# Sidecar parameter/measurement files.
WRITE_META = True

# Samples kept, from the start of the record; None keeps all of them.  See
# config.NORM_SAMPLES for why 2000.
OUT_SAMPLES = C.NORM_SAMPLES

# ----------------------------------------------------------------- derived --

TARGET_RMS = C.TARGET_RMS
RMS_SAMPLE_SHOTS = C.RMS_SAMPLE_SHOTS

# ---------------------------------------------------------------------------


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(get_project_root(), path)


def sample_indices(n_shots):
    if RMS_SAMPLE_SHOTS is None or RMS_SAMPLE_SHOTS >= n_shots:
        return np.arange(n_shots)
    return np.unique(np.linspace(0, n_shots - 1, RMS_SAMPLE_SHOTS).astype(int))


def crop(gather):
    """The samples the output keeps."""
    return gather if OUT_SAMPLES is None else gather[:OUT_SAMPLES]


def mute_boundary(tag, stage, n_shots, n_traces):
    """The mute boundary this input was built with, from its sidecar.

    Read rather than recomputed: the samples were rolled onto projected source
    positions, so the only boundary that matches them is the one the geometry
    stage wrote.  Returns None when muting is off.
    """
    if not APPLY_MUTE:
        return None
    key = f"{tag}_{stage}"
    if key not in C.META:
        raise SystemExit(f"APPLY_MUTE is on but config.META has no {key!r}")
    p = resolve(C.META[key])
    if not os.path.isfile(p):
        raise SystemExit(f"not found: {p}")
    s = np.load(p)
    t = s["mute_boundary_ms"]
    if t.shape != (n_shots, n_traces):
        raise SystemExit(f"{key} boundary is {t.shape}, the array is "
                         f"({n_shots}, {n_traces})")
    if "mute_applied" in s and bool(s["mute_applied"]):
        raise SystemExit(f"{key} says its samples are ALREADY muted - muting "
                         f"again would square the taper; set APPLY_MUTE off "
                         f"or rebuild that stage unmuted")
    return t


def apply_mute(g, t_row, dt_ms):
    """One shot, muted on the cropped grid.  A no-op when t_row is None."""
    if t_row is None:
        return g
    return g * MU.weights(t_row, g.shape[0], dt_ms).astype(g.dtype)


def mute_report(t, nt, dt_ms):
    span = nt * dt_ms
    frac = 100 * np.mean(np.clip(t, 0, span)) / span
    past = 100 * (t >= span).mean()
    return (f"mute {C.MUTE_PARAMS['velocity_m_s']:g} m/s, lead "
            f"{C.MUTE_PARAMS['lead_ms']:g} ms, taper "
            f"{C.MUTE_PARAMS['taper_ms']:g} ms: boundary {t.min():.0f}.."
            f"{t.max():.0f} ms, {frac:.1f} % of the record on average"
            + (f", {past:.1f} % of traces entirely" if past else ""))


def measure_rms(data, idx, t_mute, dt_ms):
    """Overall RMS over the sampled shots, plus the per-shot values.

    A sum of squares rather than a mean of means, so the result is the true RMS
    of the sampled block even though it is read shot by shot.
    """
    total, count = 0.0, 0
    per_shot = np.zeros(len(idx))
    for k, i in enumerate(idx):
        g = apply_mute(crop(np.asarray(data[i], dtype=np.float64)),
                       None if t_mute is None else t_mute[i], dt_ms)
        ss = float(np.sum(g ** 2))
        total += ss
        count += g.size
        per_shot[k] = np.sqrt(ss / g.size)
    return float(np.sqrt(total / count)), per_shot


def preview(tag, data, norm_scale, t_mute, dt_ms):
    n_shots = data.shape[0]
    bad = [s for s in PREVIEW_SHOTS if not 0 <= s < n_shots]
    if bad:
        raise SystemExit(f"preview shots out of range 0..{n_shots - 1}: {bad}")
    ns = data.shape[1] if SAMPLES is None else min(SAMPLES, data.shape[1])

    fig, axes = plt.subplots(1, len(PREVIEW_SHOTS),
                             figsize=(5.2 * len(PREVIEW_SHOTS), 6),
                             squeeze=False)
    for col, shot in enumerate(PREVIEW_SHOTS):
        g = apply_mute(crop(np.asarray(data[shot], dtype=np.float32)),
                       None if t_mute is None else t_mute[shot],
                       dt_ms) * norm_scale
        clip = np.percentile(np.abs(g), CLIP_PERC)
        clip = clip if clip > 0 else 1.0
        ax = axes[0][col]
        ax.imshow(g[:ns], cmap="gray", vmin=-clip, vmax=clip, aspect="auto",
                  interpolation="nearest")
        ax.set_title(f"shot {shot}", fontsize=9)
        ax.set_xlabel("trace")
        if col == 0:
            ax.set_ylabel("sample")
        print(f"      shot {shot:>4}: rms {np.sqrt(np.mean(g.astype(np.float64) ** 2)):.4g}, "
              f"|max| {np.abs(g).max():.4g}")
    fig.suptitle(f"{tag}: muted, then RMS normalised to {TARGET_RMS:g} "
                 f"(panels clipped at {CLIP_PERC:g}%)")
    fig.tight_layout()

    if SAVE_DIR:
        out_dir = resolve(SAVE_DIR)
        os.makedirs(out_dir, exist_ok=True)
        p = os.path.join(out_dir, f"rms_normalize_{tag}.png")
        fig.savefig(p, dpi=130)
        print(f"    wrote {p}")
    if SHOW:
        plt.show()
    else:
        plt.close(fig)


def write_all(data, out_rel, norm_scale, t_mute, dt_ms):
    out_path = resolve(out_rel)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if os.path.isfile(out_path):
        prev = npy_shape(out_path)
        print(f"    overwriting {out_rel} (was {prev})")
        del prev

    n_shots = data.shape[0]
    shape = (n_shots,
             data.shape[1] if OUT_SAMPLES is None else OUT_SAMPLES,
             data.shape[2])
    out = create_memmap(out_path, shape)
    print(f"    -> {out_rel}  {out.nbytes / 1e9:.2f} GB")

    rms_out = np.zeros(n_shots)
    lo, hi = np.inf, -np.inf
    t0 = time.time()
    for i in range(n_shots):
        y = apply_mute(crop(np.asarray(data[i], dtype=np.float32)),
                       None if t_mute is None else t_mute[i],
                       dt_ms) * norm_scale
        out[i] = y
        rms_out[i] = np.sqrt(np.mean(y.astype(np.float64) ** 2))
        lo = min(lo, float(y.min()))
        hi = max(hi, float(y.max()))
        if (i + 1) % 100 == 0 or i + 1 == n_shots:
            el = time.time() - t0
            print(f"      {i + 1}/{n_shots} shots  {el:.0f}s elapsed")
    out.flush()
    # The overall RMS is the quantity TARGET_RMS sets, and it comes out exact.
    # The mean of the per-shot values is always lower - Jensen - and by more
    # when the shots vary more; reported alongside so the two are not confused.
    overall = float(np.sqrt(np.mean(rms_out ** 2)))
    print(f"    amplitude range [{lo:.4g}, {hi:.4g}]")
    print(f"    overall rms {overall:.6f} (target {TARGET_RMS:g}), "
          f"mean of per-shot rms {rms_out.mean():.4f}, "
          f"per-shot {rms_out.min():.4f}..{rms_out.max():.4f}")
    return dict(amplitude_min=lo, amplitude_max=hi,
                value_range_max_abs=max(abs(lo), abs(hi)),
                rms_output_overall=overall), rms_out


def write_meta(out_rel, tag, params, info, arrays):
    stem = resolve(os.path.splitext(out_rel)[0])
    with open(stem + "_meta.json", "w", encoding="utf-8") as f:
        json.dump({"tag": tag, "parameters": params, "measured": info}, f,
                  indent=2, sort_keys=True)
    np.savez(stem + "_meta.npz", **arrays)
    print(f"    wrote {os.path.basename(stem)}_meta.json / _meta.npz")


def main():
    for tag, stage, out_rel in JOBS:
        in_rel = C.ARRAYS[tag][stage]
        in_path = resolve(in_rel)
        if not os.path.isfile(in_path):
            raise SystemExit(f"not found: {in_path}")

        data = np.load(in_path, mmap_mode="r")
        dt_ms = C.STAGE_DT_US[tag][stage] / 1000.0
        idx = sample_indices(data.shape[0])
        kept = data.shape[1] if OUT_SAMPLES is None else OUT_SAMPLES
        print(f"{tag}: {in_rel}  {data.shape} @ {dt_ms:g} ms"
              + ("" if kept == data.shape[1]
                 else f"  ->  keeping the first {kept} samples"))

        t_mute = mute_boundary(tag, stage, data.shape[0], data.shape[2])
        if t_mute is None:
            print(f"    APPLY_MUTE is off - no mute")
        else:
            print(f"    {mute_report(t_mute, kept, dt_ms)}")

        rms_in, rms_in_per_shot = measure_rms(data, idx, t_mute, dt_ms)
        norm_scale = TARGET_RMS / rms_in
        print(f"    rms {rms_in:.6g} over {len(idx)} shots -> {TARGET_RMS:g}  "
              f"(scale {norm_scale:.6g})")
        recorded = C.INPUT_RMS.get(tag)
        if recorded is not None:
            drift = rms_in / recorded - 1.0
            note = "matches" if abs(drift) < 1e-6 else "DRIFTED - update it"
            print(f"    config.INPUT_RMS[{tag!r}] = {recorded:.6g}  "
                  f"({100 * drift:+.4f} %, {note})")

        if PREVIEW:
            preview(tag, data, norm_scale, t_mute, dt_ms)

        params = dict(
            input=in_rel.replace("\\", "/"),
            input_stage=stage,
            muted=bool(APPLY_MUTE),
            mute=(dict(C.MUTE_PARAMS) if APPLY_MUTE else None),
            output=out_rel.replace("\\", "/"),
            shape=[data.shape[0],
                   data.shape[1] if OUT_SAMPLES is None else OUT_SAMPLES,
                   data.shape[2]],
            input_shape=list(data.shape),
            out_samples=OUT_SAMPLES,
            target_rms=TARGET_RMS,
            rms_sample_shots=RMS_SAMPLE_SHOTS,
            envelope_gain="none",
            filter={name: C.FILTER_PARAMS[name]
                    for name in C.NORM_FILTER_CHAIN[tag]},
        )
        info = dict(rms_input=rms_in, norm_scale=norm_scale,
                    rms_shots_measured=len(idx),
                    config_input_rms=C.INPUT_RMS.get(tag),
                    config_norm_scale=C.NORM_SCALE.get(tag),
                    output_written=bool(WRITE_ALL))
        arrays = dict(rms_input_per_shot=rms_in_per_shot,
                      rms_input_shot_index=idx)

        if WRITE_ALL:
            out_info, rms_out = write_all(data, out_rel, norm_scale,
                                          t_mute, dt_ms)
            info.update(rms_output_mean=float(rms_out.mean()),
                        rms_output_min=float(rms_out.min()),
                        rms_output_max=float(rms_out.max()), **out_info)
            arrays["rms_output_per_shot"] = rms_out
        else:
            print(f"    WRITE_ALL is off - {out_rel} not touched")

        if WRITE_META:
            write_meta(out_rel, tag, params, info, arrays)
        print()


if __name__ == "__main__":
    main()
