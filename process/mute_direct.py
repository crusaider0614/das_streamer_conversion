"""Mute everything ahead of the direct arrival, and look at what that removes.

    das  freq  (551, 3500, 1058) @ 1 ms      DAS group centres
    str  freq  (551, 3500,   93) @ 1 ms      interpolated positions
    str  raw   (551, 8000,   24) @ 0.5 ms    recorded nodes

DOMAIN and INPUT_STAGE choose among them.  The sample interval comes from
config.STAGE_DT_US and the receiver positions from the trace count, so the
0.5 ms streamer raw runs without touching anything else.

Nothing can arrive before the fastest path from the shot, so whatever sits
above that curve is noise and can go.  It is worth removing on both sides:

  * on the streamer the curvelet cascade fills that window with
    coherent-looking wavetrains built out of what was incoherent noise going
    in - adjacent-trace coherence 0.010 before the interpolation, 0.379 after,
    against 0.68 -> 0.72 for the direct arrival itself, which the solve leaves
    alone because it was already coherent.  Random noise a network can learn to
    ignore; noise wearing the shape of an event it cannot.
  * the same window also carries the block-boundary artefact the cascade leaves
    at t = 0 (20x the record median at the first sample, decaying over ~30 ms),
    and on the DAS the recorded first sample, which is 8x the median before any
    processing touches it.

The boundary
------------
    t(shot, trace) = sqrt(offset^2 + depth_m^2) / velocity_m_s - lead_ms

Offsets come from the DAS geometry - source_xy from das_data_raw_meta.npz,
receiver positions from das_data_geom_meta.npz - and the streamer's 93
positions are read off the same array through streamer_trace_group, so both
domains are muted against one set of coordinates.

velocity_m_s is meant to be faster than the direct wave, not equal to it.  Too
fast only means the boundary sits earlier than it had to and some noise
survives; too slow eats the first arrival.  The parameters live in
config.MUTE_PARAMS.

Reading the output
------------------
One window per shot - before, after, and what was removed, with the boundary
drawn on all three so it can be seen sitting under the first break rather than
through it.  Closing a window brings up the next shot.  The alignment figure,
which comes last, is the one that settles it: every trace shifted so its own boundary is at zero and
stacked, which turns "is the mute clipping the arrival" into a curve that
should be flat and low to the left of zero and rise only to the right.

Edit config.MUTE_PARAMS and the settings block below, then run from the
repository root:

    python -m process.mute_direct
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np

import config as C
from utils.data import create_memmap, get_project_root, npy_shape

# ---------------------------------------------------------------- settings --

# "das" or "str".
DOMAIN = "das"

# Which stage to read.  "freq" is where the mute belongs in the pipeline -
# the last thing before normalisation - but any stage works, and "raw" is the
# one for seeing what the mute does before the interpolation has had a chance
# to turn the pre-arrival noise into wavetrains.  The sample interval comes
# from config.STAGE_DT_US and the receiver positions from the trace count, so
# the 0.5 ms streamer raw needs nothing else changed.
INPUT_STAGE = "raw"

# Draw a few shots and show them.  Writes nothing.
PREVIEW = True
PREVIEW_SHOTS = range(0, 551, 50)
SAMPLES = None
CLIP_PERC = 99.0

# Mute every shot and write the output.  Off while the velocity is being set.
WRITE_ALL = False

# Show interactively, and optionally save.  A falsy SAVE_DIR skips saving.
SHOW = True
SAVE_DIR = None

# ----------------------------------------------------------------- derived --

DT_US = C.STAGE_DT_US[DOMAIN][INPUT_STAGE]
_M = C.MUTE_PARAMS
VELOCITY = _M["velocity_m_s"]
DEPTH = _M["depth_m"]
LEAD_MS = _M["lead_ms"]
TAPER_MS = _M["taper_ms"]

INPUT_NPY = C.ARRAYS[DOMAIN][INPUT_STAGE]
OUTPUT_NPY = os.path.join(C.NPY_DIR,
                          f"{DOMAIN}_data_{INPUT_STAGE}_muted.npy")

# ---------------------------------------------------------------------------


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(get_project_root(), path)


def geometry(domain, n_traces):
    """Source and receiver coordinates, both from the DAS metadata.

    Which receiver set is wanted follows from the trace count, and every one of
    them is read off the DAS grid: the streamer's 24 recorded nodes and its 93
    interpolated positions are both defined as DAS group centres - that is what
    decimate_das established - so taking them from there keeps every stage on
    one set of coordinates rather than two that disagree by the 0.7 % scale
    error in the SEG-Y headers.
    """
    raw = np.load(resolve(C.META["das_raw"]))
    geom = np.load(resolve(C.META["das_geom"]))
    src = raw["source_xy"]
    centres = geom["centre_xy"]
    by_count = {
        len(centres): ("DAS group centres", centres),
        C.STR_TRACES: ("streamer interpolated positions",
                       centres[geom["streamer_trace_group"]]),
        C.STR_NODES: ("streamer recorded nodes",
                      centres[geom["streamer_node_group"]]),
        C.DAS_RAW_CHANNELS: ("DAS channels", raw["receiver_xy"]),
    }
    if n_traces not in by_count:
        raise SystemExit(f"no receiver positions for {n_traces} traces; "
                         f"known trace counts: {sorted(by_count)}")
    label, rec = by_count[n_traces]
    print(f"  receivers: {label} ({len(rec)})")
    return src, rec


def mute_times(src, rec):
    """Boundary in milliseconds, shape (n_shots, n_traces)."""
    off = np.hypot(src[:, None, 0] - rec[None, :, 0],
                   src[:, None, 1] - rec[None, :, 1])
    t = np.sqrt(off ** 2 + DEPTH ** 2) / VELOCITY * 1e3 - LEAD_MS
    return np.maximum(t, 0.0), off


def mute_weights(t_ms, nt, dt_ms):
    """(nt, n_traces) in [0, 1]: 0 above the boundary, 1 below, cosine between.

    Built per shot rather than per trace so the taper is one vectorised
    expression; at 1058 traces that matters.
    """
    t = np.arange(nt)[:, None] * dt_ms
    if TAPER_MS <= 0:
        return (t >= t_ms[None, :]).astype(np.float64)
    x = (t - t_ms[None, :]) / TAPER_MS
    return np.clip(0.5 - 0.5 * np.cos(np.pi * np.clip(x, 0.0, 1.0)), 0.0, 1.0)


def aligned_stack(data, t_ms, shots, nt, dt_ms, half_ms=300.0):
    """Mean |amplitude| against time measured from each trace's own boundary.

    The single most useful check: if the boundary is safe the curve is flat at
    the noise level left of zero and climbs only to the right of it.
    """
    half = int(half_ms / dt_ms)
    acc = np.zeros(2 * half + 1)
    cnt = 0
    for s in shots:
        g = np.abs(np.asarray(data[s], dtype=np.float64))
        idx = np.round(t_ms[s] / dt_ms).astype(int)
        for j, i0 in enumerate(idx):
            a, b = i0 - half, i0 + half + 1
            if a < 0 or b > nt:
                continue
            acc += g[a:b, j]
            cnt += 1
    return (acc / max(cnt, 1)), cnt, np.arange(-half, half + 1) * dt_ms


def gather_figure(shot, before, after, t_ms, dt_ms):
    """One shot: before, after, and what the mute took."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.5))
    clip = np.percentile(np.abs(before), CLIP_PERC)
    clip = clip if clip > 0 else 1.0
    nt, ntr = before.shape
    for col, (img, name) in enumerate(((before, "before"),
                                       (after, "after"),
                                       (before - after, "removed"))):
        ax = axes[col]
        ax.imshow(img, cmap="gray", vmin=-clip, vmax=clip, aspect="auto",
                  interpolation="nearest", extent=[0, ntr, nt * dt_ms, 0])
        ax.plot(np.arange(ntr) + 0.5, t_ms, color="r", lw=1.0,
                label="mute boundary")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("trace")
        ax.set_ylim(nt * dt_ms, 0)
    axes[0].set_ylabel("time [ms]")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"{DOMAIN} shot {shot}  -  v = {VELOCITY:g} m/s, "
                 f"depth {DEPTH:g} m, lead {LEAD_MS:g} ms, "
                 f"taper {TAPER_MS:g} ms")
    fig.tight_layout()
    return fig


def alignment_figure(rel_t, curve, cnt):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, log in zip(axes, (False, True)):
        (ax.semilogy if log else ax.plot)(rel_t, np.maximum(curve, 1e-12),
                                          lw=1.2)
        ax.axvline(0.0, color="r", ls="--", lw=1, label="mute boundary")
        ax.axvline(TAPER_MS, color="orange", ls=":", lw=1,
                   label=f"taper ends (+{TAPER_MS:g} ms)")
        ax.set_xlabel("time from the boundary [ms]")
        ax.set_ylabel("mean |amplitude|")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    axes[0].set_title(f"{DOMAIN}: energy about the mute boundary "
                      f"({cnt} traces)")
    axes[1].set_title("same, log scale")
    fig.tight_layout()
    return fig


def report_margin(rel_t, curve):
    """How much of what follows the boundary is being cut into.

    Compares the level just inside the muted side with the level well after the
    boundary.  A safe boundary sits in the quiet part, so the ratio is small.
    """
    before = curve[(rel_t >= -100) & (rel_t < 0)].mean()
    after = curve[(rel_t > TAPER_MS) & (rel_t <= TAPER_MS + 150)].mean()
    print(f"    mean |amplitude| in the 100 ms above the boundary: {before:.4g}")
    print(f"    mean |amplitude| in the 150 ms below the taper:    {after:.4g}")
    print(f"    ratio {before / max(after, 1e-30):.3f}"
          f"   ({'safe - the boundary is in the quiet zone' if before < 0.3 * after else 'CHECK - the boundary is cutting into the arrival'})")


def main():
    in_path = resolve(INPUT_NPY)
    if not os.path.isfile(in_path):
        raise SystemExit(f"not found: {in_path}")

    data = np.load(in_path, mmap_mode="r")
    n_shots, nt, n_traces = data.shape
    dt_ms = DT_US / 1000.0

    src, rec = geometry(DOMAIN, n_traces)
    if len(src) != n_shots or len(rec) != n_traces:
        raise SystemExit(f"geometry is {len(src)} shots x {len(rec)} traces, "
                         f"the array is {n_shots} x {n_traces}")
    t_ms, off = mute_times(src, rec)

    print(f"{INPUT_NPY}\n  {n_shots} shots x {nt} samples @ {dt_ms:g} ms "
          f"x {n_traces} traces  ({DOMAIN} {INPUT_STAGE})")
    print(f"  v {VELOCITY:g} m/s, depth {DEPTH:g} m, lead {LEAD_MS:g} ms, "
          f"taper {TAPER_MS:g} ms")
    print(f"  offset {off.min():.1f} .. {off.max():.1f} m")
    print(f"  boundary {t_ms.min():.0f} .. {t_ms.max():.0f} ms "
          f"(record is {nt * dt_ms:.0f} ms)")
    past = (t_ms >= nt * dt_ms).sum()
    print(f"  traces whose boundary is past the end of the record: {past} "
          f"({100 * past / t_ms.size:.1f} %) - those are muted entirely")
    print(f"  mean muted fraction of the record: "
          f"{100 * np.mean(np.clip(t_ms, 0, nt * dt_ms)) / (nt * dt_ms):.1f} %")

    figs = []
    if PREVIEW:
        bad = [s for s in PREVIEW_SHOTS if not 0 <= s < n_shots]
        if bad:
            raise SystemExit(f"preview shots out of range: {bad}")
        ns = nt if SAMPLES is None else min(SAMPLES, nt)
        shots = list(PREVIEW_SHOTS)
        out_dir = None
        if SAVE_DIR:
            out_dir = resolve(SAVE_DIR)
            os.makedirs(out_dir, exist_ok=True)

        # One window per shot, drawn and shown before the next is built:
        # plt.show() blocks until it is closed, so the shots arrive one at a
        # time rather than all at once, and only one gather is in memory.
        print(f"\n  preview: {len(shots)} shots, one window at a time")
        for n, shot in enumerate(shots, 1):
            g = np.asarray(data[shot], dtype=np.float64)
            y = g * mute_weights(t_ms[shot], nt, dt_ms)
            kept = np.sum(y ** 2) / max(np.sum(g ** 2), 1e-30)
            print(f"    [{n}/{len(shots)}] shot {shot:>4}: offset "
                  f"{off[shot].min():.0f}..{off[shot].max():.0f} m, boundary "
                  f"{t_ms[shot].min():.0f}..{t_ms[shot].max():.0f} ms, "
                  f"energy kept {100 * kept:.1f} %")
            fig = gather_figure(shot, g[:ns], y[:ns], t_ms[shot], dt_ms)
            if out_dir:
                q = os.path.join(out_dir,
                                 f"mute_direct_{DOMAIN}_shot{shot:04d}.png")
                fig.savefig(q, dpi=130)
                print(f"        wrote {q}")
            if SHOW:
                plt.show()
            plt.close(fig)

        curve, cnt, rel_t = aligned_stack(data, t_ms, shots, nt, dt_ms)
        report_margin(rel_t, curve)
        figs.append(("alignment", alignment_figure(rel_t, curve, cnt)))

    for name, fig in figs:
        if SAVE_DIR:
            q = os.path.join(resolve(SAVE_DIR),
                             f"mute_direct_{DOMAIN}_{name}.png")
            fig.savefig(q, dpi=130)
            print(f"  wrote {q}")
        if SHOW:
            plt.show()
        plt.close(fig)

    if not WRITE_ALL:
        print(f"\n  WRITE_ALL is off - {OUTPUT_NPY} not touched")
        return

    out_path = resolve(OUTPUT_NPY)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if os.path.isfile(out_path):
        print(f"\n  overwriting {OUTPUT_NPY} (was {npy_shape(out_path)})")
    out = create_memmap(out_path, data.shape)
    print(f"  output {out.shape}, {out.nbytes / 1e9:.2f} GB")

    t0 = time.time()
    for i in range(n_shots):
        g = np.asarray(data[i], dtype=np.float64)
        out[i] = (g * mute_weights(t_ms[i], nt, dt_ms)).astype(np.float32)
        if (i + 1) % 50 == 0 or i + 1 == n_shots:
            el = time.time() - t0
            print(f"    {i + 1}/{n_shots} shots  {el:.0f}s elapsed, "
                  f"eta {el / (i + 1) * (n_shots - i - 1):.0f}s")
    out.flush()
    print(f"\n  wrote {out_path}")


if __name__ == "__main__":
    main()
