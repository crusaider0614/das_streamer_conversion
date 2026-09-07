"""Receiver-geometry QC for the Pohang DAS / streamer SEG-Y files.

Two jobs:

1.  Check that the files of each kind describe the *same* receiver array, by
    comparing the per-channel receiver coordinates file against file.
2.  Map the DAS and streamer receiver positions together, on a true 1:1 metric
    scale so that a metre of easting is drawn the same length as a metre of
    northing.

Coordinates come from the trace headers (GroupX/GroupY, bytes 81-88) with the
SEG-Y coordinate scalar (byte 71) applied.  The survey is in UTM metres
(CoordinateUnits = 1), so grid north is used as north.

Edit the settings block below, then run from the repository root:

    python -m process.check_receiver_geometry
"""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import segyio
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from segyio import TraceField as TF

from utils.data import get_project_root

# ---------------------------------------------------------------- settings --

# Directory holding the das/ and streamer/ subdirectories.
DATA_DIR = os.path.join(get_project_root(), "data", "pohang_shore")

# SEG-Y files to check, looked up inside each kind's subdirectory.
FILES = ["data009.segy", "data010.segy", "data011.segy", "data012.segy"]

# Subdirectories to compare, and the order they are drawn in.
KINDS = ["das", "streamer"]

# Add a second panel with the shot positions.  They span ~2 km against the
# arrays' ~90 m, so the receivers need their own panel to stay legible.
SHOW_SOURCES = True

# Where to write the figure; None to only show it on screen.
SAVE_PATH = None

# Header coordinates are stored to 0.01 m, so anything above this is a real
# difference rather than a rounding artefact.
TOL_M = 0.011

# ---------------------------------------------------------------------------

STYLE = {
    "das": dict(color="tab:red", marker="o", s=14),
    "streamer": dict(color="tab:blue", marker="^", s=34),
}


def apply_coordinate_scalar(raw, scalar):
    """Apply the SEG-Y byte-71 coordinate scalar: positive multiplies,
    negative divides, 0 and 1 leave the value alone."""
    raw = np.asarray(raw, dtype=np.float64)
    scalar = np.asarray(scalar, dtype=np.float64)
    out = raw.copy()
    neg = scalar < 0
    pos = scalar > 1
    out[neg] = raw[neg] / -scalar[neg]
    out[pos] = raw[pos] * scalar[pos]
    return out


def read_geometry(path):
    """Return the per-channel receiver table and a few file statistics.

    `receivers` maps channel number -> array of the distinct (x, y) that
    channel is given anywhere in the file; a well-formed file gives one row
    per channel.  Traces whose coordinates are both zero are treated as
    un-headered and skipped.
    """
    with segyio.open(path, "r", strict=False, ignore_geometry=True) as f:
        n_traces = f.tracecount
        n_samples = len(f.samples)
        dt_us = int(f.bin[segyio.BinField.Interval])
        gx = np.asarray(f.attributes(TF.GroupX)[:], dtype=np.int64)
        gy = np.asarray(f.attributes(TF.GroupY)[:], dtype=np.int64)
        sx = np.asarray(f.attributes(TF.SourceX)[:], dtype=np.int64)
        sy = np.asarray(f.attributes(TF.SourceY)[:], dtype=np.int64)
        scalar = np.asarray(f.attributes(TF.SourceGroupScalar)[:], dtype=np.int64)
        channel = np.asarray(f.attributes(TF.TraceNumber)[:], dtype=np.int64)
        record = np.asarray(f.attributes(TF.FieldRecord)[:], dtype=np.int64)

    headered = (gx != 0) | (gy != 0)
    rx = apply_coordinate_scalar(gx[headered], scalar[headered])
    ry = apply_coordinate_scalar(gy[headered], scalar[headered])
    ch = channel[headered]

    receivers = {}
    for c in np.unique(ch):
        m = ch == c
        receivers[int(c)] = np.unique(np.stack([rx[m], ry[m]], axis=1), axis=0)

    src_ok = (sx != 0) | (sy != 0)
    if src_ok.any():
        sources = np.unique(
            np.stack(
                [
                    apply_coordinate_scalar(sx[src_ok], scalar[src_ok]),
                    apply_coordinate_scalar(sy[src_ok], scalar[src_ok]),
                ],
                axis=1,
            ),
            axis=0,
        )
    else:
        sources = np.zeros((0, 2))

    return {
        "path": path,
        "n_traces": n_traces,
        "n_samples": n_samples,
        "dt_us": dt_us,
        "n_headered": int(headered.sum()),
        "n_records": int(np.unique(record[headered]).size) if headered.any() else 0,
        "receivers": receivers,
        "sources": sources,
    }


def receiver_array(receivers):
    """Flatten the per-channel table into ordered (channel, x, y) arrays,
    keeping only channels that have a single unambiguous position."""
    chans = sorted(c for c, pts in receivers.items() if len(pts) == 1)
    xy = np.array([receivers[c][0] for c in chans], dtype=np.float64).reshape(-1, 2)
    return np.array(chans, dtype=np.int64), xy


def merged_receivers(infos):
    """One receiver table for a whole kind, pooled over its files."""
    merged = {}
    for info in infos:
        for c, pts in info["receivers"].items():
            if len(pts) == 1:
                merged.setdefault(c, pts[0])
    chans = sorted(merged)
    xy = np.array([merged[c] for c in chans], dtype=np.float64).reshape(-1, 2)
    return chans, xy


def describe(info):
    chans, xy = receiver_array(info["receivers"])
    name = os.path.basename(info["path"])
    print(f"  {name}: {info['n_traces']:>7d} traces, {info['n_samples']} samples "
          f"@ {info['dt_us'] / 1000:g} ms")
    print(f"      headers present on {info['n_headered']:>7d} traces "
          f"({100.0 * info['n_headered'] / info['n_traces']:6.2f} %), "
          f"{len(chans)} receiver channels resolved")
    ambiguous = [c for c, pts in info["receivers"].items() if len(pts) > 1]
    if ambiguous:
        print(f"      WARNING: {len(ambiguous)} channel(s) carry more than one "
              f"position within this file: {ambiguous[:10]}")
    if len(chans) > 1:
        span = float(np.hypot(*(xy[-1] - xy[0])))
        step = np.hypot(*np.diff(xy, axis=0).T)
        print(f"      channels {chans[0]}-{chans[-1]}, span {span:8.2f} m, "
              f"spacing {step.mean():.3f} m (min {step.min():.3f}, max {step.max():.3f})")
    return chans, xy


def report_structure(infos):
    """Work out how many receiver channels each shot record holds.

    When the trace headers are intact this comes straight out of them.  When
    they are not, the only remaining handle is that each file holds whole shot
    records, so every trace count is a multiple of the channels-per-record and
    their GCD bounds it from above - which needs at least two files.

    Returns (n_channels, n_known_positions); n_channels is None when the array
    size could not be established.
    """
    n_known = len({c for i in infos for c in i["receivers"]})
    complete = [i for i in infos if i["n_headered"] == i["n_traces"]]

    if complete:
        for info in complete:
            n_ch = len(info["receivers"])
            n_rec = info["n_records"]
            name = os.path.basename(info["path"])
            print(f"  {name}: headers give {n_ch} channels x {n_rec} records", end="")
            if n_ch * n_rec == info["n_traces"]:
                print(" = trace count, consistent")
            else:
                print(f" = {n_ch * n_rec}, but the file holds "
                      f"{info['n_traces']} traces - INCONSISTENT")
        sizes = {len(i["receivers"]) for i in complete}
        if len(sizes) > 1:
            print(f"  WARNING: fully-headered files disagree on the channel "
                  f"count: {sorted(sizes)}")
        n_channels = max(sizes)
        if len(complete) < len(infos):
            print(f"  ({len(infos) - len(complete)} further file(s) are not "
                  f"fully headered; channel count taken from the complete ones)")
        return n_channels, n_known

    if len(infos) < 2:
        print("  headers are incomplete and there is only one file, so the "
              "number of channels per record cannot be determined")
        return None, n_known

    g = infos[0]["n_traces"]
    for i in infos[1:]:
        g = math.gcd(g, i["n_traces"])
    print(f"  headers are incomplete; trace counts bound the array at "
          f"{g} channels per record -> {[i['n_traces'] // g for i in infos]} "
          f"records per file")
    return g, n_known


def compare_kind(kind, infos):
    """Compare the per-channel receiver coordinates of every file of one kind
    against the first file, over the channels they share."""
    print(f"\n  --- receiver-coordinate consistency across the {len(infos)} "
          f"{kind} files ---")
    ref_name = os.path.basename(infos[0]["path"])
    ref = infos[0]["receivers"]
    all_match = len(infos) > 1
    if not all_match:
        print("  only one file present - nothing to compare against")

    for info in infos[1:]:
        name = os.path.basename(info["path"])
        other = info["receivers"]
        shared = sorted(set(ref) & set(other))
        only_ref = sorted(set(ref) - set(other))
        only_other = sorted(set(other) - set(ref))

        d = np.array([
            np.hypot(*(ref[c][0] - other[c][0]))
            for c in shared
            if len(ref[c]) == 1 and len(other[c]) == 1
        ])
        if not d.size:
            print(f"  {name} vs {ref_name}: no comparable channels")
            all_match = False
            continue

        worst = float(d.max())
        verdict = "IDENTICAL" if worst <= TOL_M else "DIFFERENT"
        if verdict != "IDENTICAL":
            all_match = False
        print(f"  {name} vs {ref_name}: {d.size:>4d} shared channels, "
              f"max offset {worst:.3f} m -> {verdict}")
        if only_ref or only_other:
            print(f"      channels only in {ref_name}: {len(only_ref)}, "
                  f"only in {name}: {len(only_other)}")

    if all_match:
        print(f"  => all {kind} files agree on every shared channel "
              f"(within {TOL_M * 1000:.0f} mm).")
    elif len(infos) > 1:
        print(f"  => the {kind} files do NOT all describe the same receiver array.")

    n_ch, n_known = report_structure(infos)
    if n_ch is not None and n_known < n_ch:
        print(f"  NOTE: only {n_known} of those {n_ch} channels carry "
              f"coordinates anywhere in these files. The check above covers that "
              f"headered subset only; the remaining "
              f"{n_ch - n_known} channels have blank trace headers and their "
              f"positions cannot be verified from the SEG-Y.")
    return all_match, n_ch, n_known


def nice_length(span, frac=0.25):
    """A round number of metres about `frac` of the given span."""
    raw = max(span, 1e-9) * frac
    exp = 10.0 ** math.floor(math.log10(raw))
    return min([1, 2, 5, 10], key=lambda m: abs(m * exp - raw)) * exp


def add_scale_bar(ax, span):
    """Anchored scale bar. AnchoredSizeBar resolves its length through
    transData at draw time, so it stays correct after the equal-aspect pass
    has adjusted the data limits."""
    length = nice_length(span)
    label = f"{length / 1000:g} km" if length >= 1000 else f"{length:g} m"
    ax.add_artist(AnchoredSizeBar(
        ax.transData, length, label, "lower left",
        pad=0.4, borderpad=0.8, sep=4, frameon=False, color="0.15",
        size_vertical=0.0,
    ))


def add_north_arrow(ax):
    ax.annotate(
        "N",
        xy=(0.955, 0.955), xytext=(0.955, 0.855),
        xycoords="axes fraction", textcoords="axes fraction",
        ha="center", va="center", fontsize=11, color="0.15",
        arrowprops=dict(arrowstyle="-|>", color="0.15", lw=1.5),
    )


def draw_receivers(ax, per_kind, coverage):
    span = 0.0
    for kind in KINDS:
        infos = per_kind.get(kind)
        if not infos:
            continue
        chans, xy = merged_receivers(infos)
        if not len(chans):
            continue
        known, total = coverage.get(kind, (len(chans), len(chans)))
        label = f"{kind} receivers ({known} ch"
        label += f" of {total})" if total and known != total else ")"
        ax.scatter(xy[:, 0], xy[:, 1], zorder=3, label=label, **STYLE[kind])
        span = max(span, float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])))
    return span


def draw_sources(ax, per_kind):
    span = 0.0
    for kind in KINDS:
        infos = per_kind.get(kind)
        if not infos:
            continue
        src = [i["sources"] for i in infos if len(i["sources"])]
        if not src:
            continue
        src = np.unique(np.concatenate(src), axis=0)
        ax.scatter(src[:, 0], src[:, 1], s=8, marker="*", zorder=2,
                   color=STYLE[kind]["color"], alpha=0.35,
                   label=f"{kind} sources ({len(src)})")
        span = max(span, float(np.ptp(src[:, 0])), float(np.ptp(src[:, 1])))
    return span


def finish_axes(ax, title, span):
    # 1:1 metric scale - one metre east is drawn as long as one metre north.
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("UTM easting [m]")
    ax.set_ylabel("UTM northing [m]")
    ax.set_title(title, fontsize=11)
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.tick_params(axis="x", labelrotation=30)
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    add_north_arrow(ax)
    add_scale_bar(ax, span)


def plot_geometry(per_kind, coverage):
    if SHOW_SOURCES:
        # Sources spread over ~2 km while the receivers occupy ~90 m, so at a
        # shared 1:1 scale the arrays would be a dot. Give them their own panel.
        fig, (ax_all, ax_recv) = plt.subplots(1, 2, figsize=(13, 7))
        span = max(draw_receivers(ax_all, per_kind, coverage),
                   draw_sources(ax_all, per_kind))
        finish_axes(ax_all, "Survey layout: sources and receivers", span)
        span = draw_receivers(ax_recv, per_kind, coverage)
        finish_axes(ax_recv, "Receivers only", span)
    else:
        fig, ax = plt.subplots(figsize=(8, 8))
        span = draw_receivers(ax, per_kind, coverage)
        finish_axes(ax, "Receiver positions", span)

    fig.suptitle("Pohang DAS vs streamer geometry (1:1 metric scale)",
                 fontsize=13)
    fig.tight_layout()

    if SAVE_PATH:
        fig.savefig(SAVE_PATH, dpi=200)
        print(f"\nsaved figure to {SAVE_PATH}")
    plt.show()


def main():
    per_kind = {}
    coverage = {}
    for kind in KINDS:
        kind_dir = os.path.join(DATA_DIR, kind)
        paths = [os.path.join(kind_dir, n) for n in FILES]
        missing = [p for p in paths if not os.path.isfile(p)]
        if missing:
            print(f"[{kind}] missing: {[os.path.basename(p) for p in missing]}")
            paths = [p for p in paths if os.path.isfile(p)]
        if not paths:
            continue

        print("=" * 78)
        print(f"{kind.upper()}  ({kind_dir})")
        infos = []
        for p in paths:
            info = read_geometry(p)
            describe(info)
            infos.append(info)
        _, n_ch, n_known = compare_kind(kind, infos)
        per_kind[kind] = infos
        coverage[kind] = (n_known, n_ch)

    print("=" * 78)
    plot_geometry(per_kind, coverage)


if __name__ == "__main__":
    main()
