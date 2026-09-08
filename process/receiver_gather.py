"""Receiver gathers at the 24 streamer node positions, streamer against DAS.

    str  numpy/str_data_line.npy  (551, 2000, 24)   @ 1 ms
    das  numpy/das_data_line.npy  (551, 2000, 3175) @ 1 ms

Both from `line`, which is the stage that makes this view worth drawing: the
shots have been rolled onto the fitted survey line, so a receiver gather
sorted by along-line position no longer carries the boat's lateral wander as
ragged moveout.  Both domains come out of it on one time axis, in one band,
and muted against the same projected geometry - see process/line_static.py.

A receiver gather fixes the receiver and puts the shots on the horizontal axis,
which is the view that separates the two kinds of noise the shot gathers cannot
tell apart:

    a horizontal band, or a whole panel that is louder than its neighbours,
        belongs to the receiver - a bad channel, or a length of cable
    a vertical stripe belongs to the shot - a source misfire, a passing vessel,
        weather during that one record

And putting the DAS beside it at the same position separates where the noise
came from.  The two instruments share the seabed and the water column but not
their electronics: something in the water shows on both, something in the
streamer's own recording system shows on the streamer alone.

The pairing is exact rather than nearest-neighbour, because the node positions
are defined on the DAS grid to begin with - see utils.mute.geometry.  Which
index gives it depends on how many traces the DAS array has, and node_to_das()
handles both: `streamer_node_channel` for the 3175 raw channels that `line`
carries, `streamer_node_group` for the 1058 group centres of `geom` and
`freq`.  The two disagree by at most 0.51 m, which is the 0.7 % scale error in
the SEG-Y header coordinates and not a real displacement.

Both domains are band-passed and both are muted ahead of the direct arrival,
so what is left between them is the instruments and not the processing.

Amplitudes are not comparable between the two.  DAS reports strain rate and the
streamer reports pressure, in unrelated units, so each panel is scaled to its
own RMS and given its own envelope gain before it is drawn; what the comparison
is for is where the energy sits in shot and time, not how big it is.  See
DISPLAY for the gain, and why a receiver gather in particular wants one - the
shot line closes on the receiver and opens again, which is a ten-fold amplitude
swing across the panel that has nothing to do with noise.

Edit the settings block below, then run from the repository root:

    python -m process.receiver_gather
"""

import os

import matplotlib
import numpy as np

import config as C
# The log-envelope gain, so the preview and the pipeline stage cannot drift
# apart.  See DISPLAY.
import process.logenv_process as L
# The along-line ordering and the fitted line it comes from.  See
# SORT_BY_POSITION.
import process.shot_geometry as G
from utils.data import get_project_root

import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------- settings --

# Which stage of each domain to read.  `line` on both sides, which is what
# makes them comparable: process/line_static.py puts them on one time axis
# (2000 samples at 1 ms), band-passes both with FILTER_PARAMS, and mutes both
# against the same projected geometry - and in the same order, band-pass
# before mute, so the two records meet their mute boundary carrying the same
# band.  Reading one domain at a different stage would put most of the
# difference in the processing rather than the instruments.
#
# Still pre-normalisation, so the per-panel RMS scaling below is what puts the
# two on a drawable scale.
STAGE = "line"
STR_NPY = C.ARRAYS["str"][STAGE]
DAS_NPY = C.ARRAYS["das"][STAGE]
STR_DT_MS = C.STAGE_DT_US["str"][STAGE] / 1000.0
DAS_DT_MS = C.STAGE_DT_US["das"][STAGE] / 1000.0

# Draw the mute boundary the stage used, from META["*_line"].  It is the one
# check that the rolled data and its mute agree: the boundary should sit just
# under the first break at every shot, with no step where the sorted order
# jumps between passes.  Ignored for stages that have no sidecar.
SHOW_MUTE = True

# Streamer nodes to draw, 0-23.  They fill the columns.
RECEIVERS = range(0, 24)
RECEIVERS_PER_FIGURE = 3

# Shots to include, and the time window in milliseconds.  None takes all.
SHOTS = None
WINDOW_MS = (0.0, 2000.0)

# Put the shots in order along the survey line instead of in recording order.
#
# The survey is four passes over one 2.1 km line, so recording order visits
# each stretch of seabed four times, hundreds of shots apart.  Sorted, the
# neighbours on the axis are neighbours on the ground: the shot interval drops
# from 15.24 m to a median of 3.33 m, which is what the receiver spacing is,
# and the axis stops being aliased over most of the band.
#
# It is not a regular grid.  The four passes were fired independently, so the
# sorted gaps run from nothing to 15.2 m with a p90 of 7.59 m, and the shots
# that land next to each other are up to 30 m apart across the line - which
# enters the offset only as a**2 / (2 * offset), a fraction of a millisecond.
# See process/shot_geometry.py, which writes the ordering out as text.
SORT_BY_POSITION = True

# Inches per panel.
PANEL_W = 4.2
PANEL_H = 4.6

CLIP_PERC = 99.0
CMAP = "seismic"

# "linear" draws true amplitude, "log" the log-envelope gain from
# config.LOG_SCALE_PARAMS - the same transform process/logenv_process.py
# applies, per domain.
#
# Worth having on here.  A receiver gather's amplitude is dominated by offset:
# the shot line closes on the receiver and opens again, so the panel runs ten
# times louder in the middle than at the ends, and on a linear clip the quiet
# shots are flat grey while the loud ones saturate.  The envelope gain flattens
# that and lets all 551 shots be read at once, which is the point of looking at
# a receiver gather in the first place.
#
# The gain is fitted per panel rather than shared: these are different
# receivers on different instruments, and there is nothing to be gained by
# forcing one scale on them.  calculate_logscale is not scale invariant either
# - log_base is added to the envelope in the envelope's own units - so each
# panel is put at config.TARGET_RMS first, as logenv_process does.
DISPLAY = "linear"
LOG_PARAMS_KEY = {"str": "str", "das": "das"}

# Overrides log_base for the display only, leaving config alone.  None uses the
# config value; smaller lifts the weak amplitudes harder.
LOG_BASE = None

# Show interactively, and optionally save.  A falsy SAVE_DIR skips saving.
SHOW = True
SAVE_DIR = None
BACKENDS = ("QtAgg", "Qt5Agg", "TkAgg", "MacOSX")

# ---------------------------------------------------------------------------


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(get_project_root(), path)


def show_figures():
    """Swap an interactive backend in if the current one cannot draw."""
    if not SHOW:
        return False
    if matplotlib.get_backend().lower() not in ("agg", "pdf", "ps", "svg"):
        return True
    for name in BACKENDS:
        try:
            matplotlib.use(name, force=True)
        except Exception:
            continue
        print(f"  backend: {matplotlib.get_backend()}")
        return True
    print(f"  warning: none of {', '.join(BACKENDS)} is installed - "
          f"set SAVE_DIR and look at the files")
    return False


def node_to_das(n_traces):
    """DAS trace index for each of the 24 streamer nodes, by trace count.

    `line` carries the 3175 raw channels, so the node index is its channel
    number; `geom` and `freq` carry the 1058 group centres, so it is the group
    the node falls in.  Both are exact - the node positions were defined on
    this grid - and they agree to 0.51 m.
    """
    g = np.load(resolve(C.META["das_geom"]))
    if n_traces == C.DAS_TRACES:
        node = g["streamer_node_group"]
        xy = g["centre_xy"][node]
    elif n_traces == C.DAS_RAW_CHANNELS:
        r = np.load(resolve(C.META["das_raw"]))
        chan, want = r["receiver_channel"], g["streamer_node_channel"]
        node = np.searchsorted(chan, want)
        if not np.array_equal(chan[node], want):
            raise SystemExit(f"streamer node channels {want} are not all in "
                             f"{C.META['das_raw']}")
        xy = r["receiver_xy"][node]
    else:
        raise SystemExit(f"no node mapping for a {n_traces}-trace DAS array; "
                         f"known: {C.DAS_TRACES} groups, "
                         f"{C.DAS_RAW_CHANNELS} channels")
    step = np.hypot(*(xy[1:] - xy[:-1]).T)
    return node, xy, step


def mute_boundaries(shots, node):
    """(str, das) mute boundary in ms for the drawn shots, or (None, None).

    Read from the sidecars rather than recomputed, so what is drawn is the
    boundary the samples actually went through.
    """
    if not SHOW_MUTE:
        return None, None
    out = []
    for tag, cols in (("str", np.arange(C.STR_NODES)), ("das", node)):
        p = resolve(C.META.get(f"{tag}_{STAGE}", ""))
        if not os.path.isfile(p):
            print(f"  no {tag} sidecar for stage {STAGE!r} - "
                  f"mute boundary not drawn")
            return None, None
        t = np.load(p)["mute_boundary_ms"]
        out.append(t[np.ix_(np.asarray(shots), np.asarray(cols))])
    return out[0], out[1]


def window(nt, dt_ms):
    if WINDOW_MS is None:
        return slice(0, nt)
    a = max(0, int(round(WINDOW_MS[0] / dt_ms)))
    b = min(nt, int(round(WINDOW_MS[1] / dt_ms)))
    if b - a < 2:
        raise SystemExit(f"WINDOW_MS {WINDOW_MS} is empty on {nt} samples "
                         f"at {dt_ms:g} ms")
    return slice(a, b)


def gather(data, trace, shots, win):
    """(n_samples, n_shots) for one receiver: the shots side by side."""
    out = np.empty((win.stop - win.start, len(shots)), dtype=np.float64)
    for j, s in enumerate(shots):
        out[:, j] = data[s, win, trace]
    return out


def unit_rms(g):
    """The panel scaled to config.TARGET_RMS over its live samples.

    Live means the rows the mute left alone; averaging the zeros in would make
    the scale depend on how much of the record was muted, which varies with
    offset and so with shot.
    """
    live = g[np.any(g != 0.0, axis=1)]
    r = float(np.sqrt(np.mean(live ** 2))) if live.size else 0.0
    return (g * (C.TARGET_RMS / r) if r > 0 else g), r


def log_gain(panel, tag):
    """The panel drawn with the log-envelope gain, and a label for it.

    The gain is fitted on the same panel it is applied to.  The axes are
    (time, shot) here rather than (time, trace), so smooth_sigma's second
    number smooths across shots - which is the right thing, since neighbouring
    shots see nearly the same wavefield.
    """
    if DISPLAY == "linear":
        return panel, "true amplitude"
    if DISPLAY != "log":
        raise SystemExit(f'DISPLAY must be "linear" or "log", not {DISPLAY!r}')
    lsp = dict(C.LOG_SCALE_PARAMS[LOG_PARAMS_KEY[tag]])
    if LOG_BASE is not None:
        lsp["log_base"] = LOG_BASE
    return panel * L.logscale(panel, lsp), (
        f"log envelope, base {lsp['log_base']:g}, "
        f"sigma {lsp['smooth_sigma']}, at rms {C.TARGET_RMS:g}")


def figure(cols, shots, t0, t1, along, t_str=None, t_das=None):
    """cols: (node, das_trace, distance_m, str_panel, das_panel) per column.

    t0, t1 are the window in milliseconds; both domains are drawn on it even
    though they are sampled differently.  `along` is the along-line position of
    each column when the shots are sorted, and None when they are not: sorted,
    the horizontal axis is the position in the ordering rather than the shot
    number, because the shot numbers are no longer in order, and a second axis
    on top carries the metres.
    """
    ncol = len(cols)
    fig, axes = plt.subplots(3, ncol, squeeze=False,
                             figsize=(PANEL_W * ncol, PANEL_H * 3),
                             height_ratios=[3, 3, 1.4])
    xs = np.arange(len(shots)) if along is not None else np.asarray(shots)
    x0, x1 = xs[0] - 0.5, xs[-1] + 0.5
    xlabel = "position along the line [sorted shot]" if along is not None \
        else "shot"
    img_ref = None

    gain_label = "true amplitude"
    for col, (node, dtr, dist, sp, dp) in enumerate(cols):
        for row, (panel, label, dt_ms, tag, tmute) in enumerate(
                ((sp, f"str node {node}", STR_DT_MS, "str",
                  None if t_str is None else t_str[:, node]),
                 (dp, f"das trace {dtr}", DAS_DT_MS, "das",
                  None if t_das is None else t_das[:, node]))):
            y, rms = unit_rms(panel)
            y, gain_label = log_gain(y, tag)
            clip = float(np.percentile(np.abs(y), CLIP_PERC)) or 1.0
            ax = axes[row][col]
            ax.imshow(y, cmap=CMAP, vmin=-clip, vmax=clip, aspect="auto",
                      interpolation="nearest",
                      extent=[x0, x1, t0 + panel.shape[0] * dt_ms, t0])
            if tmute is not None:
                ax.plot(xs, tmute, color="lime", lw=0.9,
                        label="mute boundary")
                if row == 0 and col == 0:
                    ax.legend(fontsize=7, loc="lower left")
            ax.set_title(f"{label}   rms {rms:.4g}", fontsize=10)
            if col == 0:
                ax.set_ylabel("time [ms]")
            else:
                ax.tick_params(labelleft=False)
            ax.tick_params(labelsize=8)
            if row == 0 and along is not None:
                # Metres along the line, on top.  The mapping is the sorted
                # positions themselves, so the ticks land where the shots
                # actually are and the uneven spacing shows.
                sec = ax.secondary_xaxis(
                    "top",
                    functions=(lambda v: np.interp(v, xs, along),
                               lambda m: np.interp(m, along, xs)))
                sec.set_xlabel("along the line [m]", fontsize=8)
                sec.tick_params(labelsize=7)
            # Linked pan and zoom: same columns on x, same milliseconds on y.
            if img_ref is None:
                img_ref = ax
            else:
                ax.sharex(img_ref)
                ax.sharey(img_ref)

        ax = axes[2][col]
        for panel, label in ((sp, "str"), (dp, "das")):
            r = np.sqrt((panel ** 2).mean(axis=0))
            m = np.median(r[r > 0]) if (r > 0).any() else 1.0
            ax.semilogy(xs, np.maximum(r / m, 1e-6), lw=0.8, label=label)
        ax.axhline(1.0, color="k", lw=0.5, alpha=0.4)
        ax.set_xlim(x0, x1)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
        if col == 0:
            ax.set_ylabel("per-shot RMS\n/ median")
            ax.legend(fontsize=8)
        ax.set_title(f"node {node} at {dist:.1f} m along the array",
                     fontsize=9)

    how = ("sorted along the line" if along is not None
           else "in recording order")
    fig.suptitle(f"receiver gathers, {len(shots)} shots {how}, "
                 f"{t0:.0f}-{t1:.0f} ms   {STR_NPY} vs {DAS_NPY}\n"
                 f"images: {gain_label}, fitted per panel - the two "
                 f"instruments do not share units; the RMS curve is linear")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0 - 0.7 / (PANEL_H * 3)))
    return fig


def main():
    show = show_figures()
    if not show and not SAVE_DIR:
        raise SystemExit("nothing to do: SHOW is off (or has no backend) and "
                         "SAVE_DIR is not set")

    sp, dp = resolve(STR_NPY), resolve(DAS_NPY)
    for p in (sp, dp):
        if not os.path.isfile(p):
            raise SystemExit(f"not found: {p}")
    s_data = np.load(sp, mmap_mode="r")
    d_data = np.load(dp, mmap_mode="r")
    if s_data.shape[0] != d_data.shape[0]:
        raise SystemExit(f"shot counts differ: {s_data.shape[0]} vs "
                         f"{d_data.shape[0]}")

    if s_data.shape[2] != C.STR_NODES:
        raise SystemExit(f"{STR_NPY} has {s_data.shape[2]} traces; this view "
                         f"is the {C.STR_NODES} recorded nodes")
    node, xy, step = node_to_das(d_data.shape[2])
    dist = np.concatenate([[0.0], np.cumsum(step)])
    shots = list(range(s_data.shape[0]) if SHOTS is None else SHOTS)
    along = None
    if SORT_BY_POSITION:
        o = G.sorted_order()
        rank = {int(s): i for i, s in enumerate(o["orig_idx"])}
        shots = sorted(shots, key=lambda s: rank[s])
        along = np.array([o["along_m"][rank[s]] for s in shots])
        gap = np.diff(along)
        print(f"  sorted along the line: {along[0]:.0f}..{along[-1]:.0f} m, "
              f"gaps median {np.median(gap):.2f} m, "
              f"p90 {np.percentile(gap, 90):.2f}, max {gap.max():.2f}")
    s_win = window(s_data.shape[1], STR_DT_MS)
    d_win = window(d_data.shape[1], DAS_DT_MS)
    t_str, t_das = mute_boundaries(shots, node)

    print(f"{STR_NPY}  {s_data.shape} @ {STR_DT_MS:g} ms")
    print(f"{DAS_NPY}  {d_data.shape} @ {DAS_DT_MS:g} ms")
    t0, t1 = s_win.start * STR_DT_MS, s_win.stop * STR_DT_MS
    print(f"  {len(shots)} shots, {t0:.0f}-{t1:.0f} ms "
          f"({s_win.stop - s_win.start} str samples, "
          f"{d_win.stop - d_win.start} das)")
    print(f"  node -> das trace: "
          + " ".join(f"{n}:{t}" for n, t in enumerate(node) if n % 4 == 0))

    out_dir = None
    if SAVE_DIR:
        out_dir = resolve(SAVE_DIR)
        os.makedirs(out_dir, exist_ok=True)

    picked = [r for r in RECEIVERS]
    bad = [r for r in picked if not 0 <= r < C.STR_NODES]
    if bad:
        raise SystemExit(f"receivers out of range 0..{C.STR_NODES - 1}: {bad}")

    chunk = []
    for n, rec in enumerate(picked, 1):
        dtr = int(node[rec])
        print(f"\n  [{n}/{len(picked)}] node {rec} -> das trace {dtr}, "
              f"{dist[rec]:.1f} m along the array", flush=True)
        s_panel = gather(s_data, rec, shots, s_win)
        d_panel = gather(d_data, dtr, shots, d_win)
        for panel, tag in ((s_panel, "str"), (d_panel, "das")):
            r = np.sqrt((panel ** 2).mean(axis=0))
            live = r[r > 0]
            if not live.size:
                print(f"      {tag}: empty")
                continue
            m = np.median(live)
            loud = int((r > 5 * m).sum())
            print(f"      {tag}: rms median {m:.4g}, max {r.max():.4g} "
                  f"({r.max() / m:.0f}x), {loud} shots over 5x, "
                  f"{int((r == 0).sum())} empty", flush=True)
        chunk.append((rec, dtr, dist[rec], s_panel, d_panel))

        if len(chunk) < RECEIVERS_PER_FIGURE and n < len(picked):
            continue
        fig = figure(chunk, shots, t0, t1, along, t_str, t_das)
        if out_dir:
            p = os.path.join(out_dir, f"receiver_gather_"
                             f"{chunk[0][0]:02d}-{chunk[-1][0]:02d}.png")
            fig.savefig(p, dpi=130)
            print(f"        wrote {p}", flush=True)
        if show:
            plt.show()
        plt.close(fig)
        chunk = []


if __name__ == "__main__":
    main()
