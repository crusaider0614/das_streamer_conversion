"""Where the shots actually are, and how straight the line through them is.

    numpy/das_data_raw_meta.npz  ->  source_xy  (551, 2)

Written to settle whether the shot axis can stand in for a spatial axis.  The
survey is four passes over one 2.1 km line, so sorting the shots by position
would take the 15.2 m shot interval down to about 3.3 m - but only if the four
passes lie on top of each other, and only if what looks like scatter is not
just the line's own bend.

The panels answer that in order:

    map view, true aspect          what the survey looks like from above
    along against across          the same thing with the cross axis blown up
                                  30x, with polynomial fits of degree 1 to 5
                                  laid over it
    residual from each fit        what is left once the fit is subtracted, by
                                  pass; if a curve were the answer the residual
                                  would collapse as the degree rises

The cross axis has to be exaggerated to be visible at all: the line runs 2116 m
and the shots sit within 67 m of it.

Edit the settings block below, then run from the repository root:

    python -m process.shot_geometry
"""

import os

import matplotlib
import numpy as np

import config as C
from utils.data import get_project_root

import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------- settings --

DEGREES = (1, 2, 3, 4, 5)

# The degree whose residual the third panel and the printout use.
RESIDUAL_DEGREE = 5

# A pass is a run of shots between two reversals of the along-line direction;
# runs shorter than this are noise in the direction, not a pass.
MIN_PASS_SHOTS = 20

FIGSIZE = (15.0, 11.0)

# Show interactively, and optionally save.  A falsy SAVE_PATH skips saving.
SHOW = True
SAVE_PATH = os.path.join(C.DATA_DIR, "shot_geometry.png")
BACKENDS = ("QtAgg", "Qt5Agg", "TkAgg", "MacOSX")

# The shot ordering, written as text so it can be read by anything.  A falsy
# path skips it.  See sorted_order() for what the columns mean.
SAVE_ORDER_PATH = os.path.join(C.DATA_DIR, "shot_order.txt")

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
          f"set SAVE_PATH and look at the file")
    return False


def frame(src):
    """(along, across) in the principal-axis frame of the shot positions."""
    c = src - src.mean(axis=0)
    u = np.linalg.svd(c, full_matrices=False)[2][0]
    if (c @ u)[-1] < (c @ u)[0]:          # point it the way the survey ran
        u = -u
    return c @ u, c @ np.array([-u[1], u[0]]), u


def sorted_order():
    """The shots put in order along the line, and where each one sits.

    Returns a dict with, in sorted order:

        new_idx    position in this ordering, 0 first
        orig_idx   the shot's index in the arrays as they are stored
        t          0 at the first sorted shot, 1 at the last, linear in
                   distance along the line
        perp_m     signed perpendicular distance from the fitted line
        along_m    distance along the line from its centroid
        xy         easting and northing

    The line is the principal axis of the shot positions through their
    centroid.  A straight line is the right thing to fit: a degree-5 polynomial
    reduces the residual by only 4 %, and the arc length along that polynomial
    matches the straight projection to 0.02 %, so the survey is a line the boat
    wandered about rather than a curve it followed.
    """
    src = np.load(resolve(C.META["das_raw"]))["source_xy"]
    centroid = src.mean(axis=0)
    along, perp, u = frame(src)
    order = np.argsort(along)
    a = along[order]
    span = a[-1] - a[0]
    return dict(order=order, centroid=centroid, direction=u,
                new_idx=np.arange(len(order)), orig_idx=order,
                t=(a - a[0]) / span if span > 0 else np.zeros_like(a),
                perp_m=perp[order], along_m=a, xy=src[order], span=span)


def write_order(path, o):
    """The ordering as a text table, with the line it was measured against."""
    ux, uy = o["direction"]
    cx, cy = o["centroid"]
    gap = np.diff(o["along_m"])
    lines = [
        "# shot ordering along the survey line",
        f"# source: {C.META['das_raw']}",
        "#",
        "# The line is the principal axis of the 551 shot positions, taken",
        "# through their centroid.  Straight, not curved: fitting a degree-5",
        "# polynomial instead reduces the residual by 4 %, and its arc length",
        "# matches the straight projection to 0.02 %.",
        "#",
        f"# centroid      {cx:.3f} {cy:.3f}",
        f"# direction     {ux:.9f} {uy:.9f}   (unit vector)",
        f"# normal        {-uy:.9f} {ux:.9f}",
        "#",
        "#   along_m = (xy - centroid) . direction",
        "#   perp_m  = (xy - centroid) . normal",
        "#",
        f"# {len(o['order'])} shots, along {o['along_m'][0]:.1f} to "
        f"{o['along_m'][-1]:.1f} m, span {o['span']:.1f} m",
        f"# gaps after sorting: median {np.median(gap):.2f} m, "
        f"p90 {np.percentile(gap, 90):.2f} m, max {gap.max():.2f} m",
        f"# perpendicular distance: rms {np.sqrt((o['perp_m'] ** 2).mean()):.2f} m,"
        f" max |{np.abs(o['perp_m']).max():.2f}| m",
        "#",
        "# new_idx  orig_idx            t        perp_m       along_m"
        "          easting         northing",
    ]
    for i in range(len(o["order"])):
        lines.append(
            f"{o['new_idx'][i]:9d} {o['orig_idx'][i]:9d} "
            f"{o['t'][i]:12.9f} {o['perp_m'][i]:13.4f} "
            f"{o['along_m'][i]:13.4f} {o['xy'][i, 0]:16.4f} "
            f"{o['xy'][i, 1]:16.4f}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def passes(along):
    """Index ranges between reversals of the along-line direction."""
    turn = np.flatnonzero(np.diff(np.sign(np.diff(along))) != 0) + 1
    edges = np.concatenate([[0], turn + 1, [len(along)]])
    return [(a, b) for a, b in zip(edges[:-1], edges[1:])
            if b - a >= MIN_PASS_SHOTS]


def main():
    show = show_figures()
    if not show and not SAVE_PATH:
        raise SystemExit("nothing to do: SHOW is off (or has no backend) and "
                         "SAVE_PATH is not set")

    meta = resolve(C.META["das_raw"])
    if not os.path.isfile(meta):
        raise SystemExit(f"not found: {meta}")
    src = np.load(meta)["source_xy"]
    along, across, u = frame(src)
    segs = passes(along)

    print(f"{C.META['das_raw']}: {len(src)} shots")
    print(f"  principal axis {u[0]:+.4f}, {u[1]:+.4f}")
    print(f"  along {along.min():.0f}..{along.max():.0f} m "
          f"({along.max() - along.min():.0f} m), "
          f"across {across.min():.0f}..{across.max():.0f} m "
          f"({across.max() - across.min():.0f} m)")
    print(f"  {len(segs)} passes of {MIN_PASS_SHOTS}+ shots, "
          f"{sum(b - a for a, b in segs)} of {len(src)} shots in them")

    fits = {}
    print(f"\n    {'degree':>7} {'residual std [m]':>18} {'max |resid|':>13} "
          f"{'fit swings [m]':>16}")
    for deg in DEGREES:
        p = np.polyfit(along, across, deg)
        r = across - np.polyval(p, along)
        fits[deg] = (p, r)
        curve = np.polyval(p, np.sort(along))
        print(f"    {deg:>7} {r.std():>18.2f} {np.abs(r).max():>13.2f} "
              f"{curve.max() - curve.min():>16.1f}")
    print(f"  a curve buys {1 - fits[max(DEGREES)][1].std() / fits[1][1].std():.1%}"
          f" over a straight line, so the scatter is the boat, not the bend")

    resid = fits[RESIDUAL_DEGREE][1]
    print(f"\n    {'pass':>5} {'shots':>7} {'along [m]':>20} "
          f"{'residual from deg ' + str(RESIDUAL_DEGREE) + ' [m]':>30}")
    for i, (a, b) in enumerate(segs, 1):
        r = resid[a:b]
        print(f"    {i:>5} {b - a:>7} {along[a:b].min():>9.0f}.."
              f"{along[a:b].max():<9.0f} {r.mean():>14.2f} +-{r.std():>6.2f} "
              f"(max |{np.abs(r).max():.1f}|)")

    fig, axes = plt.subplots(3, 1, figsize=FIGSIZE,
                             height_ratios=[2.0, 2.2, 1.6])
    colours = plt.get_cmap("tab10")
    grid = np.linspace(along.min(), along.max(), 800)

    ax = axes[0]
    for i, (a, b) in enumerate(segs):
        ax.plot(src[a:b, 0], src[a:b, 1], ".-", ms=3, lw=0.6,
                color=colours(i), label=f"pass {i + 1} ({b - a} shots)")
    ax.set_aspect("equal")
    ax.set_xlabel("easting [m]")
    ax.set_ylabel("northing [m]")
    ax.set_title("map view, true aspect - the four passes are one line",
                 fontsize=10)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    ax = axes[1]
    for i, (a, b) in enumerate(segs):
        ax.plot(along[a:b], across[a:b], ".", ms=4, color=colours(i),
                alpha=0.7, label=f"pass {i + 1}")
    for j, deg in enumerate(DEGREES):
        p, r = fits[deg]
        ax.plot(grid, np.polyval(p, grid), lw=1.6,
                color=plt.get_cmap("viridis")(j / max(len(DEGREES) - 1, 1)),
                label=f"degree {deg}  (resid {r.std():.2f} m)")
    span = across.max() - across.min()
    ax.set_ylim(across.min() - 0.1 * span, across.max() + 0.1 * span)
    ax.set_xlabel("along the principal axis [m]")
    ax.set_ylabel("across [m]")
    ax.set_title(f"cross axis exaggerated about "
                 f"{(along.max() - along.min()) / span:.0f}x - the fits are "
                 f"almost the same line, which is the point", fontsize=10)
    ax.legend(fontsize=8, ncol=2, loc="best")
    ax.grid(alpha=0.3)

    ax = axes[2]
    for i, (a, b) in enumerate(segs):
        ax.plot(along[a:b], resid[a:b], ".", ms=4, color=colours(i),
                alpha=0.8, label=f"pass {i + 1}")
    ax.axhline(0.0, color="k", lw=0.8)
    for s in (-1, 1):
        ax.axhline(s * resid.std(), color="k", lw=0.6, ls="--", alpha=0.5)
    ax.set_xlabel("along the principal axis [m]")
    ax.set_ylabel(f"residual from\ndegree {RESIDUAL_DEGREE} [m]")
    ax.set_title(f"what the fit does not explain: {resid.std():.1f} m rms of "
                 f"boat wander, shared by all four passes", fontsize=10)
    ax.legend(fontsize=8, ncol=4, loc="best")
    ax.grid(alpha=0.3)

    fig.suptitle(f"{len(src)} shot positions, {C.META['das_raw']}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))

    if SAVE_ORDER_PATH:
        o = sorted_order()
        p = resolve(SAVE_ORDER_PATH)
        write_order(p, o)
        gap = np.diff(o["along_m"])
        print(f"\n  sorted along the line: gaps median {np.median(gap):.2f} m, "
              f"p90 {np.percentile(gap, 90):.2f}, max {gap.max():.2f}, "
              f"{int((gap > 8).sum())} over 8 m")
        print(f"  wrote {p}")

    if SAVE_PATH:
        p = resolve(SAVE_PATH)
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        fig.savefig(p, dpi=130)
        print(f"  wrote {p}")
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
