"""Shot and receiver positions of the translated DAS files, along the line.

    das_data_fake_str*.npy  (551 shots, 2000 samples, 264 receivers)
        ->  export/fake_str_geometry.txt

Every position is given three ways: the along-line coordinate and its 2-D
grid index; the projected point back in the survey's own map coordinates
(x_proj, y_proj - easting and northing, the SEG-Y frame); and the position
before projection (x_orig, y_orig).  The first two describe the same point,
the third is where it was recorded.

The survey is a line the boat wandered about - a degree-5 polynomial fit
reduces the residual by only 4 % and its arc length matches the straight
projection to 0.02 % - so a 2-D section along that line is a fair description
of it.  Everything here is the perpendicular projection onto the principal
axis of the shot positions, the same frame process/line_static.py moved the
data onto, so the traces and these coordinates agree with each other.

The shot axis is in SORTED order
--------------------------------
process/logenv_process.py sorts the shots along the line before writing the
receiver-major arrays, and inference/translate_das.py keeps that order, so
shot 0 of the output is the southernmost shot rather than the first one
recorded.  The table below is written in that same order: row i is index i of
the array's shot axis.  `orig` carries the recording index for anything that
still needs it.

What is measured and what is assumed
------------------------------------
Measured: every along-line coordinate here, from the SEG-Y source positions
and the decimated receiver positions.

Assumed: the two depths.  87 m of water comes from the near-zero-offset
arrival on the seafloor DAS, 58 ms at 1500 m/s, and the seabed is taken to be
flat at that depth - there is no bathymetry in this dataset to say otherwise.
The source depth is not recorded anywhere; SOURCE_DEPTH_M is 0 and should be
set to the gun depth if it is known.

RECEIVER_POSITIONS
------------------
"header" projects the receiver coordinates as they are.  They give an along-
line span of 804.12 m for 263 intervals, a mean of 3.058 m where the
acquisition interval is 3.00 m exactly, and they wander 8.8 m rms off the
line.  config.py says why: those coordinates were never surveyed, they were
filled into the headers from the streamer node positions, so the scatter is
bookkeeping rather than geometry.

"uniform" lays the 264 receivers at exactly DAS_STRIDE x
DAS_CHANNEL_INTERVAL_M x 12 = 3.00 m, centred on the same midpoint.  That is
the interval the fibre actually has, and it is the default.  It runs in the
direction the header positions do - receiver index northward, which is down
the along-line axis.  Against the header positions the difference is 3.6 m
rms, -4.3 m at receiver 0 and 10.9 m at receiver 263.

Edit the settings block, then run from the repository root:

    python -m inference.export_line_geometry
"""

import os

import numpy as np

import config as C
import process.shot_geometry as G
from utils.data import get_project_root

# ---------------------------------------------------------------- settings --

OUT_TXT = os.path.join("export", "fake_str_geometry.txt")

# Model grid.  1.0 m gives about five nodes per wavelength at the 320 Hz
# upper corner in 1450 m/s sediment, which is the shortest wavelength the
# band-pass leaves.
DX_M = 1.0
DZ_M = 1.0

# Where ix = 0 sits.  "min" puts it at the smallest along-line coordinate of
# anything in the survey - the first shot - so every index is non-negative.
# A number puts it at that along-line coordinate instead.
ORIGIN = "min"

# Depths, in metres below the sea surface.  See the header.
SOURCE_DEPTH_M = 0.0
RECEIVER_DEPTH_M = 87.0

# "uniform" or "header"; see the header.
RECEIVER_POSITIONS = "uniform"

# ---------------------------------------------------------------------------


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(get_project_root(), path)


def line_frame():
    """The projection frame: the streamer sidecar carries the fitted line."""
    s = np.load(resolve(C.META["str_line"]))
    return np.asarray(s["line_centroid"]), np.asarray(s["line_direction"])


def positions():
    """Along-line metres for the receivers and the sorted shots, plus the
    unprojected map positions of both, shots in the same sorted order."""
    c, u = line_frame()
    z = np.load(resolve(C.META["das_deci"]))
    rec_xy = np.asarray(z["receiver_xy"], dtype=np.float64)
    a_rec = (rec_xy - c) @ u

    if RECEIVER_POSITIONS == "uniform":
        n = len(a_rec)
        dx = C.DAS_CHANNEL_INTERVAL_M * int(z["stride"])
        # In the direction the header positions run.  The receiver index
        # increases northward and the along-line axis southward, so laying
        # the index out along +along put receiver 0 at the far end - every
        # receiver mirrored about the midpoint, up to 789 m off.
        step = np.sign(np.polyfit(np.arange(n), a_rec, 1)[0])
        a_rec = (np.arange(n) - (n - 1) / 2.0) * dx * step + a_rec.mean()
    elif RECEIVER_POSITIONS != "header":
        raise SystemExit(f"RECEIVER_POSITIONS {RECEIVER_POSITIONS!r} is not "
                         f"'uniform' or 'header'")

    order = np.asarray(G.sorted_order()["orig_idx"], dtype=int)
    a_src = ((z["source_xy_projected"] - c) @ u)[order]
    # the recorded source positions; deci only carries the projected ones
    src_xy = np.asarray(np.load(resolve(C.META["das_line"]))["source_xy"],
                        dtype=np.float64)[order]
    return a_rec, a_src, order, rec_xy, src_xy


def on_line(a):
    """Along-line metres -> the projected point in map coordinates."""
    c, u = line_frame()
    return np.asarray(c)[None, :] + np.asarray(a)[:, None] * np.asarray(u)


def main():
    a_rec, a_src, order, rec_xy, src_xy = positions()
    rec_p, src_p = on_line(a_rec), on_line(a_src)
    lo = min(a_rec.min(), a_src.min()) if ORIGIN == "min" else float(ORIGIN)
    ix = lambda a: (a - lo) / DX_M

    hi = max(a_rec.max(), a_src.max())
    nx = int(np.ceil((hi - lo) / DX_M)) + 1
    print(f"line frame from {C.META['str_line']}")
    print(f"  receivers {len(a_rec)} ({RECEIVER_POSITIONS}): "
          f"{a_rec.min():.2f} .. {a_rec.max():.2f} m, "
          f"interval {np.diff(a_rec).mean():.4f} m")
    print(f"  shots     {len(a_src)} (sorted): "
          f"{a_src.min():.2f} .. {a_src.max():.2f} m, "
          f"median interval {np.median(np.diff(a_src)):.4f} m")
    print(f"  model     {hi - lo:.2f} m -> {nx} nodes at dx {DX_M} m, "
          f"ix 0 at {lo:.2f} m")
    print(f"  iz        source {SOURCE_DEPTH_M / DZ_M:.0f}, "
          f"receiver {RECEIVER_DEPTH_M / DZ_M:.0f} at dz {DZ_M} m")

    p = resolve(OUT_TXT)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        w = f.write
        w("# shot and receiver positions of das_data_fake_str*.npy\n")
        w("# projected onto the principal axis of the shot positions\n")
        w("# x_proj y_proj: the projected point in map coordinates\n")
        w("# x_orig y_orig: the recorded position, before projection\n")
        w("#\n")
        w(f"# line centroid   {line_frame()[0][0]:.3f} "
          f"{line_frame()[0][1]:.3f}\n")
        w(f"# line direction  {line_frame()[1][0]:.6f} "
          f"{line_frame()[1][1]:.6f}\n")
        w(f"# dx {DX_M} m, dz {DZ_M} m, ix 0 at along = {lo:.3f} m, "
          f"nx {nx}\n")
        w(f"# receiver positions: {RECEIVER_POSITIONS}\n")
        w(f"# source depth {SOURCE_DEPTH_M} m (assumed), "
          f"receiver depth {RECEIVER_DEPTH_M} m (87 m of water, flat seabed"
          f" assumed)\n")
        w("#\n")
        w("# the shot axis of the array is SORTED along the line; `orig` is\n")
        w("# the recording index of the same shot\n")
        w("#\n")
        w(f"# receivers: {len(a_rec)}\n")
        w("# index  along_m        ix      iz        x_proj         y_proj"
          "        x_orig         y_orig\n")
        for i, a in enumerate(a_rec):
            w(f"R {i:5d} {a:12.3f} {ix(a):10.3f} "
              f"{RECEIVER_DEPTH_M / DZ_M:7.3f} "
              f"{rec_p[i, 0]:14.3f} {rec_p[i, 1]:14.3f} "
              f"{rec_xy[i, 0]:14.3f} {rec_xy[i, 1]:14.3f}\n")
        w(f"#\n# shots: {len(a_src)}\n")
        w("# index   orig  along_m        ix      iz        x_proj"
          "         y_proj        x_orig         y_orig\n")
        for i, a in enumerate(a_src):
            w(f"S {i:5d} {order[i]:6d} {a:12.3f} {ix(a):10.3f} "
              f"{SOURCE_DEPTH_M / DZ_M:7.3f} "
              f"{src_p[i, 0]:14.3f} {src_p[i, 1]:14.3f} "
              f"{src_xy[i, 0]:14.3f} {src_xy[i, 1]:14.3f}\n")
    print(f"  wrote {OUT_TXT}  ({os.path.getsize(p) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
