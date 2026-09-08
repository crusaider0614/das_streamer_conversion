"""Source and receiver coordinates for both domains, depth included.

    data/pohang_shore/{das,streamer}/data0{09..12}.segy   trace headers only
      ->  numpy/geometry.npz     arrays
      ->  numpy/geometry.txt     the same thing readable, with the caveats

Reads headers and no samples, so it costs seconds rather than the minutes the
`make_*_npy.py` scripts take.  Those two write `*_meta.npz` with x and y only:
process/make_streamer_npy.py and make_das_npy.py pull SourceX/Y, GroupX/Y and
the byte-71 scalar and stop there, so the elevation fields have never been
carried through.  They are populated.

What the headers hold
---------------------
    byte 41  ReceiverGroupElevation    populated, both domains
    byte 45  SourceSurfaceElevation    -100 constant, both domains
    byte 69  ElevationScalar           -100, so divide by 100
    byte 71  SourceGroupScalar         -100, so divide by 100

    byte 49  SourceDepth               zero
    byte 53  ReceiverDatumElevation    zero
    byte 57  SourceDatumElevation      zero
    byte 61  SourceWaterDepth          zero
    byte 65  GroupWaterDepth           zero

So there is one depth per receiver and one per source, and nothing that states
the water depth.  Byte 37 (offset) is zero on the streamer and, on the DAS,
holds 319..1113 where the source-receiver distance is 0.5..2350 m - it is not
offset, and this script ignores it.

Sign convention here: `z_m` is the header elevation, negative downwards, and
`depth_m` is -z_m, positive downwards.  Both are written so neither has to be
guessed at.

Two things to be careful of
---------------------------
The receiver elevations are static - identical in every record of every file -
and they vary along the cable rather than randomly, correlating with distance
from its start at -0.93 on the streamer and -0.96 on the DAS.  That is the
shape of a cable running from shallow water into deeper water, which fits a
shore survey.

But the numbers are small: 0.37-1.96 m on the streamer's 24 nodes and
0.29-7.74 m over the DAS's 793 m of fibre.  Earlier work in this project put
the water depth at about 87 m, inferred from the near-zero-offset arrival time
on the DAS rather than from any header.  Those two cannot both be right about
the same thing, and this script does not try to settle it - it carries the
header values through and prints them so the disagreement is visible.  Nothing
currently depends on the answer: config.MUTE_PARAMS uses depth_m = 0.

Edit the settings block below, then run from the repository root:

    python -m process.make_geometry
"""

import os

import numpy as np
import segyio
from segyio import TraceField as TF

import config as C
from utils.data import get_project_root

# ---------------------------------------------------------------- settings --

FILES = ["data009.segy", "data010.segy", "data011.segy", "data012.segy"]
DOMAINS = {"das": "das", "str": "streamer"}

OUT_NPZ = os.path.join(C.NPY_DIR, "geometry.npz")
OUT_TXT = os.path.join(C.NPY_DIR, "geometry.txt")

# The channel grouping the DAS array was built with, so the group centres here
# match das_data_geom.npy trace for trace.
GROUP_META = C.META["das_geom"]

# ---------------------------------------------------------------------------

KEYS = dict(record=TF.FieldRecord, channel=TF.TraceNumber,
            cscal=TF.SourceGroupScalar, escal=TF.ElevationScalar,
            sx=TF.SourceX, sy=TF.SourceY, sz=TF.SourceSurfaceElevation,
            gx=TF.GroupX, gy=TF.GroupY, gz=TF.ReceiverGroupElevation)


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(get_project_root(), path)


def factor(scalar):
    """The SEG-Y scalar as a multiplier: negative divides, positive multiplies."""
    s = float(scalar)
    if s < 0:
        return 1.0 / abs(s)
    return abs(s) if s > 0 else 1.0


def read_file(path):
    """Per-record source xyz and per-channel receiver xyz from one file.

    Traces are sorted by (FieldRecord, TraceNumber), the same ordering
    make_streamer_npy.py and make_das_npy.py use, so record k here is shot k
    of the arrays they wrote.
    """
    with segyio.open(path, "r", strict=False, ignore_geometry=True) as f:
        h = {k: np.asarray(f.attributes(v)[:], dtype=np.int64)
             for k, v in KEYS.items()}

    records, channels = np.unique(h["record"]), np.unique(h["channel"])
    n_rec, n_ch = len(records), len(channels)
    if n_rec * n_ch != len(h["record"]):
        raise SystemExit(f"{os.path.basename(path)}: {len(h['record'])} traces "
                         f"do not factor into {n_rec} x {n_ch}")
    order = np.lexsort((h["channel"], h["record"]))

    cf, ef = factor(h["cscal"][0]), factor(h["escal"][0])

    def grid(key, f):
        return h[key][order].reshape(n_rec, n_ch).astype(np.float64) * f

    src = np.stack([grid("sx", cf)[:, 0], grid("sy", cf)[:, 0],
                    grid("sz", ef)[:, 0]], axis=1)
    rz = grid("gz", ef)
    rec = np.stack([grid("gx", cf)[0], grid("gy", cf)[0], rz[0]], axis=1)
    # the receivers are meant to be static; say so if they are not
    drift = float(np.abs(rz - rz[0]).max())
    return src, rec, records, cf, ef, drift


def collect(domain, sub):
    """One domain's four files, concatenated in file order."""
    src, recs, records, drifts = [], [], [], []
    cf = ef = None
    for name in FILES:
        p = os.path.join(get_project_root(), C.DATA_DIR, sub, name)
        if not os.path.isfile(p):
            raise SystemExit(f"not found: {p}")
        s, r, rc, cf, ef, drift = read_file(p)
        src.append(s)
        recs.append(r)
        records.append(rc)
        drifts.append(drift)
        print(f"    {name}: {len(s)} shots, {len(r)} channels, "
              f"receiver depth drift within the file {drift:.4f} m")
    rec = recs[0]
    spread = max(float(np.abs(r - rec).max()) for r in recs)
    print(f"    receivers agree across the four files to {spread:.4f} m")
    return (np.concatenate(src), rec, np.concatenate(records),
            cf, ef, max(drifts), spread)


def main():
    out = {}
    print("reading trace headers only\n")
    info = {}
    for tag, sub in DOMAINS.items():
        print(f"  {tag}  ({sub}/)")
        src, rec, records, cf, ef, drift, spread = collect(tag, sub)
        out[f"{tag}_source_xyz"] = src
        out[f"{tag}_receiver_xyz"] = rec
        out[f"{tag}_record"] = records
        info[tag] = dict(cf=cf, ef=ef, drift=drift, spread=spread,
                         n_shots=len(src), n_ch=len(rec))
        print(f"    coordinate scalar x{cf:g}, elevation scalar x{ef:g}\n")

    das_src = out["das_source_xyz"]
    str_src = out["str_source_xyz"]
    if len(das_src) != len(str_src):
        raise SystemExit(f"shot counts differ: das {len(das_src)}, "
                         f"str {len(str_src)}")
    dsrc = float(np.abs(das_src - str_src).max())
    print(f"  the two domains' source positions agree to {dsrc:.4f} m")

    # DAS group centres, on the same grouping das_data_geom.npy was built with
    g = np.load(resolve(GROUP_META))
    chans = g["channels"]                       # (1058, 3) channel numbers
    das_rec = out["das_receiver_xyz"]
    idx = chans - chans.min()                   # channel number -> row
    if idx.max() >= len(das_rec):
        raise SystemExit(f"group channels reach {chans.max()} but only "
                         f"{len(das_rec)} channels were read")
    out["das_group_xyz"] = das_rec[idx].mean(axis=1)
    out["das_group_arc_m"] = g["arc_length_m"]
    out["streamer_node_group"] = g["streamer_node_group"]
    out["streamer_trace_group"] = g["streamer_trace_group"]
    out["str_node_xyz"] = out["das_group_xyz"][g["streamer_node_group"]]
    out["str_trace_xyz"] = out["das_group_xyz"][g["streamer_trace_group"]]

    # consistency: do the streamer's own 24 receiver headers agree with the DAS
    # group centres the pipeline treats as the same points?
    own = out["str_receiver_xyz"]
    grp = out["str_node_xyz"]
    dxy = float(np.hypot(*(own[:, :2] - grp[:, :2]).T).max())
    dz = float(np.abs(own[:, 2] - grp[:, 2]).max())
    print(f"  streamer node headers vs the DAS group centres the pipeline "
          f"uses:\n    horizontal max {dxy:.3f} m, depth max {dz:.3f} m")

    lines = [
        "# source and receiver geometry, depth included",
        f"# from {', '.join(FILES)} in {list(DOMAINS.values())}",
        "#",
        "# z_m is the header elevation, negative downwards.",
        "# depth_m is -z_m, positive downwards.",
        "#",
        "# Populated headers: ReceiverGroupElevation (41),",
        "# SourceSurfaceElevation (45), ElevationScalar (69),",
        "# SourceGroupScalar (71).  SourceDepth, the datum elevations and both",
        "# water depths are all zero, so no header states the water depth.",
        "# Byte 37 is not offset on this data - see the module docstring.",
        "#",
    ]
    for tag in DOMAINS:
        i = info[tag]
        src = out[f"{tag}_source_xyz"]
        rec = out[f"{tag}_receiver_xyz"]
        lines += [
            f"# {tag}: {i['n_shots']} shots, {i['n_ch']} channels",
            f"#   scalars: coordinate x{i['cf']:g}, elevation x{i['ef']:g}",
            f"#   source   z {src[:, 2].min():+.3f}..{src[:, 2].max():+.3f} m"
            f"  (depth {-src[:, 2].max():.3f}..{-src[:, 2].min():.3f} m)",
            f"#   receiver z {rec[:, 2].min():+.3f}..{rec[:, 2].max():+.3f} m"
            f"  (depth {-rec[:, 2].max():.3f}..{-rec[:, 2].min():.3f} m),"
            f" {len(np.unique(rec[:, 2]))} distinct",
            f"#   receiver drift within a file {i['drift']:.4f} m,"
            f" across files {i['spread']:.4f} m",
            "#",
        ]
    arc = out["das_group_arc_m"]
    gz = out["das_group_xyz"][:, 2]
    lines += [
        f"# das group centres: {len(gz)} groups over "
        f"{arc.max() - arc.min():.2f} m of fibre,"
        f" z {gz.min():+.3f}..{gz.max():+.3f} m",
        f"#   correlation of depth with distance along the fibre "
        f"{np.corrcoef(arc, gz)[0, 1]:+.4f}",
        f"# streamer node headers vs das group centres: horizontal "
        f"{dxy:.3f} m, depth {dz:.3f} m at worst",
        f"# the two domains' source positions agree to {dsrc:.4f} m",
        "#",
        "# arrays in geometry.npz:",
    ]
    for k in sorted(out):
        lines.append(f"#   {k:<24} {np.shape(out[k])}")
    lines += [
        "#",
        "# streamer nodes, from the streamer's own headers",
        "# node        easting        northing        z_m    depth_m",
    ]
    for j, r in enumerate(out["str_receiver_xyz"]):
        lines.append(f"{j:6d} {r[0]:14.3f} {r[1]:15.3f} {r[2]:10.3f} "
                     f"{-r[2]:10.3f}")
    lines += [
        "#",
        "# sources, in shot order",
        "# shot        easting        northing        z_m    depth_m",
    ]
    for j, s in enumerate(out["str_source_xyz"]):
        lines.append(f"{j:6d} {s[0]:14.3f} {s[1]:15.3f} {s[2]:10.3f} "
                     f"{-s[2]:10.3f}")

    npz, txt = resolve(OUT_NPZ), resolve(OUT_TXT)
    os.makedirs(os.path.dirname(os.path.abspath(npz)), exist_ok=True)
    np.savez(npz, **out)
    with open(txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  wrote {npz}")
    print(f"  wrote {txt}")


if __name__ == "__main__":
    main()
