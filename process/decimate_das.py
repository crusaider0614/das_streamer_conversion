"""Group the DAS channels 3:1 over the whole fibre.

Reduces `das_data_raw.npy` (551 x 3500 x 3175 at 0.25 m) to
`das_data_geom.npy` (551 x 3500 x 1058 at 0.75 m) by averaging every three
adjacent channels.

Why a fixed stride, and why 0.25 m
----------------------------------
The channel positions in the SEG-Y headers were not surveyed: they were filled
in from the streamer node positions, dividing each node-to-node segment evenly
among the channels that fall in it.  That shows in the numbers - the apparent
channel spacing is constant inside every segment but jumps between segments
(0.2512 where a segment holds 12 channels, 0.2519 where it holds 13), and every
segment comes out 0.7 % longer than 0.25 m per channel would give.  Taking those
coordinates at face value suggests a fibre that wanders; it does not.

The real acquisition interval is a uniform 0.25 m, so channel n sits at 0.25 n
along the fibre and a fixed stride of three is exactly 0.75 m everywhere, with
no drift to correct.  Groups are therefore contiguous blocks of three - which
also makes the whole operation a reshape and a mean.

The average is unweighted.  Over three samples a 1-2-1 taper suppresses white
noise less (x1.63 against x1.73) and, its effective aperture being narrower,
also rejects less at the wavenumber the decimation folds about.

Correspondence with the streamer is reported, not enforced: the streamer nodes
sit 12 or 13 channels apart, so they do not all land on a 3-channel grid
whatever the offset is.  ANCHOR_CHANNEL lines the grid up with the first node;
the rest come out within one channel, 0.25 m, of a group centre.

Edit the settings block below, then run from the repository root:

    python -m process.decimate_das
"""

import os
import time

import numpy as np

from utils.data import create_memmap, get_project_root, npy_shape

# ---------------------------------------------------------------- settings --

DAS_NPY = os.path.join("data", "pohang_shore", "numpy", "das_data_raw.npy")
DAS_META = os.path.join("data", "pohang_shore", "numpy", "das_data_raw_meta.npz")
STR_META = os.path.join("data", "pohang_shore", "numpy", "str_data_raw_meta.npz")
OUT_PATH = os.path.join("data", "pohang_shore", "numpy", "das_data_geom.npy")

# Channels averaged per output trace, and the true acquisition interval.
STRIDE = 3
CHANNEL_INTERVAL_M = 0.25

# Channel the group grid is aligned to, so that group centres fall on
# ANCHOR_CHANNEL + k * STRIDE.  146 is the DAS channel at the first streamer
# node, which lines the grid up with the streamer where they overlap.
ANCHOR_CHANNEL = 146

# The streamer geometry the correspondence is reported against: how many
# traces the curvelet interpolation produced, and its upsampling factor.
N_STREAMER_TRACES = 93
UPSAMPLE = 4

# Shots per read.  Every channel is needed, so a chunk is
# CHUNK x 3500 x 3175 x 4 bytes.
CHUNK = 4

# ---------------------------------------------------------------------------


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(get_project_root(), path)


def group_grid(n_ch):
    """Contiguous STRIDE-wide groups aligned on ANCHOR_CHANNEL.

    Returns the first channel of the first group and the group centres.
    """
    half = STRIDE // 2
    first = (ANCHOR_CHANNEL - half) % STRIDE          # first channel used
    n_groups = (n_ch - first) // STRIDE
    centres = first + half + STRIDE * np.arange(n_groups)
    return first, centres


def streamer_channels(str_xy, das_xy):
    """DAS channel nearest each recorded streamer node, and each interpolated
    trace.  Nearest-neighbour on the header coordinates is safe here: they were
    built from the streamer nodes, so the registration between the two is exact
    even though the spacing is not."""
    node = np.array([int(np.argmin(np.hypot(*(das_xy - p).T))) for p in str_xy])
    t = np.arange(N_STREAMER_TRACES) / float(UPSAMPLE)
    trace = np.interp(t, np.arange(len(node)), node.astype(float))
    return node, trace


def main():
    das_path = resolve(DAS_NPY)
    for p in (das_path, resolve(DAS_META), resolve(STR_META)):
        if not os.path.isfile(p):
            raise SystemExit(f"not found: {p}")

    with np.load(resolve(DAS_META)) as m:
        das_xy = m["receiver_xy"]
    with np.load(resolve(STR_META)) as m:
        str_xy = m["receiver_xy"]

    data = np.load(das_path, mmap_mode="r")
    n_shots, n_samp, n_ch = data.shape
    print(f"{das_path}\n  {n_shots} shots x {n_samp} samples x {n_ch} channels "
          f"@ {CHANNEL_INTERVAL_M:g} m")

    first, centres = group_grid(n_ch)
    n_out = len(centres)
    last = first + STRIDE * n_out
    print(f"  {STRIDE}-channel groups anchored on channel {ANCHOR_CHANNEL}: "
          f"{n_out} traces @ {STRIDE * CHANNEL_INTERVAL_M:g} m")
    print(f"  uses channels {first}..{last - 1} of 0..{n_ch - 1} "
          f"({first} before, {n_ch - last} after, unused)")
    print(f"  fibre covered {n_out * STRIDE * CHANNEL_INTERVAL_M:.2f} m")

    node, trace = streamer_channels(str_xy, das_xy)
    node_group = (node - first) // STRIDE
    node_offset = node - centres[np.clip(node_group, 0, n_out - 1)]
    trace_group = np.clip(np.round((trace - centres[0]) / STRIDE).astype(int),
                          0, n_out - 1)
    trace_offset = trace - centres[trace_group]

    print(f"\n  streamer correspondence")
    print(f"    24 recorded nodes -> DAS channels {node[0]}..{node[-1]}, "
          f"groups {node_group[0]}..{node_group[-1]}")
    vals, cnt = np.unique(np.abs(node_offset), return_counts=True)
    print(f"    node offset from its group centre: "
          + ", ".join(f"{v} chan x{c}" for v, c in zip(vals, cnt))
          + f"  (max {np.abs(node_offset).max() * CHANNEL_INTERVAL_M:.2f} m)")
    print(f"    {N_STREAMER_TRACES} interpolated traces -> groups "
          f"{trace_group[0]}..{trace_group[-1]}, "
          f"offset max {np.abs(trace_offset).max() * CHANNEL_INTERVAL_M:.3f} m")

    out_path = resolve(OUT_PATH)
    if os.path.isfile(out_path):
        prev = npy_shape(out_path)
        print(f"\n  overwriting {out_path} (was {prev})")
        del prev
    out = create_memmap(out_path, (n_shots, n_samp, n_out))
    print(f"  output {out.shape}, {out.nbytes / 1e9:.2f} GB")

    t0 = time.time()
    for start in range(0, n_shots, CHUNK):
        stop = min(start + CHUNK, n_shots)
        slab = np.asarray(data[start:stop, :, first:last], dtype=np.float32)
        out[start:stop] = slab.reshape(stop - start, n_samp, n_out,
                                       STRIDE).mean(axis=3)
        el = time.time() - t0
        print(f"    {stop}/{n_shots} shots  {el:.0f}s elapsed, "
              f"eta {el / stop * (n_shots - stop):.0f}s")
    out.flush()

    print(f"\n  wrote {out_path}")
    print(f"  amplitude range [{out[0].min():.4g}, {out[0].max():.4g}] (shot 0)")

    meta_path = os.path.splitext(out_path)[0] + "_meta.npz"
    np.savez(
        meta_path,
        centre_channel=centres,
        channels=centres[:, None] + np.arange(-(STRIDE // 2),
                                              STRIDE - STRIDE // 2)[None, :],
        arc_length_m=centres * CHANNEL_INTERVAL_M,
        centre_xy=das_xy[centres],
        stride=STRIDE,
        channel_interval_m=CHANNEL_INTERVAL_M,
        # where the streamer sits in this grid
        streamer_node_channel=node,
        streamer_node_group=node_group,
        streamer_node_offset_chan=node_offset,
        streamer_trace_channel=trace,
        streamer_trace_group=trace_group,
        streamer_trace_offset_chan=trace_offset,
    )
    print(f"  wrote {meta_path}")


if __name__ == "__main__":
    main()
