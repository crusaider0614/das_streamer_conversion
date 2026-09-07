"""Direct-arrival mute: geometry, boundary times and taper weights.

Shared by the stages that need to apply it - process/band_limit.py,
process/decimate_intp.py, process/freq_filter.py - and by
process/mute_direct.py, which is the standalone preview.

The boundary is

    t(shot, trace) = sqrt(offset^2 + depth_m^2) / velocity_m_s - lead_ms

with the parameters in config.MUTE_PARAMS.  Nothing can arrive before the
fastest path from the shot, so everything above that curve is noise; a velocity
faster than the medium puts the boundary earlier and errs towards muting less.

Every receiver set is read off the DAS grid.  The streamer's 24 recorded nodes
and its 93 interpolated positions are both defined as DAS group centres - that
is what process/decimate_das.py established - so taking them from there keeps
every stage of both domains on one set of coordinates, rather than two that
disagree by the 0.7 % scale error in the SEG-Y headers.
"""

import os

import numpy as np

import config as C
from utils.data import get_project_root


def _resolve(path):
    return path if os.path.isabs(path) else os.path.join(get_project_root(), path)


def geometry(n_traces, verbose=True):
    """(source_xy, receiver_xy) for an array with this many traces."""
    raw = np.load(_resolve(C.META["das_raw"]))
    geom = np.load(_resolve(C.META["das_geom"]))
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
    if verbose:
        print(f"  mute geometry: {label} ({len(rec)} receivers, "
              f"{len(raw['source_xy'])} shots)")
    return raw["source_xy"], rec


def boundary_ms(n_traces, params=None, verbose=True):
    """Mute boundary in milliseconds, shape (n_shots, n_traces)."""
    p = params or C.MUTE_PARAMS
    src, rec = geometry(n_traces, verbose=verbose)
    off = np.hypot(src[:, None, 0] - rec[None, :, 0],
                   src[:, None, 1] - rec[None, :, 1])
    t = np.sqrt(off ** 2 + p["depth_m"] ** 2) / p["velocity_m_s"] * 1e3 \
        - p["lead_ms"]
    if verbose:
        print(f"    v {p['velocity_m_s']:g} m/s, depth {p['depth_m']:g} m, "
              f"lead {p['lead_ms']:g} ms, taper {p['taper_ms']:g} ms")
        print(f"    offset {off.min():.0f}..{off.max():.0f} m  ->  "
              f"boundary {np.maximum(t, 0).min():.0f}.."
              f"{np.maximum(t, 0).max():.0f} ms")
    return np.maximum(t, 0.0), off


def weights(t_ms, nt, dt_ms, taper_ms=None):
    """(nt, n_traces) in [0, 1] for one shot's boundary row.

    Zero above the boundary, one below, raised cosine between:
    w = sin^2(pi x / 2) with x = (t - t_boundary) / taper_ms clipped to [0, 1].
    The slope is zero at both ends, so the ramp adds no kink of its own.

    Note the ramp sits on the kept side - the boundary is where the weight is
    zero and full amplitude is only reached taper_ms later.  With a lead that
    is smaller than the taper, the transition therefore finishes after the
    geometric time rather than before it.
    """
    taper = C.MUTE_PARAMS["taper_ms"] if taper_ms is None else taper_ms
    t = np.arange(nt)[:, None] * dt_ms
    if taper <= 0:
        return (t >= t_ms[None, :]).astype(np.float64)
    x = (t - t_ms[None, :]) / taper
    return np.clip(0.5 - 0.5 * np.cos(np.pi * np.clip(x, 0.0, 1.0)), 0.0, 1.0)


def summary(t_ms, nt, dt_ms):
    """One line describing how much of the record the mute takes."""
    span = nt * dt_ms
    frac = 100 * np.mean(np.clip(t_ms, 0, span)) / span
    past = 100 * (t_ms >= span).mean()
    return (f"    muting the first {frac:.1f} % of the record on average"
            + (f", {past:.1f} % of traces entirely" if past else ""))
