"""Build the streamer array from the raw SEG-Y shot gathers.

Reads the four files, sorts the traces into shot gathers and concatenates them
into one (n_shots, n_samples, n_channels) array.  As configured it keeps the
SEG-Y's own time grid, giving (551, 8000, 24) at 0.5 ms - the input to the
curvelet trace interpolation, which needs the full time bandwidth to constrain
the steep events it has to reconstruct.  Resampling onto the DAS grid
(1000 us x 3500 samples) comes after interpolation; set TARGET_DT_US and
TARGET_SAMPLES to do it here instead.

When TARGET_DT_US is a multiple of the file's own sample interval the time axis
is decimated with an anti-alias filter ahead of it: `scipy.signal.resample_poly`
designs a Kaiser-windowed FIR for the ratio, applies it with the group delay
compensated (linear phase, no waveform distortion), and drops samples in one
pass.  When it equals the file's interval nothing is filtered or dropped.

Sample *values* are otherwise untouched - no bandpass, no envelope scaling, no
normalisation.  Those belong to the later stages
(`process/logenv_process.py` and whatever produces `str_data_bl.npy`).

Edit the settings block below, then run from the repository root:

    python -m process.make_streamer_npy
"""

import os

import numpy as np
import segyio
from scipy.signal import resample_poly
from segyio import TraceField as TF

from utils.data import get_project_root

# ---------------------------------------------------------------- settings --

# Directory holding the streamer SEG-Y files, and the files to concatenate.
# Shots end up in this order, so keep it consistent with the DAS array.
SEGY_DIR = os.path.join(get_project_root(), "data", "pohang_shore", "streamer")
FILES = ["data009.segy", "data010.segy", "data011.segy", "data012.segy"]

# Where the array is written.
OUT_PATH = os.path.join(get_project_root(), "data", "pohang_shore", "numpy",
                        "str_data_raw.npy")

# Target time grid.  Only integer decimation is supported: TARGET_DT_US must
# be a whole multiple of the file's own sample interval, and setting it equal
# to that interval skips the resampling entirely.
#
# Currently set to the SEG-Y's own grid (500 us x 8000 samples), because the
# curvelet trace interpolation runs on the raw data - resampling time first
# would throw away the high frequencies that constrain the steep events the
# interpolation has to reconstruct.  Move to the DAS grid (1000 us x 3500)
# after interpolating.
TARGET_DT_US = 500
TARGET_SAMPLES = 8000

# Also write <OUT_PATH stem>_meta.npz with the per-shot source coordinates,
# the receiver coordinates and the FieldRecord numbers, for matching shots
# against the DAS array or filtering by offset later.
WRITE_META = True

# ---------------------------------------------------------------------------


def sample_interval_us(f):
    """Sample interval in microseconds, from the binary header with the trace
    header as a fallback."""
    dt = int(f.bin[segyio.BinField.Interval])
    if dt <= 0:
        dt = int(f.header[0][TF.TRACE_SAMPLE_INTERVAL])
    if dt <= 0:
        raise ValueError("no usable sample interval in the file headers")
    return dt


def read_gathers(path):
    """Read one SEG-Y into (n_records, n_samples, n_channels).

    Traces are sorted by (FieldRecord, TraceNumber) rather than trusting the
    on-disk order, and every record is required to hold the same channel set.
    """
    with segyio.open(path, "r", strict=False, ignore_geometry=True) as f:
        dt_us = sample_interval_us(f)
        n_samples = len(f.samples)
        data = f.trace.raw[:]                                    # (n_traces, n_samples)
        record = np.asarray(f.attributes(TF.FieldRecord)[:], dtype=np.int64)
        channel = np.asarray(f.attributes(TF.TraceNumber)[:], dtype=np.int64)
        scalar = np.asarray(f.attributes(TF.SourceGroupScalar)[:], dtype=np.int64)
        sx = np.asarray(f.attributes(TF.SourceX)[:], dtype=np.int64)
        sy = np.asarray(f.attributes(TF.SourceY)[:], dtype=np.int64)
        gx = np.asarray(f.attributes(TF.GroupX)[:], dtype=np.int64)
        gy = np.asarray(f.attributes(TF.GroupY)[:], dtype=np.int64)

    records = np.unique(record)
    channels = np.unique(channel)
    n_rec, n_ch = len(records), len(channels)

    if n_rec * n_ch != len(record):
        raise ValueError(
            f"{os.path.basename(path)}: {len(record)} traces do not factor into "
            f"{n_rec} records x {n_ch} channels - the gathers are not regular"
        )

    order = np.lexsort((channel, record))
    if not np.array_equal(channel[order].reshape(n_rec, n_ch),
                          np.tile(channels, (n_rec, 1))):
        raise ValueError(
            f"{os.path.basename(path)}: records do not all carry the same "
            f"channel numbers"
        )

    gathers = data[order].reshape(n_rec, n_ch, n_samples)
    gathers = np.ascontiguousarray(gathers.transpose(0, 2, 1))   # -> (rec, t, ch)

    meta = {
        "record": records,
        "source_xy": scaled_xy(sx, sy, scalar, order, n_rec, n_ch)[:, 0, :],
        "receiver_xy": scaled_xy(gx, gy, scalar, order, n_rec, n_ch)[0, :, :],
    }
    return gathers, dt_us, meta


def scaled_xy(x, y, scalar, order, n_rec, n_ch):
    """(n_records, n_channels, 2) coordinates in metres, SEG-Y byte-71 scalar
    applied: positive multiplies, negative divides."""
    x = x[order].astype(np.float64)
    y = y[order].astype(np.float64)
    s = scalar[order].astype(np.float64)
    for a in (x, y):
        neg, pos = s < 0, s > 1
        a[neg] /= -s[neg]
        a[pos] *= s[pos]
    return np.stack([x, y], axis=1).reshape(n_rec, n_ch, 2)


def resample_to(gathers, dt_us, target_dt_us):
    """Anti-alias filter and decimate along the time axis.

    resample_poly builds the FIR for the exact up/down ratio, so its cutoff
    lands at the new Nyquist (500 Hz for 1 ms), and it compensates the filter
    delay - the output stays zero-phase.  padtype="line" removes the linear
    trend before padding, which keeps the filter from ringing at the record
    ends.
    """
    if target_dt_us == dt_us:
        return gathers.astype(np.float32, copy=False)
    if target_dt_us % dt_us:
        raise ValueError(
            f"{dt_us} us -> {target_dt_us} us is not an integer ratio; this "
            f"script only handles integer decimation"
        )
    q = target_dt_us // dt_us
    out = resample_poly(gathers, up=1, down=q, axis=1,
                        window=("kaiser", 5.0), padtype="line")
    return out.astype(np.float32, copy=False)


def build(paths):
    blocks, metas = [], []
    for path in paths:
        gathers, dt_us, meta = read_gathers(path)
        n_rec, n_t, n_ch = gathers.shape
        print(f"  {os.path.basename(path)}: {n_rec} shots x {n_t} samples "
              f"@ {dt_us / 1000:g} ms x {n_ch} channels")

        gathers = resample_to(gathers, dt_us, TARGET_DT_US)
        if gathers.shape[1] < TARGET_SAMPLES:
            raise ValueError(
                f"{os.path.basename(path)}: only {gathers.shape[1]} samples "
                f"after resampling, need {TARGET_SAMPLES}"
            )
        dropped = gathers.shape[1] - TARGET_SAMPLES
        gathers = gathers[:, :TARGET_SAMPLES]

        steps = []
        if TARGET_DT_US != dt_us:
            steps.append(f"anti-alias filtered and decimated {TARGET_DT_US // dt_us}x")
        if dropped:
            steps.append(f"cropped {dropped} samples off the tail")
        note = ", ".join(steps) if steps else "time axis kept as recorded"
        print(f"      -> {gathers.shape[0]} x {gathers.shape[1]} "
              f"@ {TARGET_DT_US / 1000:g} ms x {gathers.shape[2]}  ({note})")

        blocks.append(gathers)
        metas.append(meta)

    widths = {b.shape[2] for b in blocks}
    if len(widths) > 1:
        raise ValueError(f"files disagree on the channel count: {sorted(widths)}")

    return np.concatenate(blocks, axis=0), metas


def write_meta(paths, metas):
    meta_path = os.path.splitext(OUT_PATH)[0] + "_meta.npz"
    np.savez(
        meta_path,
        source_xy=np.concatenate([m["source_xy"] for m in metas], axis=0),
        receiver_xy=metas[0]["receiver_xy"],
        record=np.concatenate([m["record"] for m in metas], axis=0),
        file_index=np.concatenate([
            np.full(len(m["record"]), i, dtype=np.int32)
            for i, m in enumerate(metas)
        ]),
        file_names=np.array([os.path.basename(p) for p in paths]),
    )
    print(f"wrote {meta_path}")


def main():
    paths = [os.path.join(SEGY_DIR, n) for n in FILES]
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise SystemExit(f"missing: {[os.path.basename(p) for p in missing]}")

    print(f"reading {len(paths)} files from {SEGY_DIR}")
    data, metas = build(paths)

    os.makedirs(os.path.dirname(os.path.abspath(OUT_PATH)), exist_ok=True)
    np.save(OUT_PATH, data)
    print(f"\nwrote {OUT_PATH}")
    print(f"  shape {data.shape}, dtype {data.dtype}, "
          f"{data.nbytes / 1e6:.0f} MB")
    print(f"  amplitude range [{data.min():.4g}, {data.max():.4g}] "
          f"(values untouched apart from the anti-alias filter)")

    if WRITE_META:
        write_meta(paths, metas)


if __name__ == "__main__":
    main()
