"""Build the DAS array from the raw SEG-Y shot gathers.

The DAS counterpart of `process/make_streamer_npy.py`.  Reads the four files,
sorts the traces into shot gathers and concatenates them into
`das_data_raw.npy`, shape (551, 3500, 3175) - 3175 channels at 0.257 m over
809 m, already on the 1 ms / 3500 sample grid, so nothing is resampled.

That is 22.8 GB, which is simply how big the data are: the four SEG-Y files
come to 24.9 GB between them.  Two consequences shape the code:

  * gathers are read and written one record at a time into a memory-mapped
    output, never the whole file at once - `f.trace.raw[:]` on one of these
    would pull 6 GB into RAM;
  * CHANNELS can restrict the output to a channel range.  The streamer's 93
    interpolated positions map onto DAS channels 146-431, so `slice(140, 440)`
    covers the co-located aperture in 2.2 GB if the full fibre is not needed.

Sample values are untouched - no filtering, no scaling.  As with the streamer,
that belongs to the later stages.

Edit the settings block below, then run from the repository root:

    python -m process.make_das_npy
"""

import os
import time

import numpy as np
import segyio
from segyio import TraceField as TF

from utils.data import get_project_root

# ---------------------------------------------------------------- settings --

# Directory holding the DAS SEG-Y files, and the files to concatenate.  Keep
# the order the same as make_streamer_npy.py so shot i means the same shot in
# both arrays.
SEGY_DIR = os.path.join(get_project_root(), "data", "pohang_shore", "das")
FILES = ["data009.segy", "data010.segy", "data011.segy", "data012.segy"]

OUT_PATH = os.path.join(get_project_root(), "data", "pohang_shore", "numpy",
                        "das_data_raw.npy")

# Channels to keep, as a slice over the 3175 recorded ones.  None keeps them
# all (22.8 GB).  slice(140, 440) is the aperture the streamer covers (2.2 GB).
CHANNELS = None

# Also write <OUT_PATH stem>_meta.npz with the receiver and per-shot source
# coordinates and the FieldRecord numbers.
WRITE_META = True

# ---------------------------------------------------------------------------


def apply_coordinate_scalar(raw, scalar):
    """SEG-Y byte-71 scalar: positive multiplies, negative divides."""
    raw = np.asarray(raw, dtype=np.float64)
    scalar = np.asarray(scalar, dtype=np.float64)
    out = raw.copy()
    neg, pos = scalar < 0, scalar > 1
    out[neg] = raw[neg] / -scalar[neg]
    out[pos] = raw[pos] * scalar[pos]
    return out


def scan_headers(path):
    """Trace layout and geometry, without touching the sample data."""
    with segyio.open(path, "r", strict=False, ignore_geometry=True) as f:
        n_traces = f.tracecount
        n_samples = len(f.samples)
        dt_us = int(f.bin[segyio.BinField.Interval])
        record = np.asarray(f.attributes(TF.FieldRecord)[:], dtype=np.int64)
        channel = np.asarray(f.attributes(TF.TraceNumber)[:], dtype=np.int64)
        scalar = np.asarray(f.attributes(TF.SourceGroupScalar)[:], dtype=np.int64)
        gx = np.asarray(f.attributes(TF.GroupX)[:], dtype=np.int64)
        gy = np.asarray(f.attributes(TF.GroupY)[:], dtype=np.int64)
        sx = np.asarray(f.attributes(TF.SourceX)[:], dtype=np.int64)
        sy = np.asarray(f.attributes(TF.SourceY)[:], dtype=np.int64)

    records = np.unique(record)
    channels = np.unique(channel)
    n_rec, n_ch = len(records), len(channels)
    if n_rec * n_ch != n_traces:
        raise ValueError(
            f"{os.path.basename(path)}: {n_traces} traces do not factor into "
            f"{n_rec} records x {n_ch} channels")

    # Fast path: traces already laid out record-major with channels ascending,
    # so each record is one contiguous read.
    contiguous = (np.array_equal(record, np.repeat(records, n_ch))
                  and np.array_equal(channel, np.tile(channels, n_rec)))

    order = None if contiguous else np.lexsort((channel, record))

    return {
        "path": path, "n_traces": n_traces, "n_samples": n_samples,
        "dt_us": dt_us, "records": records, "channels": channels,
        "contiguous": contiguous, "order": order,
        "receiver_xy": np.stack([
            apply_coordinate_scalar(gx[:n_ch], scalar[:n_ch]),
            apply_coordinate_scalar(gy[:n_ch], scalar[:n_ch])], axis=1),
        "source_xy": np.stack([
            apply_coordinate_scalar(sx[::n_ch], scalar[::n_ch]),
            apply_coordinate_scalar(sy[::n_ch], scalar[::n_ch])], axis=1),
    }


def create_output(path, shape, attempts=5, pause=3.0):
    """open_memmap, retried.

    numpy writes the .npy header, closes the file and reopens it to map; on
    Windows another process can hold it open in between - a virus scanner
    picking up the new file, or the disk being busy - and the reopen fails with
    PermissionError even though the directory is writable.  Observed once here
    while another job was writing to the same disk, and it went away on retry.
    """
    for attempt in range(1, attempts + 1):
        try:
            return np.lib.format.open_memmap(path, mode="w+",
                                             dtype=np.float32, shape=shape)
        except PermissionError as exc:
            if attempt == attempts:
                raise
            print(f"  {type(exc).__name__} creating the output "
                  f"(attempt {attempt}/{attempts}), retrying in {pause:g}s")
            time.sleep(pause)


def main():
    paths = [os.path.join(SEGY_DIR, n) for n in FILES]
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise SystemExit(f"missing: {[os.path.basename(p) for p in missing]}")

    print(f"scanning headers of {len(paths)} files in {SEGY_DIR}")
    infos = []
    for p in paths:
        t0 = time.time()
        info = scan_headers(p)
        infos.append(info)
        print(f"  {os.path.basename(p)}: {len(info['records'])} records x "
              f"{len(info['channels'])} channels x {info['n_samples']} samples "
              f"@ {info['dt_us'] / 1000:g} ms, "
              f"{'contiguous' if info['contiguous'] else 'needs sorting'} "
              f"({time.time() - t0:.0f}s)")

    n_ch = {len(i["channels"]) for i in infos}
    n_samp = {i["n_samples"] for i in infos}
    if len(n_ch) > 1 or len(n_samp) > 1:
        raise SystemExit(f"files disagree: channels {n_ch}, samples {n_samp}")
    n_ch, n_samp = n_ch.pop(), n_samp.pop()

    sel = slice(0, n_ch) if CHANNELS is None else CHANNELS
    keep = np.arange(n_ch)[sel]
    n_out = len(keep)
    n_shots = sum(len(i["records"]) for i in infos)

    nbytes = n_shots * n_samp * n_out * 4
    print(f"\n  output {n_shots} shots x {n_samp} samples x {n_out} channels "
          f"= {nbytes / 1e9:.1f} GB")
    if CHANNELS is not None:
        print(f"  (channels {keep[0]}-{keep[-1]} of {n_ch})")

    os.makedirs(os.path.dirname(os.path.abspath(OUT_PATH)), exist_ok=True)
    out = create_output(OUT_PATH, (n_shots, n_samp, n_out))

    shot = 0
    t_start = time.time()
    for info in infos:
        name = os.path.basename(info["path"])
        n_rec = len(info["records"])
        with segyio.open(info["path"], "r", strict=False,
                         ignore_geometry=True) as f:
            for r in range(n_rec):
                if info["contiguous"]:
                    block = f.trace.raw[r * n_ch:(r + 1) * n_ch]
                else:
                    idx = info["order"][r * n_ch:(r + 1) * n_ch]
                    block = np.stack([f.trace.raw[int(k)] for k in idx])
                # (channels, samples) -> (samples, channels), then subset
                out[shot] = np.ascontiguousarray(block.T[:, sel])
                shot += 1
                if shot % 50 == 0 or shot == n_shots:
                    el = time.time() - t_start
                    print(f"    {shot}/{n_shots} shots  ({name})  "
                          f"{el:.0f}s elapsed, "
                          f"eta {el / shot * (n_shots - shot):.0f}s")
    out.flush()

    print(f"\n  wrote {OUT_PATH}")
    print(f"  shape {out.shape}, dtype {out.dtype}")
    print(f"  amplitude range [{out[0].min():.4g}, {out[0].max():.4g}] "
          f"(shot 0; values untouched)")

    if WRITE_META:
        meta_path = os.path.splitext(OUT_PATH)[0] + "_meta.npz"
        np.savez(
            meta_path,
            receiver_xy=infos[0]["receiver_xy"][sel],
            receiver_channel=infos[0]["channels"][sel],
            source_xy=np.concatenate([i["source_xy"] for i in infos], axis=0),
            record=np.concatenate([i["records"] for i in infos]),
            file_index=np.concatenate([
                np.full(len(i["records"]), k, dtype=np.int32)
                for k, i in enumerate(infos)]),
            file_names=np.array([os.path.basename(p) for p in paths]),
        )
        print(f"  wrote {meta_path}")


if __name__ == "__main__":
    main()
