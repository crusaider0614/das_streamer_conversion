"""DAS decimated on the receiver axis and cropped - nothing else.

    numpy/das_data_raw.npy  (551, 3500, 3175) @ 0.25 m, 1 ms
      ->  export/das_{S}x_{CROP}ms.npy   (551, 2000, 793) @ 1.00 m at S = 4
      ->  export/das_{S}x_{CROP}ms.txt   shot and receiver coordinates

An export, not a pipeline stage, and deliberately plainer than the `line`
route.  What is and is not done:

    done      receiver-axis decimation S:1, with the wavenumber low-pass the
              decimation itself requires
              the record cropped to CROP_MS from t = 0

    not done  no line fit and no roll.  The shots keep their RECORDED
              positions; they are not projected onto the survey line and the
              samples are not statically shifted onto it.  process/
              line_static.py is the route that does that, for the 12:1 pair
              the CUT training uses; this export is for whatever wants the
              array as recorded.
              no band-pass, no mute, no normalisation.

Because there is no roll, the geometry here is simply the recorded geometry,
and the text file quotes it directly - no projected coordinates, no along/across
line columns, nothing that would only make sense relative to a fitted line.

The decimation
--------------
S:1 needs a low-pass at the output Nyquist 1 / (2 * S * 0.25) c/m or the
wavenumbers above it fold.  Done in the wavenumber domain over the whole
channel axis, flat to K_FLAT * Nyquist and raised-cosine to it, which is the
only way to get a transition that sharp - a boxcar of any length leaves 8-11 %
of the band folded, measured.  See process/decimate_receiver.py for that
comparison.

At S = 4 the output interval is 1.00 m and almost nothing is lost: only 0.16 %
of the DAS's 20-300 Hz energy sits above even the 0.75 m grid's Nyquist, so a
1.00 m grid is comfortably above what the data carries.

The crop and the decimation commute exactly - one is along time, the other
along the receiver axis, and no filter here spans time - so the crop is done
first, on the cheaper array.

Edit the settings block below, then run from the repository root:

    python -m process.export_das_decimated
"""

import os
import time

import numpy as np
import scipy.fft as sf

import config as C
from utils.data import create_memmap, get_project_root, npy_shape

# ---------------------------------------------------------------- settings --

# Channels averaged per output trace.  4 x 0.25 m = 1.00 m.
STRIDE = 4

# Milliseconds of record kept, from t = 0.
CROP_MS = 2000.0

# The wavenumber mask is flat to K_FLAT * k_Nyquist and raised-cosine to it.
K_FLAT = 0.95

# Tapered channel pad for the circular transform, cropped off afterwards.
K_PAD = 64

# Where it goes.  Outside config.DATA_DIR - this is an export, not a pipeline
# product.  Relative paths resolve against the repository root.
OUT_DIR = "export"

# Write the array.  Off writes only the text file, which costs a second.
WRITE_ARRAY = True

# Also write the full (n_shots, n_traces) offset matrix into the text file.
# Off by default: at stride 4 that is 551 x 793 numbers, about 9 MB.
WRITE_OFFSETS = False

# ----------------------------------------------------------------- derived --

DX = C.DAS_CHANNEL_INTERVAL_M
DT_MS = C.STAGE_DT_US["das"]["raw"] / 1000.0
K_NYQ = 1.0 / (2.0 * STRIDE * DX)
NT = int(round(CROP_MS / DT_MS))
STEM = f"das_{STRIDE}x_{CROP_MS:.0f}ms"

# ---------------------------------------------------------------------------


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(get_project_root(), path)


def pad_x(g, p):
    """Channel axis extended with the edge trace faded to zero."""
    if p <= 0:
        return g
    w = (0.5 * (1.0 + np.cos(np.pi * np.arange(1, p + 1) / p))
         ).astype(g.dtype)
    return np.concatenate([g[:, :1] * w[::-1][None, :], g,
                           g[:, -1:] * w[None, :]], axis=1)


def build_kernel(n_use):
    """The wavenumber mask, times the ramp that centres the sample.

    The output trace sits at the centre of its block of STRIDE channels, which
    for an even STRIDE is a half-channel off the grid; a phase ramp puts it
    there exactly.
    """
    nx = n_use + 2 * K_PAD
    k = sf.rfftfreq(nx, DX)
    a, b = K_FLAT * K_NYQ, K_NYQ
    x = (k - a) / max(b - a, 1e-12)
    m = np.where(k <= a, 1.0,
                 np.where(k >= b, 0.0,
                          0.5 + 0.5 * np.cos(np.pi * np.clip(x, 0.0, 1.0))))
    c = (STRIDE - 1) / 2.0
    return (m * np.exp(2j * np.pi * k * c * DX)).astype(np.complex64), k, m


def decimate(g, n_use, kern):
    """Wavenumber low-pass over the whole channel axis, then sample."""
    y = pad_x(g[:, :n_use], K_PAD)
    nx = y.shape[1]
    F = sf.rfft(y, axis=1)
    F *= kern[None, :]
    z = sf.irfft(F, n=nx, axis=1)
    return z[:, K_PAD:K_PAD + n_use:STRIDE]


def geometry(n_use, n_out):
    """Recorded source positions, and block-centre receiver positions."""
    raw = np.load(resolve(C.META["das_raw"]))
    rec_all, chan = raw["receiver_xy"], raw["receiver_channel"]
    rec = rec_all[:n_use].reshape(n_out, STRIDE, 2).mean(axis=1)
    ch = chan[:n_use].reshape(n_out, STRIDE)
    arc = (np.arange(n_out) * STRIDE + (STRIDE - 1) / 2.0) * DX
    return raw["source_xy"], rec, ch, arc


def write_text(path, src, rec, ch, arc, off, n_in, n_use):
    n_shots, n_out = off.shape
    p = C.MUTE_PARAMS
    t = np.maximum(np.sqrt(off ** 2 + p["depth_m"] ** 2)
                   / p["velocity_m_s"] * 1e3 - p["lead_ms"], 0.0)
    L = [
        "# DAS shot and receiver geometry",
        f"# configuration: {STRIDE}:1 receiver decimation, "
        f"{CROP_MS:.0f} ms of record, nothing else",
        "#",
        f"#   shots             {n_shots}",
        f"#   receivers         {n_out}  (from {n_in} channels at {DX:g} m; "
        f"{n_in - n_use} trailing channels dropped)",
        f"#   receiver interval {STRIDE * DX:g} m",
        f"#   samples           {NT} at {DT_MS:g} ms = {CROP_MS:.0f} ms",
        f"#   array shape       ({n_shots}, {NT}, {n_out})",
        f"#   array dtype       float32",
        "#",
        "# Coordinates are the SEG-Y easting/northing in metres.  Receiver",
        f"# positions are block centres - the mean of each block of {STRIDE}",
        "# channels.  Source positions are AS RECORDED: no line has been",
        "# fitted and no static shift applied, so there is nothing here that",
        "# is relative to a survey line.",
        "#",
        "# The samples are raw apart from the decimation and the crop - no",
        "# band-pass, no mute, no normalisation.",
        "#",
        f"#   offsets           {off.min():.2f} .. {off.max():.2f} m",
        f"#   (for reference) the direct-arrival mute this project uses would",
        f"#   be sqrt(offset^2 + {p['depth_m']:g}^2) / {p['velocity_m_s']:g}"
        f" * 1e3 - ({p['lead_ms']:g}) ms,",
        f"#   raised-cosine taper {p['taper_ms']:g} ms, giving "
        f"{t.min():.0f} .. {t.max():.0f} ms.  Not applied.",
        "#",
        f"# --- {n_shots} shots "
        f"----------------------------------------",
        "# shot           east          north",
    ]
    for i in range(n_shots):
        L.append(f"{i:6d} {src[i, 0]:14.3f} {src[i, 1]:14.3f}")
    L += [
        "#",
        f"# --- {n_out} receivers "
        f"------------------------------------",
        "# trace   chan_first    chan_last           east"
        "          north    along_fibre_m",
    ]
    for j in range(n_out):
        L.append(f"{j:6d} {int(ch[j, 0]):12d} {int(ch[j, -1]):12d} "
                 f"{rec[j, 0]:14.3f} {rec[j, 1]:14.3f} {arc[j]:16.3f}")
    if WRITE_OFFSETS:
        L += ["#",
              f"# --- offsets in metres, {n_shots} rows x {n_out} columns ---"]
        for i in range(n_shots):
            L.append(" ".join(f"{v:.2f}" for v in off[i]))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return len(L)


def main():
    in_rel = C.ARRAYS["das"]["raw"]
    in_path = resolve(in_rel)
    if not os.path.isfile(in_path):
        raise SystemExit(f"not found: {in_path}")
    data = np.load(in_path, mmap_mode="r")
    n_shots, nt_in, n_in = data.shape
    if NT > nt_in:
        raise SystemExit(f"{CROP_MS:.0f} ms is {NT} samples but the record is "
                         f"only {nt_in}")
    n_out = n_in // STRIDE
    n_use = n_out * STRIDE

    print(f"{in_rel}  {data.shape} @ {DT_MS:g} ms x {DX:g} m")
    print(f"  {STRIDE}:1 -> {n_out} traces @ {STRIDE * DX:g} m, "
          f"k_Nyquist {1 / (2 * DX):.3f} -> {K_NYQ:.4f} c/m")
    print(f"  crop {nt_in} -> {NT} samples ({CROP_MS:.0f} ms)")
    if n_use != n_in:
        print(f"  {n_in - n_use} trailing channels dropped so the blocks "
              f"divide evenly")
    print(f"  no line fit, no roll, no band-pass, no mute, no normalisation")

    kern, k, m = build_kernel(n_use)
    at_half = m[np.argmin(np.abs(k - 0.5 * K_NYQ))]
    print(f"  k mask: flat to {K_FLAT:g} x Nyquist, "
          f"{20 * np.log10(max(at_half, 1e-300)):+.2f} dB at half Nyquist, "
          f"worst gain above Nyquist {m[k > K_NYQ].max():.2e}")

    src, rec, ch, arc = geometry(n_use, n_out)
    if len(src) != n_shots:
        raise SystemExit(f"{len(src)} source positions for {n_shots} shots")
    off = np.hypot(src[:, None, 0] - rec[None, :, 0],
                   src[:, None, 1] - rec[None, :, 1])
    print(f"  offsets {off.min():.2f}..{off.max():.2f} m "
          f"(recorded source positions)")

    out_dir = resolve(OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    txt = os.path.join(out_dir, f"{STEM}.txt")
    lines = write_text(txt, src, rec, ch, arc, off, n_in, n_use)
    print(f"  wrote {txt}  ({os.path.getsize(txt) / 1e6:.2f} MB, "
          f"{lines} lines)")

    if not WRITE_ARRAY:
        print(f"  WRITE_ARRAY is off - no .npy written")
        return

    npy = os.path.join(out_dir, f"{STEM}.npy")
    if os.path.isfile(npy):
        print(f"  overwriting {npy} (was {npy_shape(npy)})")
    out = create_memmap(npy, (n_shots, NT, n_out))
    print(f"  output {out.shape}, {out.nbytes / 1e9:.2f} GB")

    t0 = time.time()
    for i in range(n_shots):
        g = np.asarray(data[i, :NT], dtype=np.float32)
        out[i] = decimate(g, n_use, kern).astype(np.float32)
        if (i + 1) % 50 == 0 or i + 1 == n_shots:
            el = time.time() - t0
            print(f"    {i + 1}/{n_shots} shots  {el:.0f}s elapsed, "
                  f"eta {el / (i + 1) * (n_shots - i - 1):.0f}s", flush=True)
    out.flush()
    del out
    print(f"  wrote {npy}")


if __name__ == "__main__":
    main()
