"""Decimate the DAS receiver axis, and compare the ways of doing it.

    numpy/das_data_line.npy  (551, 2000, 3175) @ 0.25 m
      ->  numpy/das_data_dec.npy  (551, 2000, 264) @ 3.00 m
      ->  numpy/das_data_dec_meta.npz

The DAS samples the fibre every 0.25 m and the streamer every 3.098 m, a factor
of 12.4.  Bringing the DAS down to the streamer's spacing is the alternative to
interpolating the streamer up, and it is the better trade: the interpolation
was measured to manufacture what it cannot know - a one-trace spike lands on
0.90 of the adjacent interpolated trace and above 1 % of 29 of the 93 - while
the DAS is genuinely oversampled for this band.  Only 0.16 % of the 20-300 Hz
energy sits above the 0.75 m grid's Nyquist, and 5.50 % above the 3.00 m one.

Averaging 12 channels also buys sqrt(12) = 3.46x, or 10.8 dB, against
spatially incoherent noise - but so does any filter that keeps a twelfth of
the wavenumber band, because that gain is set by the bandwidth reduction and
not by the filter's shape.  What the filters differ on is the signal.

The methods
-----------
All of them write the same output grid - the centre of each block of STRIDE
channels - so the panels can be compared trace for trace.

    boxcar W      the mean of W channels centred on the block centre.  W has
                  nothing to do with STRIDE: W = STRIDE is the familiar block
                  average, W < STRIDE smooths less and folds more, W > STRIDE
                  is a longer filter on the same output grid.
    k-sinc        low-pass the whole channel axis in the wavenumber domain -
                  flat to K_FLAT * k_Nyquist, raised cosine to k_Nyquist, zero
                  above - then sample.  A brick wall in k is a sinc in x,
                  hence the name.  Nothing can fold, by construction.

Measured over 6 shots at 12:1, against an ideal brick wall, separating the two
kinds of error - folded energy, which lands on signal and cannot be undone,
from passband loss, which is a known amplitude response:

    filter           support   alias error   passband loss
    boxcar 12         3.00 m        10.97 %          9.29 %
    Hann 24           6.00 m         8.03 %         13.40 %
    sinc-Kaiser 49   12.25 m         8.56 %         10.88 %
    k-domain 0.95      global         0.00 %          4.84 %

So a longer local window does not fix the aliasing: the transition the job
needs is a twelfth of the input Nyquist, and no spatially compact filter is
that sharp.  The wavenumber filter is not a compact filter - its support is the
whole array - which is why it can be.

Two details worth knowing
-------------------------
The wavenumber transform is circular in x, so the two ends of the fibre would
otherwise be neighbours; K_PAD channels of tapered pad absorb that and are
cropped off.  And the k-sinc samples the block centre exactly, by a half-
channel phase ramp, so it lands where the even-width boxcars already do; an
odd W cannot be centred on the integer grid and sits 0.125 m off.

Neither this stage nor `line` mutes.  The mute edge is a STEP along the
receiver axis - at a fixed time some channels are muted and their neighbours
are not - and a step is not band-limited in x, so any filter with reach along
that axis rings across it and what leaks in comes from the direct arrival, the
loudest thing in the record.  Measured at 12:1, as rms above the boundary over
the rms in the 30 ms below it, 6 shots x 264 receivers:

    boxcar 3 / 6 / 12     0.00 %                   reach 0 samples
    k-sinc                5.36 % mean, 48.3 % p99  reach median 86, max 435

The boundary varies by only 1.538 ms median inside one 12-channel block, so
the compact boxcars mix nothing a 2-sample guard does not cover.  The k-sinc's
support is the whole array, and re-zeroing afterwards does not undo the
ringing that landed BELOW the boundary.  So the mute waits for
process/rms_normalize.py, which applies it to both domains at once and then
measures the RMS on what is left.  The boundary is written to the sidecar
here; REMUTE can apply it, and is off.

Edit the settings block below, then run from the repository root:

    python -m process.decimate_receiver
"""

import os
import time

import matplotlib
import numpy as np

import config as C
import process.logenv_process as L
import utils.mute as M
from utils.data import create_memmap, get_project_root, npy_shape

import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------- settings --

DOMAIN = "das"
INPUT_STAGE = "line"

# Output channel spacing, as a number of input channels.  12 x 0.25 m = 3.00 m,
# against the streamer's 3.098 m node interval.
STRIDE = 12

# Boxcar widths to compare, in input channels.  Independent of STRIDE - see
# the docstring.  Empty tuple to compare only the k-sinc.
BOXCAR_WINDOWS = (3, 6, 12)

# Compare the wavenumber filter as well.
SHOW_KSINC = True

# The k mask is flat to K_FLAT * k_Nyquist and raised-cosine to k_Nyquist.
# 0.95 costs 4.84 % of the passband and 0.99 only 0.47 %, but a sharper mask
# rings further in x; 0.95 is where that trade sits comfortably.
K_FLAT = 0.95

# Tapered channel pad for the circular transform, cropped off afterwards.
K_PAD = 64

# Re-apply the direct-arrival mute on the output geometry.
#
# OFF.  The mute belongs to the stage after this one, so that it lands on the
# two domains in the same place: process/rms_normalize.py mutes das `deci` and
# str `line` together, each against its own sidecar boundary, and measures the
# RMS on what is left.  Muting here as well would square the taper.
#
# The boundary is computed and written to the sidecar either way.
REMUTE = False

# Write the output array and its sidecar.  Off shows the comparison and
# touches nothing.
WRITE_ALL = True

# The method written when WRITE_ALL is on.  One of the names the comparison
# builds: "k-sinc", or "boxcar <W>" for a W in BOXCAR_WINDOWS - note the space
# and the hyphen, and that the run lists the valid names if this misses.
WRITE_METHOD = "k-sinc"

# Shot gathers to draw.  None picks PREVIEW_N evenly spaced.
#
# The shot direction, not the receiver direction: this preview is about what
# each filter does along the receiver axis, and a shot gather is where that
# axis is the horizontal one.  A receiver gather would put the shot axis
# across and hide it - process/receiver_gather.py is the view for that.
PREVIEW_SHOTS = range(0, 551, 50)
PREVIEW_N = 3

# Time window of the panels, in milliseconds.
WINDOW_MS = (0.0, 2000.0)

# f-k spectra under each gather.  FK_PAD_X tapered traces of pad absorb the
# transform's periodicity in x; FK_F_MAX and FK_DB_FLOOR set the axes.
FK_PAD_X = 16
FK_F_MAX = 320.0
FK_DB_FLOOR = -70.0
FK_H = 3.0

# Inches per panel.
PANEL_W = 4.6
PANEL_H = 4.4

# "linear" draws true amplitude, "log" the log-envelope gain from
# config.LOG_SCALE_PARAMS.
DISPLAY = "linear"
LOG_PARAMS_KEY = "das"
CLIP_PERC = 99.0
CMAP = "seismic"

# Show interactively, and optionally save.  A falsy SAVE_DIR skips saving.
SHOW = True
SAVE_DIR = C.DATA_DIR
BACKENDS = ("QtAgg", "Qt5Agg", "TkAgg", "MacOSX")

# ----------------------------------------------------------------- derived --

DX = C.DAS_CHANNEL_INTERVAL_M
K_NYQ = 1.0 / (2.0 * STRIDE * DX)
OUT_NPY = C.ARRAYS[DOMAIN]["deci"]
OUT_META = C.META[f"{DOMAIN}_deci"]

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


# --------------------------------------------------------------- the grids --


def block_index(n_in):
    """(n_out, channels used) for contiguous blocks of STRIDE."""
    n_out = n_in // STRIDE
    return n_out, n_out * STRIDE


def boxcar_index(n_out, w):
    """(n_out, w) input channels for each output trace.

    Centred on the block centre, (STRIDE - 1) / 2.  An even w lands on it
    exactly; an odd w cannot, and sits half a channel - 0.125 m - to one side.
    """
    off = int(round((STRIDE - 1) / 2.0 - (w - 1) / 2.0))
    if off < 0:
        raise SystemExit(f"boxcar {w} is wider than the {STRIDE}-channel "
                         f"block by more than the block can centre")
    return (np.arange(n_out)[:, None] * STRIDE + off
            + np.arange(w)[None, :])


def centre_offset_m(w):
    off = int(round((STRIDE - 1) / 2.0 - (w - 1) / 2.0))
    return (off + (w - 1) / 2.0 - (STRIDE - 1) / 2.0) * DX


# ------------------------------------------------------------- the filters --


def pad_x(g, p):
    """Channel axis extended with the edge trace faded to zero."""
    if p <= 0:
        return g
    w = 0.5 * (1.0 + np.cos(np.pi * np.arange(1, p + 1) / p))
    return np.concatenate([g[:, :1] * w[::-1][None, :], g,
                           g[:, -1:] * w[None, :]], axis=1)


def k_mask(k):
    """Flat to K_FLAT * k_Nyquist, raised cosine to k_Nyquist, zero above."""
    a, b = K_FLAT * K_NYQ, K_NYQ
    x = (np.abs(k) - a) / max(b - a, 1e-12)
    return np.where(np.abs(k) <= a, 1.0,
                    np.where(np.abs(k) >= b, 0.0,
                             0.5 + 0.5 * np.cos(np.pi * np.clip(x, 0.0, 1.0))))


def boxcar(g, idx):
    return g[:, idx].mean(axis=2)


def ksinc(g, n_use, kern):
    """Wavenumber low-pass over the whole channel axis, then sample.

    `kern` is the mask times the half-channel phase ramp that puts the sample
    on the block centre, so this lands on the same positions the even-width
    boxcars do.
    """
    y = pad_x(g[:, :n_use], K_PAD)
    nx = y.shape[1]
    z = np.fft.irfft(np.fft.rfft(y, axis=1) * kern[None, :], n=nx, axis=1)
    return z[:, K_PAD:K_PAD + n_use:STRIDE]


def build_ksinc_kernel(n_use):
    """The k mask and centring ramp, built once for the padded length."""
    nx = n_use + 2 * K_PAD
    k = np.fft.rfftfreq(nx, DX)
    c = (STRIDE - 1) / 2.0
    return k_mask(k) * np.exp(2j * np.pi * k * c * DX), k, nx


def report_kernel(kern, k, n_use):
    m = np.abs(kern)
    at_half = m[np.argmin(np.abs(k - 0.5 * K_NYQ))]
    at_nyq = m[np.argmin(np.abs(k - K_NYQ))]
    above = m[k > K_NYQ]
    print(f"  k-sinc: flat to {K_FLAT:g} x k_Nyq, mask "
          f"{20 * np.log10(max(at_half, 1e-300)):+.2f} dB at half Nyquist, "
          f"{20 * np.log10(max(at_nyq, 1e-300)):+.1f} dB at Nyquist, "
          f"worst above it {above.max():.2e}")
    print(f"    padded channel axis {n_use} + 2 x {K_PAD} = "
          f"{n_use + 2 * K_PAD}, sampled at the block centre "
          f"(+{(STRIDE - 1) / 2.0 * DX:.3f} m)")


def methods(n_out, n_use):
    """[(name, callable(g) -> (nt, n_out), centre offset in m)]."""
    out = []
    for w in BOXCAR_WINDOWS:
        idx = boxcar_index(n_out, w)
        if idx.max() >= n_use:
            raise SystemExit(f"boxcar {w} reaches channel {idx.max()} of "
                             f"{n_use}")
        out.append((f"boxcar {w}", (lambda i: lambda g: boxcar(g, i))(idx),
                    centre_offset_m(w)))
    if SHOW_KSINC:
        kern, k, nx = build_ksinc_kernel(n_use)
        report_kernel(kern, k, n_use)
        out.append(("k-sinc", (lambda kk: lambda g: ksinc(g, n_use, kk))(kern),
                    0.0))
    if not out:
        raise SystemExit("nothing to do: BOXCAR_WINDOWS is empty and "
                         "SHOW_KSINC is off")
    return out


# ------------------------------------------------------------- the geometry --


def out_geometry(n_out, n_use):
    """Block-centre coordinates, and the mute boundary they imply.

    Source positions come from the `line` sidecar, because that is the
    geometry the input array was rolled onto - not the recorded positions.
    """
    raw = np.load(resolve(C.META["das_raw"]))
    rec = raw["receiver_xy"][:n_use].reshape(n_out, STRIDE, 2).mean(axis=1)
    side = resolve(C.META[f"{DOMAIN}_{INPUT_STAGE}"])
    if not os.path.isfile(side):
        raise SystemExit(f"not found: {side} - run process.line_static first")
    s = np.load(side)
    src = s["source_xy_projected"]
    off = np.hypot(src[:, None, 0] - rec[None, :, 0],
                   src[:, None, 1] - rec[None, :, 1])
    p = C.MUTE_PARAMS
    t = np.maximum(np.sqrt(off ** 2 + p["depth_m"] ** 2)
                   / p["velocity_m_s"] * 1e3 - p["lead_ms"], 0.0)
    return rec, off, t, s


# ---------------------------------------------------------------- preview --


def gain(panel):
    if DISPLAY == "linear":
        return panel, "true amplitude"
    if DISPLAY != "log":
        raise SystemExit(f'DISPLAY must be "linear" or "log", not {DISPLAY!r}')
    lsp = dict(C.LOG_SCALE_PARAMS[LOG_PARAMS_KEY])
    live = panel[np.any(panel != 0.0, axis=1)]
    r = float(np.sqrt(np.mean(live ** 2))) if live.size else 0.0
    y = panel * (C.TARGET_RMS / r) if r > 0 else panel
    return y * L.logscale(y, lsp), (f"log envelope, base {lsp['log_base']:g}, "
                                    f"at rms {C.TARGET_RMS:g}")


def fk(g, dt_s, dx):
    """(f, k, power in dB re its own maximum) for one shot gather.

    The trace axis is tapered-padded first: the 2-D transform is periodic in x
    too, so without it the last trace and the first are neighbours and that
    step smears across the whole plane.
    """
    y = pad_x(g, FK_PAD_X)
    nt, nx = y.shape
    F = np.fft.fft(np.fft.rfft(y, axis=0), axis=1)
    p = np.abs(F) ** 2
    p /= max(p.max(), 1e-300)
    return (np.fft.rfftfreq(nt, dt_s), np.fft.fftshift(np.fft.fftfreq(nx, dx)),
            10.0 * np.log10(np.maximum(np.fft.fftshift(p, axes=1), 1e-30)))


def fk_panel(ax, g, dt_ms, dx, k_nyq_out, title):
    """One f-k spectrum, with the output grid's Nyquist marked."""
    f, k, p = fk(g, dt_ms * 1e-3, dx)
    hi = min(len(f) - 1, int(np.searchsorted(f, FK_F_MAX)))
    ax.imshow(p[:hi + 1], cmap="magma", vmin=FK_DB_FLOOR, vmax=0.0,
              aspect="auto", origin="lower", interpolation="nearest",
              extent=[k[0], k[-1], f[0], f[hi]])
    for s in (-1, 1):
        ax.axvline(s * k_nyq_out, color="cyan", lw=0.9, ls="--")
    ax.set_xlabel("wavenumber [cycles/m]", fontsize=8)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=7)


def figure(shot, names, cols, dxs, t_mute, rec_along, t0, dt_ms):
    """One shot: two rows - the gathers, then their f-k spectra.

    `cols` runs (input, method, method, ...) for this shot; `dxs` is the trace
    spacing of each of those columns, since the input is on DX and every
    output on STRIDE * DX.  Columns therefore compare the data against each
    way of decimating it, and the spectrum under each says where its energy
    went.

    The trace axis is drawn in metres along the fibre rather than in traces,
    so a 3175-channel column and a 264-trace one line up and can be zoomed
    together.  `t_mute` is this shot's boundary on the output grid, at
    `rec_along` metres; it is drawn on every column - the same physical curve
    - and none of the panels has it applied.

    One figure per shot rather than all of them stacked: at a dozen preview
    shots a single figure would be two dozen rows tall and unreadable, and
    main() shows them one at a time so only one is in memory.
    """
    head = ["input"] + list(names)
    ncol = len(head)
    fig, axes = plt.subplots(2, ncol, squeeze=False,
                             figsize=(PANEL_W * ncol, PANEL_H + FK_H),
                             height_ratios=[PANEL_H, FK_H])
    label = "true amplitude"
    ref = None
    for c, name in enumerate(head):
        g = np.asarray(cols[c], dtype=np.float64)
        dx = dxs[c]
        y, label = gain(g)
        clip = float(np.percentile(np.abs(y), CLIP_PERC)) or 1.0
        ax = axes[0][c]
        ntr = y.shape[1]
        ax.imshow(y, cmap=CMAP, vmin=-clip, vmax=clip, aspect="auto",
                  interpolation="nearest",
                  extent=[0, ntr * dx, t0 + y.shape[0] * dt_ms, t0])
        if t_mute is not None:
            ax.plot(rec_along, t_mute, color="lime", lw=0.8,
                    label="mute boundary (not applied)")
            if c == 0:
                ax.legend(fontsize=7, loc="lower left")
        ax.set_title(f"{name}   {ntr} traces @ {dx:g} m", fontsize=9)
        ax.set_xlabel("along the fibre [m]", fontsize=8)
        ax.tick_params(labelsize=7)
        if c == 0:
            ax.set_ylabel("time [ms]")
        else:
            ax.tick_params(labelleft=False)
        # linked in metres and milliseconds, so the columns line up despite
        # carrying different trace counts
        if ref is None:
            ref = ax
        else:
            ax.sharex(ref)
            ax.sharey(ref)
        fk_panel(axes[1][c], g, dt_ms, dx, K_NYQ,
                 f"f-k, {name}" + ("   dashed: output Nyquist "
                                   f"{K_NYQ:.4f} c/m" if c == 0 else ""))
    fig.suptitle(f"{DOMAIN} {INPUT_STAGE} shot {shot}   receiver decimation "
                 f"{STRIDE}:1, {DX:g} -> {STRIDE * DX:g} m   ({label})\n"
                 f"the input is on {DX:g} m and every method on "
                 f"{STRIDE * DX:g} m, so the trace axis is drawn in metres")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0 - 0.6 / (PANEL_H + FK_H)))
    return fig


def pick(n_shots):
    want = PREVIEW_SHOTS
    if want is None:
        want = np.linspace(0, n_shots - 1, PREVIEW_N).round().astype(int)
    bad = [int(j) for j in want if not 0 <= j < n_shots]
    if bad:
        raise SystemExit(f"preview shots out of range 0..{n_shots - 1}: "
                         f"{bad}")
    return [int(j) for j in want]


# ------------------------------------------------------------------- stage --


def main():
    show = show_figures()
    in_path = resolve(C.ARRAYS[DOMAIN][INPUT_STAGE])
    if not os.path.isfile(in_path):
        raise SystemExit(f"not found: {in_path}")
    data = np.load(in_path, mmap_mode="r")
    n_shots, nt, n_in = data.shape
    dt_ms = C.STAGE_DT_US[DOMAIN][INPUT_STAGE] / 1000.0

    n_out, n_use = block_index(n_in)
    print(f"{C.ARRAYS[DOMAIN][INPUT_STAGE]}")
    print(f"  {n_shots} shots x {nt} samples @ {dt_ms:g} ms x {n_in} channels "
          f"@ {DX:g} m")
    print(f"  {STRIDE}:1 -> {n_out} traces @ {STRIDE * DX:g} m, "
          f"k_Nyquist {1 / (2 * DX):.3f} -> {K_NYQ:.4f} c/m")
    if n_use != n_in:
        print(f"  {n_in - n_use} trailing channels dropped so the blocks "
              f"divide evenly")
    print(f"  incoherent-noise gain from {STRIDE} channels: "
          f"x{np.sqrt(STRIDE):.2f} ({10 * np.log10(STRIDE):.1f} dB), the same "
          f"for every filter below - it is the bandwidth, not the shape")

    ms = methods(n_out, n_use)
    names = [m[0] for m in ms]
    for name, _, off in ms:
        print(f"    {name:<12} output centre offset {off:+.3f} m")

    rec, off, t_mute, side = out_geometry(n_out, n_use)
    # metres along the fibre of each output trace, for the drawn axis
    rec_along = (np.arange(n_out) * STRIDE + (STRIDE - 1) / 2.0) * DX
    print(f"  output geometry: offset {off.min():.1f}..{off.max():.0f} m, "
          f"mute boundary {t_mute.min():.0f}..{t_mute.max():.0f} ms")
    print(f"    the preview never applies it; the written array "
          f"{'does' if REMUTE else 'does not'} (REMUTE)")
    if bool(side["mute_applied"]) if "mute_applied" in side else False:
        print(f"    WARNING: {C.META[f'{DOMAIN}_{INPUT_STAGE}']} says the "
              f"input was ALREADY muted - re-run process.line_static with "
              f"APPLY_MUTE off, or the ringing this stage avoids is baked in")

    a = max(0, int(round(WINDOW_MS[0] / dt_ms)))
    b = min(nt, int(round(WINDOW_MS[1] / dt_ms)))
    win = slice(a, b)
    prev = pick(n_shots)
    print(f"  preview shots {prev}, window {a * dt_ms:.0f}-"
          f"{b * dt_ms:.0f} ms")

    dxs = [DX] + [STRIDE * DX] * len(ms)

    out = None
    if WRITE_ALL:
        if WRITE_METHOD not in names:
            raise SystemExit(f"WRITE_METHOD {WRITE_METHOD!r} is not among "
                             f"{names}")
        out_path = resolve(OUT_NPY)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        if os.path.isfile(out_path):
            print(f"  overwriting {OUT_NPY} (was {npy_shape(out_path)})")
        out = create_memmap(out_path, (n_shots, nt, n_out))
        print(f"  writing {WRITE_METHOD!r}: {out.shape}, "
              f"{out.nbytes / 1e9:.2f} GB")
    else:
        print(f"  WRITE_ALL is off - showing the comparison only, "
              f"{OUT_NPY} not touched")

    # The write pass, over every shot, drawing nothing.  Kept separate from
    # the preview so a blocking plt.show() cannot land in the middle of it.
    if out is not None:
        fn_write = dict((n, f) for n, f, _ in ms)[WRITE_METHOD]
        t0 = time.time()
        for s in range(n_shots):
            y = fn_write(np.asarray(data[s], dtype=np.float64))
            if REMUTE:
                y = y * M.weights(t_mute[s], nt, dt_ms)
            out[s] = y.astype(np.float32)
            if (s + 1) % 25 == 0 or s + 1 == n_shots:
                el = time.time() - t0
                print(f"    {s + 1}/{n_shots} shots  {el:.0f}s elapsed, "
                      f"eta {el / (s + 1) * (n_shots - s - 1):.0f}s",
                      flush=True)
        out.flush()
        del out
        print(f"  wrote {resolve(OUT_NPY)}")
        meta = resolve(OUT_META)
        np.savez(
            meta,
            receiver_xy=rec, source_xy_projected=side["source_xy_projected"],
            along_m=side["along_m"], sorted_order=side["sorted_order"],
            offset_m=off.astype(np.float32),
            mute_boundary_ms=t_mute.astype(np.float32),
            input_stage=INPUT_STAGE, dt_us=C.STAGE_DT_US[DOMAIN][INPUT_STAGE],
            n_samples=nt, stride=STRIDE, dx_m=STRIDE * DX,
            method=WRITE_METHOD, k_flat=K_FLAT, k_pad=K_PAD, remuted=REMUTE,
        )
        print(f"  wrote {meta}")

    # One window per shot, built and shown before the next is made: plt.show()
    # blocks until it is closed, so the shots arrive one at a time and only
    # one figure is ever in memory.
    if not prev:
        return
    out_dir = resolve(SAVE_DIR) if SAVE_DIR else None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    print(f"\n  preview: {len(prev)} shots, one window at a time")
    for n, s in enumerate(prev, 1):
        g = np.asarray(data[s], dtype=np.float64)
        # The preview keeps each filter's output BEFORE the mute, so the f-k
        # spectra measure the filters and not the mute edge - which is a step
        # along x, is not band-limited, and rings under any filter with reach.
        # The boundary is drawn on the panels instead.
        cols = [g[win, :n_use]] + [fn(g)[win] for _, fn, _ in ms]
        print(f"    [{n}/{len(prev)}] shot {s:>4}: offset "
              f"{off[s].min():.0f}..{off[s].max():.0f} m, boundary "
              f"{t_mute[s].min():.0f}..{t_mute[s].max():.0f} ms", flush=True)
        fig = figure(s, names, cols, dxs, t_mute[s], rec_along,
                     a * dt_ms, dt_ms)
        if out_dir:
            p = os.path.join(out_dir, f"decimate_receiver_{DOMAIN}_"
                                      f"{STRIDE}x_shot{s:04d}.png")
            fig.savefig(p, dpi=130)
            print(f"        wrote {p}", flush=True)
        if show:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
