"""f-k spectrum of the streamer gathers.

Shows where the recorded energy sits relative to the spatial Nyquist set by the
group interval.  That is the thing to look at before trace interpolation: a
curvelet (or any other) interpolator can only reconstruct what is not already
wrapped around in k, so the frequency at which the steepest real events cross
k_Nyquist is the practical ceiling on what the interpolation can recover.

For an event with apparent velocity v, energy at frequency f lies at
k = f / v, so it aliases once f > v * k_Nyquist.  The script prints that
frequency for each velocity in VELOCITY_LINES and draws the corresponding
f = v k lines on the spectrum, so the crossing can be read off directly.

Two domains are available:

  "shot"      one shot gather, nt x 24 traces at the 3.12 m group interval.
              This is the axis the trace interpolation works along.
  "receiver"  one common-receiver gather, nt x 551 shots.  The sail line is
              far more densely sampled than the receiver array, so this shows
              the same wavefield with much less aliasing - useful as a
              reference for what the shot-domain spectrum *should* look like.
              Its DX_M is only a nominal average: the boat's shot spacing is
              not uniform and the track doubles back, so read it qualitatively.

Edit the settings block below, then run from the repository root:

    python -m process.plot_fk
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal.windows import tukey

from utils.data import get_project_root

# ---------------------------------------------------------------- settings --

# (n_shots, n_samples, n_channels) array.  Relative paths resolve against the
# project root.
NPY_PATH = os.path.join("data", "pohang_shore", "numpy", "str_data_geom.npy")

# Sample interval of the array, in milliseconds.  Must match what NPY_PATH
# actually holds - str_data_geom.npy is on the 1 ms grid, str_data_raw.npy and
# str_data_intp.npy are on the SEG-Y's 0.5 ms one.
DT_MS = 1.0

# "shot": gather across the receivers.  "receiver": gather across shots.
DOMAIN = "shot"

# Trace spacing in metres for each domain.  0.7745 m is the interpolated
# streamer spacing: the array spans 71.25 m (285 DAS channels at the true
# 0.25 m acquisition interval) over its 92 intervals.  Use 3.1196 instead when
# pointing this at the un-interpolated str_data_raw.npy.  The shot spacing is a
# nominal average along the sail line and is not uniform.
DX_SHOT_M = 0.7745
DX_RECEIVER_M = 3.8

# Which gather.  In "shot" domain SHOT_INDEX picks the gather and
# CHANNEL_INDEX is ignored; in "receiver" domain it is the other way round.
SHOT_INDEX = 0
CHANNEL_INDEX = 12

# Average |F| over this many gathers, starting at the index above and stepping
# by AVERAGE_STEP.  1 shows a single gather; averaging suppresses the noise in
# the spectrum without smearing the dips.
AVERAGE_COUNT = 32
AVERAGE_STEP = 16

# Zero-pad factors before the FFT.  Padding interpolates the spectrum so the
# dips are legible - with only 24 traces the unpadded k axis is very coarse -
# but it adds no information and does not undo aliasing.
PAD_T = 1
PAD_X = 8

# Tukey taper applied along each axis before the transform, as a fraction of
# the axis length.  Without it the record edges leak across the whole spectrum.
TAPER_T = 0.1
TAPER_X = 0.2

# Display range in dB below the peak.
DB_FLOOR = -60.0

# Show the full frequency axis, -Nyquist to +Nyquist, instead of f >= 0 only.
# The data are real, so |F(-f, -k)| = |F(f, k)| and the lower half is a point
# reflection of the upper one - no new information, but the full plane makes
# the two dip directions and their aliasing wings easier to read off.
SHOW_NEGATIVE_F = True

# Frequency axis limit in Hz; None for the full Nyquist.  With SHOW_NEGATIVE_F
# the axis runs from -F_MAX to +F_MAX.
F_MAX = 250.0

# Apparent velocities (m/s) to overlay as f = v k lines, and to report the
# aliasing frequency for.  1500 is the water speed - the steepest thing a
# marine gather normally contains, so it sets the worst case.
VELOCITY_LINES = [1500.0, 2500.0, 4000.0]

# Also draw the t-x gather next to the spectrum.
SHOW_GATHER = True
CLIP_PERC = 99.0

# Where to write the figure; None to only show it.
SAVE_PATH = None

# ---------------------------------------------------------------------------


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(get_project_root(), path)


def gather_at(data, index):
    """One (n_samples, n_traces) gather in the configured domain."""
    if DOMAIN == "shot":
        return np.asarray(data[index], dtype=float)
    if DOMAIN == "receiver":
        return np.asarray(data[:, :, index], dtype=float).T
    raise SystemExit(f'DOMAIN must be "shot" or "receiver", not {DOMAIN!r}')


def gather_count(data):
    return data.shape[0] if DOMAIN == "shot" else data.shape[2]


def fk_amplitude(gather, nt_pad, nx_pad):
    """|F(f, k)| for one gather, k fftshifted, and f fftshifted as well when
    SHOW_NEGATIVE_F is set.

    Tapered on both axes, then transformed in time and space.
    """
    nt, nx = gather.shape
    w = np.outer(tukey(nt, TAPER_T), tukey(nx, TAPER_X))
    if SHOW_NEGATIVE_F:
        F = np.fft.fft(gather * w, n=nt_pad, axis=0)
        F = np.fft.fft(F, n=nx_pad, axis=1)
        return np.abs(np.fft.fftshift(F, axes=(0, 1)))
    F = np.fft.rfft(gather * w, n=nt_pad, axis=0)
    F = np.fft.fft(F, n=nx_pad, axis=1)
    return np.abs(np.fft.fftshift(F, axes=1))


def frequency_axis(nt_pad, dt):
    if SHOW_NEGATIVE_F:
        return np.fft.fftshift(np.fft.fftfreq(nt_pad, d=dt))
    return np.fft.rfftfreq(nt_pad, d=dt)


def aliasing_report(k_nyq, f_nyq):
    print(f"  spatial Nyquist  k = {k_nyq:.4f} cycles/m  "
          f"(dx = {1 / (2 * k_nyq):.4f} m)")
    print(f"  temporal Nyquist f = {f_nyq:.1f} Hz  (dt = {DT_MS:g} ms)")
    for v in VELOCITY_LINES:
        f_alias = v * k_nyq
        note = "above the temporal Nyquist - never aliased" if f_alias >= f_nyq \
            else "energy above this aliases"
        print(f"  v = {v:6.0f} m/s -> aliases above {f_alias:6.1f} Hz   ({note})")


def draw_gather(ax, gather):
    clip = np.percentile(np.abs(gather), CLIP_PERC)
    nt, nx = gather.shape
    ax.imshow(gather, cmap="gray", vmin=-clip, vmax=clip, aspect="auto",
              interpolation="nearest",
              extent=[0, nx, nt * DT_MS / 1000.0, 0])
    ax.set_xlabel("trace")
    ax.set_ylabel("time [s]")
    ax.set_title(f"{DOMAIN} gather")


def draw_fk(ax, amp, f, k, k_nyq, f_nyq):
    db = 20 * np.log10(np.maximum(amp, 1e-30) / amp.max())
    im = ax.pcolormesh(k, f, db, cmap="turbo", vmin=DB_FLOOR, vmax=0.0,
                       shading="auto")

    f_top = F_MAX if F_MAX else f_nyq
    for v in VELOCITY_LINES:
        # Both dip directions: f = v k and f = -v k. Matplotlib clips them to
        # the axis limits, so the same call serves the half and full planes.
        kk = np.array([-f_top / v, f_top / v])
        ax.plot(kk, v * kk, color="white", lw=1.0, alpha=0.8)
        ax.plot(kk, -v * kk, color="white", lw=1.0, alpha=0.8)
        f_alias = min(v * k_nyq, f_top)
        ax.annotate(f"{v:.0f} m/s", xy=(f_alias / v, f_alias),
                    xytext=(3, -10), textcoords="offset points",
                    color="white", fontsize=8)

    for sign in (-1, 1):
        ax.axvline(sign * k_nyq, color="white", ls="--", lw=1.0, alpha=0.9)
    if SHOW_NEGATIVE_F:
        ax.axhline(0.0, color="white", lw=0.6, alpha=0.4)

    ax.set_xlabel("wavenumber k [cycles/m]")
    ax.set_ylabel("frequency [Hz]")
    ax.set_ylim(-f_top if SHOW_NEGATIVE_F else 0.0, f_top)
    ax.set_title("f-k amplitude [dB re peak]\ndashed: spatial Nyquist")
    return im


def main():
    path = resolve(NPY_PATH)
    if not os.path.isfile(path):
        raise SystemExit(f"not found: {path}")

    data = np.load(path, mmap_mode="r")
    if data.ndim != 3:
        raise SystemExit(f"expected (shots, samples, channels), got {data.shape}")
    print(f"{path}\n  {data.shape[0]} shots x {data.shape[1]} samples "
          f"x {data.shape[2]} channels")

    dx = DX_SHOT_M if DOMAIN == "shot" else DX_RECEIVER_M
    base = SHOT_INDEX if DOMAIN == "shot" else CHANNEL_INDEX
    total = gather_count(data)
    indices = [i for i in range(base, total, AVERAGE_STEP)][:AVERAGE_COUNT]
    if not indices:
        raise SystemExit("the gather selection is empty")

    first = gather_at(data, indices[0])
    nt, nx = first.shape
    nt_pad = int(nt * PAD_T)
    nx_pad = int(nx * PAD_X)

    dt = DT_MS / 1000.0
    f = frequency_axis(nt_pad, dt)
    k = np.fft.fftshift(np.fft.fftfreq(nx_pad, d=dx))
    k_nyq = 1.0 / (2.0 * dx)
    f_nyq = 1.0 / (2.0 * dt)

    print(f"  domain '{DOMAIN}': {nt} samples x {nx} traces at dx = {dx:g} m")
    print(f"  averaging {len(indices)} gathers, "
          f"padded to {nt_pad} x {nx_pad}, "
          f"f axis {'-Nyquist..+Nyquist' if SHOW_NEGATIVE_F else '0..Nyquist'}")
    aliasing_report(k_nyq, f_nyq)

    amp = np.zeros((len(f), nx_pad))
    print(f"  spectrum grid {amp.shape[0]} x {amp.shape[1]}")
    for i in indices:
        amp += fk_amplitude(gather_at(data, i), nt_pad, nx_pad)
    amp /= len(indices)

    if SHOW_GATHER:
        fig, (ax_g, ax_fk) = plt.subplots(
            1, 2, figsize=(13, 6.5), gridspec_kw={"width_ratios": [1, 1.4]})
        draw_gather(ax_g, first)
    else:
        fig, ax_fk = plt.subplots(figsize=(8, 6.5))

    im = draw_fk(ax_fk, amp, f, k, k_nyq, f_nyq)
    fig.colorbar(im, ax=ax_fk, label="dB")
    fig.suptitle(f"Streamer f-k spectrum - {DOMAIN} domain, "
                 f"{len(indices)} gathers averaged")
    fig.tight_layout()

    if SAVE_PATH:
        out = resolve(SAVE_PATH)
        fig.savefig(out, dpi=150)
        print(f"saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
