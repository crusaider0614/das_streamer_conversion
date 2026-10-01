"""Average amplitude spectra of the far-offset traces: DAS, translated, streamer.

    DAS          das_data_rg_{train,infer}.npy   or das_data_norm.npy
    translated   das_data_fake_str.npy           whichever epoch translate_das.py wrote
    streamer     str_data_rg_train.npy           or str_data_norm.npy

Only traces at or beyond the far split - the streamer's own minimum offset,
184.06 m, the same split eval_cut_checkpoints.py uses - so all three sides are
looking at the offset range both instruments actually cover.

RECEIVERS
---------
    "paired"  the 24 DAS receivers inside the streamer aperture against the
              24 streamer traces, row for row, and a (receiver, shot) pair is
              used only when it is far on both sides.  The same traces on all
              three curves, so a difference between them is the translation
              and not the population.
    "all"     every far DAS trace on the whole 264-receiver line for DAS and
              translated, against the 24 streamer traces.  More traces, but
              the DAS curves then include receivers the streamer never saw.

DOMAIN
------
    "log"     every side as stored, in the log-envelope domain the generator
              works in.  Nothing is estimated.
    "normal"  the envelope gain off.  DAS and streamer come from their `norm`
              files, which are exactly the data before the gain went on;
              the translated side has its streamer gain taken back off with
              calculate_norscale_inversion, PAD_FRONT zeros in front as in
              process/logenv_process.py.  The streamer parameters smooth over
              time only (sigma (5, 0)), so inverting just the traces used here
              is the same as inverting the whole gather - which keeps "paired"
              at about half a second a shot.  "all" inverts 264 traces a shot,
              about six seconds, so raise SHOT_STEP for it.

Spectra are averaged as power and drawn as amplitude in dB.  With
PER_TRACE_NORM each trace's power spectrum is divided by its own total before
averaging, so every trace counts once - the same weighting the per-trace
centroid in eval_cut_checkpoints.py has.  Without it the loud traces dominate.

The printed centroid is the centroid of the averaged spectrum, not the median
of per-trace centroids that eval_cut_checkpoints.py reports, so the two are
close but not equal.

Edit the settings block below, then run from the repository root:

    python -m inference.plot_far_spectrum
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import config as C
import process.shot_geometry as G
from utils.data import get_project_root
from utils.process import calculate_norscale_inversion

# ---------------------------------------------------------------- settings --

# Label for the translated curve - the file itself does not record which
# checkpoint wrote it.
FAKE_LABEL = "translated (ep 150)"

RECEIVERS = "paired"        # "paired" or "all"
DOMAIN = "log"              # "log" or "normal"

# Every SHOT_STEP-th shot, in sorted order.
SHOT_STEP = 1

# (start, stop) in ms of the part of each trace to transform; None is the
# whole 2000 ms.  The muted top is zero on every side and only adds zeros.
TIME_WINDOW_MS = None

# Far split in metres; None takes the streamer's own minimum offset.
OFFSET_SPLIT_M = None

PER_TRACE_NORM = True

# Inversion of the translated side in DOMAIN "normal".  Same meaning as in
# inference/plot_translated_shots.py.
ITERATIONS = 50
PAD_FRONT = 2000

DT_MS = 1.0
F_MAX_HZ = 500.0

# Where to write the figure; None only shows it.
SAVE_PATH = None

# ---------------------------------------------------------------------------


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(get_project_root(), path)


def load(path, name):
    p = resolve(path)
    if not os.path.isfile(p):
        raise SystemExit(f"not found: {p}  ({name})")
    return np.load(p, mmap_mode="r")


def shared_split(rec_xy):
    """Receiver indices inside / outside the streamer aperture, by position."""
    s = np.load(resolve(C.META["str_line"]))
    a_rec = (rec_xy - s["line_centroid"]) @ s["line_direction"]
    a_str = (s["receiver_xy"] - s["line_centroid"]) @ s["line_direction"]
    lo, hi = a_str.min(), a_str.max()
    inside = np.flatnonzero((a_rec >= lo) & (a_rec <= hi))
    outside = np.flatnonzero((a_rec < lo) | (a_rec > hi))
    return inside, outside


def sorted_offsets(order):
    """(receiver, shot) offsets in sorted shot order, DAS and streamer."""
    z = np.load(resolve(C.META["das_deci"]))
    off_das = z["offset_m"][order].T.astype(np.float64)
    s = np.load(resolve(C.META["str_line"]))
    off_str = s["offset_projected_m"][order].T.astype(np.float64)
    return off_das, off_str, z["receiver_xy"]


class Sides:
    """Shot gathers (n_t, n_traces) of the three sides, in sorted shot order.

    `das(i, cols)` and `fake(i, cols)` take full-line receiver indices;
    `str(i)` returns all 24 streamer traces.
    """

    def __init__(self, order, inside, outside):
        self.order = order
        self.fake_arr = load(C.ARRAYS["das"]["fake_str"], "translated")
        if DOMAIN == "log":
            self.rg = [load(C.ARRAYS["das"]["rg_train"], "DAS rg_train"),
                       load(C.ARRAYS["das"]["rg_infer"], "DAS rg_infer")]
            n = len(inside) + len(outside)
            self.owner, self.row = np.empty(n, int), np.empty(n, int)
            self.owner[inside], self.row[inside] = 0, np.arange(len(inside))
            self.owner[outside], self.row[outside] = 1, np.arange(len(outside))
            self.str_arr = load(C.ARRAYS["str"]["rg_train"], "streamer rg_train")
        else:
            self.das_arr = load(C.ARRAYS["das"]["norm"], "DAS norm")
            self.str_arr = load(C.ARRAYS["str"]["norm"], "streamer norm")

    def das(self, i, cols):
        if DOMAIN == "normal":
            return np.asarray(self.das_arr[self.order[i]][:, cols], np.float64)
        # one strided read per array that holds any of the columns, rather
        # than one per receiver
        out = np.empty((self.fake_arr.shape[1], len(cols)))
        own = self.owner[cols]
        for o in np.unique(own):
            j = np.flatnonzero(own == o)
            block = np.asarray(self.rg[o][:, :, i])
            out[:, j] = block[self.row[cols[j]]].T
        return out

    def fake(self, i, cols):
        g = np.asarray(self.fake_arr[i][:, cols], np.float64)
        if DOMAIN == "normal":
            g = invert_str(g)
        return g

    def str(self, i):
        if DOMAIN == "normal":
            return np.asarray(self.str_arr[self.order[i]], np.float64)
        return np.asarray(self.str_arr[:, :, i], np.float64).T


def invert_str(g):
    lsp = dict(C.LOG_SCALE_PARAMS["str"])
    if PAD_FRONT:
        g = np.concatenate([np.zeros((PAD_FRONT, g.shape[1])), g], axis=0)
    _, est = calculate_norscale_inversion(g, iterations=ITERATIONS, **lsp)
    return est[PAD_FRONT:] if PAD_FRONT else est


class Accum:
    """Running sum of power spectra over traces."""

    def __init__(self, n_f):
        self.p = np.zeros(n_f)
        self.n = 0

    def add(self, x):
        if x.shape[1] == 0:
            return
        P = np.abs(np.fft.rfft(x, axis=0)) ** 2
        tot = P.sum(0)
        live = tot > 0
        P = P[:, live]
        if PER_TRACE_NORM:
            P = P / tot[live]
        self.p += P.sum(1)
        self.n += P.shape[1]

    def mean(self):
        return self.p / max(self.n, 1)


def window(g):
    if TIME_WINDOW_MS is None:
        return g
    a, b = (int(round(t / DT_MS)) for t in TIME_WINDOW_MS)
    return g[a:b]


def centroid(f, p):
    return float((f * p).sum() / p.sum()) if p.sum() > 0 else np.nan


def main():
    if RECEIVERS not in ("paired", "all"):
        raise SystemExit(f"RECEIVERS {RECEIVERS!r} is not 'paired' or 'all'")
    if DOMAIN not in ("log", "normal"):
        raise SystemExit(f"DOMAIN {DOMAIN!r} is not 'log' or 'normal'")

    order = np.asarray(G.sorted_order()["orig_idx"], dtype=int)
    off_das, off_str, rec_xy = sorted_offsets(order)
    inside, outside = shared_split(rec_xy)
    if len(inside) != off_str.shape[0]:
        raise SystemExit(f"{len(inside)} DAS receivers inside the aperture "
                         f"against {off_str.shape[0]} streamer traces")
    split = float(off_str.min()) if OFFSET_SPLIT_M is None else OFFSET_SPLIT_M

    sides = Sides(order, inside, outside)
    n_shot = sides.fake_arr.shape[0]
    n_t = window(np.zeros((sides.fake_arr.shape[1], 1))).shape[0]
    f = np.fft.rfftfreq(n_t, DT_MS / 1000.0)
    acc = {k: Accum(len(f)) for k in ("das", "fake", "str")}

    shots = range(0, n_shot, SHOT_STEP)
    print(f"{RECEIVERS} receivers, {DOMAIN} domain, far >= {split:.2f} m, "
          f"{len(shots)} shots, per-trace norm {PER_TRACE_NORM}")

    for n, i in enumerate(shots):
        far_d = off_das[:, i] >= split
        far_s = off_str[:, i] >= split
        if RECEIVERS == "paired":
            keep = far_d[inside] & far_s
            cols = inside[keep]
            s_cols = np.flatnonzero(keep)
        else:
            cols = np.flatnonzero(far_d)
            s_cols = np.flatnonzero(far_s)
        if len(cols):
            acc["das"].add(window(sides.das(i, cols)))
            acc["fake"].add(window(sides.fake(i, cols)))
        if len(s_cols):
            acc["str"].add(window(sides.str(i)[:, s_cols]))
        if (n + 1) % 50 == 0:
            print(f"  {n + 1}/{len(shots)} shots", flush=True)

    curves = {
        "das": ("DAS input", "tab:blue", acc["das"].mean()),
        "fake": (FAKE_LABEL, "tab:red", acc["fake"].mean()),
        "str": ("streamer", "k", acc["str"].mean()),
    }
    band = f <= F_MAX_HZ
    for k, (name, _, p) in curves.items():
        print(f"  {name:22s} {acc[k].n:7d} traces   centroid "
              f"{centroid(f, p):6.1f} Hz   peak {f[np.argmax(p)]:6.1f} Hz")

    def db(p):
        return 10 * np.log10(np.maximum(p, 1e-30) / p[band].max())

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                                   gridspec_kw=dict(height_ratios=(2, 1)))
    for name, colour, p in curves.values():
        ax0.plot(f[band], db(p)[band], color=colour, lw=1.2,
                 label=f"{name}  ({centroid(f, p):.1f} Hz)")
    ax0.set_ylabel("amplitude [dB, own peak = 0]")
    ax0.set_title(f"far offsets (>= {split:.1f} m), {RECEIVERS} receivers, "
                  f"{DOMAIN} domain", fontsize=10)
    ax0.legend(fontsize=9)
    ax0.grid(alpha=0.3)

    # Both against the streamer, each normalised to its total power first so
    # the ratio is a difference of spectral shape and not of overall level.
    ref = curves["str"][2] / curves["str"][2][band].sum()
    for k in ("das", "fake"):
        name, colour, p = curves[k]
        r = (p / p[band].sum()) / np.maximum(ref, 1e-30)
        ax1.plot(f[band], 10 * np.log10(np.maximum(r, 1e-30))[band],
                 color=colour, lw=1.2, label=f"{name} / streamer")
    ax1.axhline(0, color="k", lw=0.8)
    ax1.set_xlabel("frequency [Hz]")
    ax1.set_ylabel("shape ratio [dB]")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    fig.tight_layout()

    if SAVE_PATH:
        fig.savefig(resolve(SAVE_PATH), dpi=150)
        print(f"saved {resolve(SAVE_PATH)}")
    plt.show()


if __name__ == "__main__":
    main()
