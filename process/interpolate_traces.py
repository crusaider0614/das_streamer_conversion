"""Run the curvelet trace interpolation over the whole survey.

The production counterpart to `process/intp_sweep.py`: one fixed set of
hyperparameters, every shot, streamed to disk instead of held in memory.  Takes
`str_data_bl.npy` (551 x 7000 x 24, band-limited to 220 Hz) up to
551 x 7000 x 93 in two 2x stages, 3.12 -> 1.56 -> 0.78 m.

Three things the sweep script does not do:

  * writes straight into a memory-mapped output, so peak memory stays at one
    gather per worker rather than the whole survey;
  * spreads shots across processes - the cascade is independent per shot, and
    the transforms are single-threaded, so this scales nearly linearly;
  * records which shots are finished, so an interrupted run picks up where it
    stopped instead of starting over.

Edit the settings block below, then run from the repository root:

    python -m process.interpolate_traces
"""

import multiprocessing as mp
import os
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import process.intp_sweep as S  # noqa: E402
from utils.data import get_project_root  # noqa: E402

# ---------------------------------------------------------------- settings --

# The 220 Hz band-limited streamer array, on the SEG-Y's own 0.5 ms grid.
# Named `_bl` rather than `_freq`: `_freq` now means the band-pass applied on
# the common 1 ms geometry grid at the far end of the pipeline.  The file does
# not currently exist - regenerate it from str_data_raw.npy if this has to be
# re-run.
INPUT_NPY = os.path.join("data", "pohang_shore", "numpy", "str_data_bl.npy")
OUTPUT_NPY = os.path.join("data", "pohang_shore", "numpy", "str_data_intp.npy")

# Shots to process: None means all of them.
SHOT_START = 0
SHOT_STOP = None

# Samples to process: None means the whole record.
T_START = 0
T_SAMPLES = None

# Hyperparameters, as chosen on the 220 Hz filtered data.  See intp_sweep.py
# for what each one does and why these values.
THRESHOLD = 0.05
ALIAS_SCALES = 2
REPLACE_OBSERVED = False
STAGE_FACTOR = 2
N_STAGES = 2
NBSCALES = 4
NBANGLES_COARSE = 16
FINEST = 1
IS_REAL_FORWARD = 0
IS_REAL_INVERSE = 0
MU = 0.0
ALPHA = 1.0
N_ITER = 10
BLOCK_SAMPLES = 400
BLOCK_OVERLAP = 0.5
TRACE_PAD = 16
TRACE_TAPER = 1.0

# Worker processes.  The transforms are single-threaded, so this is close to a
# linear speed-up; leave headroom for the rest of the machine.  1 runs inline,
# which is the easier setting to debug with.
N_WORKERS = 12

# Redo shots that are already marked finished.
FORCE = False

# QC figure drawn from the finished file: this many shots, evenly spaced.
# None to skip.
QC_FIGURE = os.path.join("data", "pohang_shore", "interpolate_qc.png")
QC_SHOTS = 4
CLIP_PERC = 99.0

# ---------------------------------------------------------------------------


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(get_project_root(), path)


def configure():
    """Push the settings above into the interpolation module.

    The workers are spawned on Windows, so each child re-imports intp_sweep
    with its own file defaults; setting them explicitly here means the run does
    not silently depend on what that file happens to hold.
    """
    S.THRESHOLD_LIST = [THRESHOLD]
    S.ALIAS_SCALES_LIST = [ALIAS_SCALES]
    S.REPLACE_OBSERVED = REPLACE_OBSERVED
    S.STAGE_FACTOR = STAGE_FACTOR
    S.N_STAGES = N_STAGES
    S.NBSCALES = NBSCALES
    S.NBANGLES_COARSE = NBANGLES_COARSE
    S.FINEST = FINEST
    S.IS_REAL_FORWARD = IS_REAL_FORWARD
    S.IS_REAL_INVERSE = IS_REAL_INVERSE
    S.MU = MU
    S.ALPHA = ALPHA
    S.N_ITER = N_ITER
    S.BLOCK_SAMPLES = BLOCK_SAMPLES
    S.BLOCK_OVERLAP = BLOCK_OVERLAP
    S.TRACE_PAD = TRACE_PAD
    S.TRACE_TAPER = TRACE_TAPER


_CTX = {}


def _init_worker(in_path, out_path, t0, t1):
    configure()
    _CTX["src"] = np.load(in_path, mmap_mode="r")
    _CTX["dst"] = np.load(out_path, mmap_mode="r+")
    _CTX["t0"] = t0
    _CTX["t1"] = t1


def _run_shot(shot):
    gather = np.asarray(_CTX["src"][shot, _CTX["t0"]:_CTX["t1"]], dtype=float)
    out, residuals = S.cascade(gather, THRESHOLD, ALIAS_SCALES)
    _CTX["dst"][shot] = out.astype(np.float32)
    return shot, residuals[-1]


def fmt_hms(seconds):
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


def qc_figure(out_path, src, shots, t0, t1, path):
    dst = np.load(out_path, mmap_mode="r")
    fig, axes = plt.subplots(2, len(shots), figsize=(4 * len(shots), 9),
                             squeeze=False)
    for j, shot in enumerate(shots):
        g = np.asarray(src[shot, t0:t1], dtype=float)
        clip = np.percentile(np.abs(g), CLIP_PERC)
        axes[0][j].imshow(g, cmap="gray", vmin=-clip, vmax=clip, aspect="auto",
                          interpolation="nearest")
        axes[0][j].set_title(f"shot {shot} in ({g.shape[1]} traces)", fontsize=10)
        axes[1][j].imshow(np.asarray(dst[shot]), cmap="gray", vmin=-clip,
                          vmax=clip, aspect="auto", interpolation="nearest")
        axes[1][j].set_title(f"out ({dst.shape[2]} traces)", fontsize=10)
        axes[1][j].set_xlabel("trace")
    axes[0][0].set_ylabel("sample")
    axes[1][0].set_ylabel("sample")
    fig.suptitle(f"curvelet interpolation x{STAGE_FACTOR ** N_STAGES}, "
                 f"threshold {THRESHOLD:.0%}, alias {ALIAS_SCALES}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    configure()
    in_path = resolve(INPUT_NPY)
    if not os.path.isfile(in_path):
        raise SystemExit(f"not found: {in_path}")
    out_path = resolve(OUTPUT_NPY)
    done_path = os.path.splitext(out_path)[0] + ".done.npy"

    src = np.load(in_path, mmap_mode="r")
    n_shots, nt, nx_in = src.shape
    t0 = T_START
    t1 = nt if T_SAMPLES is None else min(T_START + T_SAMPLES, nt)
    nt_out = t1 - t0
    nx_out = nx_in
    for _ in range(N_STAGES):
        nx_out = (nx_out - 1) * STAGE_FACTOR + 1

    stop = n_shots if SHOT_STOP is None else min(SHOT_STOP, n_shots)
    shots = list(range(SHOT_START, stop))

    print(f"{in_path}\n  {n_shots} shots x {nt} samples x {nx_in} traces")
    print(f"  -> {len(shots)} shots x {nt_out} samples x {nx_out} traces")
    S.report_stages(nx_in)
    n_blocks = len(S.block_starts(nt_out)[0])
    print(f"  {n_blocks} blocks per stage, threshold {THRESHOLD:.0%}, "
          f"alias {ALIAS_SCALES}, replace {REPLACE_OBSERVED}, "
          f"trace pad {TRACE_PAD}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if os.path.isfile(out_path) and not FORCE:
        dst = np.load(out_path, mmap_mode="r+")
        if dst.shape != (n_shots, nt_out, nx_out):
            raise SystemExit(
                f"{out_path} exists with shape {dst.shape}, expected "
                f"{(n_shots, nt_out, nx_out)}. Delete it or set FORCE = True.")
    else:
        dst = np.lib.format.open_memmap(
            out_path, mode="w+", dtype=np.float32,
            shape=(n_shots, nt_out, nx_out))
        dst.flush()
    print(f"  output {out_path}  ({dst.nbytes / 1e9:.2f} GB)")

    if os.path.isfile(done_path) and not FORCE:
        done = np.load(done_path)
        if done.shape != (n_shots,):
            done = np.zeros(n_shots, dtype=bool)
    else:
        done = np.zeros(n_shots, dtype=bool)

    todo = [s for s in shots if not done[s]]
    if len(todo) < len(shots):
        print(f"  {len(shots) - len(todo)} shots already finished, "
              f"{len(todo)} to go")
    if not todo:
        print("  nothing to do")
    else:
        del dst
        t_start = time.time()
        completed = 0
        args = (in_path, out_path, t0, t1)
        if N_WORKERS <= 1:
            _init_worker(*args)
            results = map(_run_shot, todo)
            for shot, res in results:
                done[shot] = True
                completed += 1
                report(shot, res, completed, len(todo), t_start, done, done_path)
        else:
            with mp.Pool(N_WORKERS, initializer=_init_worker,
                         initargs=args) as pool:
                for shot, res in pool.imap_unordered(_run_shot, todo,
                                                     chunksize=1):
                    done[shot] = True
                    completed += 1
                    report(shot, res, completed, len(todo), t_start, done,
                           done_path)
        np.save(done_path, done)
        print(f"\n  finished in {fmt_hms(time.time() - t_start)}")

    if QC_FIGURE and QC_SHOTS:
        picks = list(np.linspace(SHOT_START, stop - 1, QC_SHOTS).astype(int))
        p = resolve(QC_FIGURE)
        qc_figure(out_path, src, picks, t0, t1, p)
        print(f"  wrote {p}")


def report(shot, res, completed, total, t_start, done, done_path):
    elapsed = time.time() - t_start
    eta = elapsed / completed * (total - completed)
    print(f"  [{completed}/{total}] shot {shot:4d} residual {res:.4f}  "
          f"elapsed {fmt_hms(elapsed)}, eta {fmt_hms(eta)}")
    if completed % 25 == 0:
        np.save(done_path, done)


if __name__ == "__main__":
    main()
