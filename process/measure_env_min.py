"""The smallest trace envelope in a whole array, for config.ENV_MIN.

    numpy/{das_data_deci,str_data_line}.npy  ->  printed, paste into config

calculate_logscale subtracts `data_env_log.min()` from the log envelope, and
that single number decides where its gain peaks:

    peak envelope = e * (env_min + log_base) - log_base

Left to itself it is the minimum of whatever array happens to be handed in, so
the gain a trace receives depends on which gather it was fitted in - a shot
gather, a receiver gather, one shot or all of them.  The fix is to measure the
minimum once over the whole (shot x receiver) dataset and pass it in; then
every route agrees by construction rather than by coincidence.

What is measured here is the ENVELOPE minimum, not the log-envelope minimum,
because the envelope does not depend on log_base: the caller forms
log10(env_min + log_base) itself, so one number per domain serves every
log_base.  utils.process.calculate_logscale takes it as `env_min`.

Deliberately measured WITHOUT logenv_process.PAD_TAIL.  The pad is zeros, so
its own envelope is near zero and would make the minimum meaningless - the
number wanted here is the quietest sample of the real record.

Edit the settings block below, then run from the repository root:

    python -m process.measure_env_min
"""

import os
import time

import numpy as np

import config as C
from utils.data import get_project_root
from utils.process import envelope_1d

# ---------------------------------------------------------------- settings --

# (tag, stage) to measure - the arrays logenv_process reads.
JOBS = (("das", "norm"), ("str", "norm"))

# Shots to read; None reads every one.  The minimum is a minimum, so a
# subsample can only overestimate it - use None for the number that goes into
# config.
SHOTS = None

# ---------------------------------------------------------------------------


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(get_project_root(), path)


def main():
    out = {}
    for tag, stage in JOBS:
        rel = C.ARRAYS[tag][stage]
        p = resolve(rel)
        if not os.path.isfile(p):
            print(f"{tag}: {rel} missing - skipped\n")
            continue
        data = np.load(p, mmap_mode="r")
        n_shots = data.shape[0]
        idx = range(n_shots) if SHOTS is None else \
            np.unique(np.linspace(0, n_shots - 1, SHOTS).astype(int))
        print(f"{tag}: {rel}  {data.shape}, {len(idx)} shots")

        lo = np.inf
        where = None
        # a running histogram of the low tail, to show whether the minimum is
        # a lone outlier or the floor of a population
        tail = []
        t0 = time.time()
        for k, i in enumerate(idx):
            e = envelope_1d(np.asarray(data[i], dtype=np.float32))
            m = float(e.min())
            if m < lo:
                lo = m
                where = (int(i), *[int(v) for v in
                                   np.unravel_index(e.argmin(), e.shape)])
            tail.append(np.percentile(e, [0.01, 0.1, 1.0]))
            if (k + 1) % 100 == 0 or k + 1 == len(idx):
                el = time.time() - t0
                print(f"    {k + 1}/{len(idx)} shots  {el:.0f}s", flush=True)
        tail = np.array(tail)
        print(f"  min envelope {lo:.6e}  at shot {where[0]}, "
              f"sample {where[1]}, trace {where[2]}")
        print(f"  per-shot low tail, median over shots: "
              f"p0.01 {np.median(tail[:, 0]):.3e}, "
              f"p0.1 {np.median(tail[:, 1]):.3e}, "
              f"p1 {np.median(tail[:, 2]):.3e}")
        base = C.LOG_SCALE_PARAMS[tag]["log_base"]
        print(f"  with log_base {base:g}:  log10(env_min + log_base) = "
              f"{np.log10(lo + base):.9f}   against log10(log_base) = "
              f"{np.log10(base):.9f}")
        print(f"    gain peaks at env = e * (env_min + base) - base = "
              f"{np.e * (lo + base) - base:.6e}\n")
        out[tag] = lo

    if out:
        print("paste into config.py:\n")
        print("ENV_MIN = {" + ", ".join(f'"{k}": {v!r}'
                                        for k, v in out.items()) + "}")


if __name__ == "__main__":
    main()
