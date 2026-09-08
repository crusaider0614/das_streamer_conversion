"""Receiver gathers from the pohang_shore pair, for unpaired DAS -> streamer.

One sample is one patch of one RECEIVER gather:

    das  data/pohang_shore/das_data_rg_train.npy   (24,  2000, 551)
    str  data/pohang_shore/str_data_rg_train.npy   (24,  2000, 551)
    das  data/pohang_shore/das_data_rg_infer.npy   (240, 2000, 551)

Axis order is (receiver, sample, shot), written by process/logenv_process.py,
and the shot axis is already ordered along the survey line.  So `idx` selects
a receiver and the crop takes a window of (time, shot).

Why receiver gathers rather than shot gathers
---------------------------------------------
A shot gather has only 24 traces on the streamer side, which is too narrow to
train on.  A receiver gather has 551 - every shot in the survey - and both
domains have the same 551, so one CROP_SIZE means the same thing on both
sides.  The shot axis is sorted along the line, which takes the shot interval
from 15.24 m down to a median of 3.33 m; unsorted it is four interleaved
passes and neighbouring columns are hundreds of shots apart.

Irregular shot spacing is left alone deliberately.  It is shared by the two
domains at every shot, so it cannot tell them apart and gives the
discriminator nothing to latch onto; and the axis is aliased over 60 % of its
gaps, so interpolating it would invent rather than reconstruct.

`rg_train` against `rg_infer`
-----------------------------
The streamer array covers 71.3 m of the DAS's 793 m of fibre.  `rg_train` is
the DAS receivers inside that shared aperture - the only ones with a streamer
counterpart to be trained against - and `rg_infer` is the remaining 240, which
the model can be run on but not trained against.  Pass stage="rg_infer" for
those.

The mute
--------
The direct-arrival mute leaves the top of every trace exactly zero: 28.5 % of
the DAS record on average and 37.0 % of the streamer's.  A random time crop
can therefore land entirely in it, so MIN_LIVE rejects patches that are mostly
zeros - without that a third of the batch is empty.
"""

import os
import random

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

from utils.data import get_project_root

# Minimum fraction of a patch that has to be non-zero for it to be used.  The
# mute makes the top of the record exactly zero, so a crop that lands there is
# all zeros and teaches nothing.
MIN_LIVE = 0.35

# Attempts to find a patch that clears MIN_LIVE before giving up and returning
# the best of them.  The window is 512 of 2000 samples and the mute reaches
# 1367 ms at worst, so a few tries is normally enough.
MAX_TRIES = 20

# Receivers the clip percentiles are measured over; None uses all of them.
# The percentile needs the samples in memory, so this bounds that read
# independently of how big the array is.
CLIP_RECEIVERS = None


class PohangShoreDataset(Dataset):
    def __init__(
            self,
            is_das=True,
            crop_size=None,
            total_length=1,
            is_flip=False,
            is_negative=False,
            noise=0.0,
            is_train=True,
            clip=99,
            stage="rg_train",
    ):
        self.is_das = is_das
        self.total_length = total_length
        self.is_flip = is_flip
        self.is_negative = is_negative
        self.noise = noise
        self.is_train = is_train

        name = ("das" if is_das else "str") + "_data_" + stage + ".npy"
        path = os.path.join(get_project_root(), "data", "pohang_shore", name)
        if not os.path.isfile(path):
            raise SystemExit(
                f"not found: {path}\n"
                f"run process/logenv_process.py with WRITE_ALL on to build "
                f"the receiver-major arrays")
        self.data = np.load(path, mmap_mode="r")
        if self.data.ndim != 3:
            raise SystemExit(f"{name} is {self.data.shape}; expected "
                             f"(receiver, sample, shot)")
        self.n_data, self.n_samp, self.n_shot = self.data.shape

        # Clip bounds from a subsample of receivers rather than the whole
        # array: np.percentile has to hold what it is given, and this keeps
        # that bounded.  Clipping itself is elementwise, so it is applied per
        # patch in __getitem__ - the same values as clipping the array up
        # front, without materialising it.
        idx = np.arange(self.n_data)
        if CLIP_RECEIVERS is not None and CLIP_RECEIVERS < self.n_data:
            idx = np.unique(np.linspace(0, self.n_data - 1,
                                        CLIP_RECEIVERS).astype(int))
        sample = np.asarray(self.data[idx], dtype=np.float32)
        self.lower_clip = float(np.percentile(sample, (100 - clip) / 2))
        self.upper_clip = float(np.percentile(sample, 100 - (100 - clip) / 2))
        self.value_range = max(self.upper_clip, -self.lower_clip) / 2.0
        if self.value_range <= 0:
            raise SystemExit(f"{name}: clip range is degenerate "
                             f"[{self.lower_clip}, {self.upper_clip}]")
        del sample

        if crop_size is None:
            self.is_crop = False
            self.crop_size = (self.n_samp, self.n_shot)
        else:
            assert len(crop_size) == 2
            self.is_crop = True
            self.crop_size = (min(crop_size[0], self.n_samp),
                              min(crop_size[1], self.n_shot))

    def __len__(self):
        return self.total_length

    def live_fraction(self, patch):
        return float(np.count_nonzero(patch)) / patch.size

    def crop(self, gather):
        """A random (time, shot) window, preferring one that is not all mute.

        The time start is jittered slightly past both ends and clipped back,
        which keeps the first and last windows reachable instead of half as
        likely as the interior ones.
        """
        nt, nx = self.crop_size
        best, best_live = None, -1.0
        for _ in range(MAX_TRIES if self.is_train else 1):
            mt = self.n_samp // 500
            st = random.randint(-mt, self.n_samp - nt + mt)
            st = int(np.clip(st, 0, self.n_samp - nt))
            sx = random.randint(0, self.n_shot - nx)
            patch = gather[st:st + nt, sx:sx + nx]
            live = self.live_fraction(patch)
            if live > best_live:
                best, best_live = patch, live
            if live >= MIN_LIVE:
                return patch
        return best

    def __getitem__(self, idx):
        gather = self.data[idx % self.n_data]
        datum = (self.crop(gather) if self.is_crop
                 else np.asarray(gather)).astype(np.float32).copy()

        if self.is_flip and random.random() < 0.5:
            # mirrors the shot axis, which reverses the direction the line was
            # sailed in - a real acquisition either way
            datum = np.flip(datum, axis=1)

        if self.is_negative and random.random() < 0.5:
            datum = -datum

        np.clip(datum, self.lower_clip, self.upper_clip, out=datum)
        datum = datum / self.value_range

        if self.noise != 0.0:
            datum = datum + (self.noise * np.random.randn(*datum.shape)
                             ).astype(np.float32)

        return np.ascontiguousarray(datum)[None]


if __name__ == "__main__":
    crop_size = (512, 72)
    A_dataset = PohangShoreDataset(
        is_das=True, crop_size=crop_size, total_length=20,
        is_flip=False, is_negative=False, noise=0.0, is_train=True)
    B_dataset = PohangShoreDataset(
        is_das=False, crop_size=crop_size, total_length=20,
        is_flip=False, is_negative=False, noise=0.0, is_train=True)
    print(f"A das {A_dataset.data.shape}  crop {A_dataset.crop_size}  "
          f"clip [{A_dataset.lower_clip:.4g}, {A_dataset.upper_clip:.4g}]  "
          f"value_range {A_dataset.value_range:.4g}")
    print(f"B str {B_dataset.data.shape}  crop {B_dataset.crop_size}  "
          f"clip [{B_dataset.lower_clip:.4g}, {B_dataset.upper_clip:.4g}]  "
          f"value_range {B_dataset.value_range:.4g}")

    for i in range(0, 20, 5):
        a = A_dataset[i].squeeze()
        b = B_dataset[i].squeeze()
        print(f"  [{i}] das {a.shape} live {A_dataset.live_fraction(a):.2f} "
              f"range {a.min():+.3f}..{a.max():+.3f}   "
              f"str {b.shape} live {B_dataset.live_fraction(b):.2f} "
              f"range {b.min():+.3f}..{b.max():+.3f}")
        fig, axs = plt.subplots(1, 2, figsize=(11, 6))
        for ax, img, name in ((axs[0], a, "das"), (axs[1], b, "str")):
            ax.imshow(img, cmap="seismic", vmin=-1.0, vmax=1.0, aspect="auto")
            ax.set_title(f"{name} receiver {i % A_dataset.n_data}")
            ax.set_xlabel("shot (sorted along the line)")
        axs[0].set_ylabel("sample")
        plt.tight_layout()
        plt.show()
