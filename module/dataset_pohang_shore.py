import os
import random

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

from utils.data import get_project_root


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
    ):
        self.is_das = is_das
        self.total_length = total_length
        self.is_flip = is_flip
        self.is_negative = is_negative
        self.noise = noise
        self.is_train = is_train

        # `_norm` is RMS-normalised only, no envelope gain; `_log` is the same
        # thing with calculate_logscale applied as well.  Both are (551, 3500,
        # nx), so the [:, :4000] slice below is a no-op kept from when the
        # records were longer.
        stage = "norm"
        name = ("das" if is_das else "str") + "_data_" + stage + ".npy"
        self.data = np.load(os.path.join(get_project_root(), "data",
                                         "pohang_shore", name),
                            mmap_mode="r")[:, :4000]
        self.n_data = self.data.shape[0]

        lower_clip = np.percentile(self.data, (100 - clip) / 2)
        upper_clip = np.percentile(self.data, 100 - (100 - clip) / 2)

        if self.is_train:
            self.data = np.clip(self.data, lower_clip, upper_clip)
            self.value_range = np.max(np.abs(self.data)) / 2.0
        else:
            self.value_range = max(upper_clip, -lower_clip) / 2.0

        if crop_size is None:
            self.is_crop = False
        else:
            assert len(crop_size) == 2
            self.is_crop = True
            self.crop_size = (crop_size[0], crop_size[1])

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if self.is_das:
            if self.is_train:
                datum = self.data[idx % self.n_data, :, 44: 116].copy()
            else:
                datum = self.data[idx % self.n_data].copy()
        else:
            datum = self.data[idx % self.n_data].copy()

        nt, dx = datum.shape
        if self.is_crop:
            mt = nt // 500
            st = random.randint(-mt, nt - self.crop_size[0] + mt)
            st = np.clip(st, 0, nt - self.crop_size[0])
            et = st + self.crop_size[0]
            datum = datum[st: et]

        if self.is_flip:
            check_value = random.random()
            if check_value < 0.5:
                datum = np.flip(datum, axis=1)

        if self.is_negative:
            check_value = random.random()
            if check_value < 0.5:
                datum = -datum

        datum = datum / self.value_range

        if self.noise != 0.0:
            datum += self.noise * np.random.randn(datum.shape[0], datum.shape[1]).astype(np.float32)

        return datum[None]


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    batch_size = 1
    crop_size = None
    A_dataset = PohangShoreDataset(
        is_das=True,
        crop_size=crop_size,
        total_length=20,
        is_flip=False,
        is_negative=False,
        noise=0.0,
        is_train=False,
    )
    B_dataset = PohangShoreDataset(
        is_das=False,
        crop_size=crop_size,
        total_length=20,
        is_flip=False,
        is_negative=False,
        noise=0.0,
        is_train=True,
    )

    for i in range(0, 280, 10):
        A_numpy = A_dataset.__getitem__(i).squeeze()
        B_numpy = B_dataset.__getitem__(i).squeeze()

        print(A_numpy.shape, B_numpy.shape)
        print(A_numpy.max(), A_numpy.min())
        print(B_numpy.max(), B_numpy.min())

        nt, nx = A_numpy.shape

        hb = -np.ones((nt, 2))

        fig, axs = plt.subplots(1, 2)

        axs[0].imshow(A_numpy, cmap='seismic', vmin=-1.0, vmax=1.0, aspect='auto')
        axs[1].imshow(B_numpy, cmap='seismic', vmin=-1.0, vmax=1.0, aspect='auto')

        plt.show()
