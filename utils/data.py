import os
# import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import FortranFile
#import pylops
#from pylops.utils.wavelets import ricker
from pathlib import Path


def npy_shape(path):
    """Shape of a .npy file, read from its header without mapping it.

    np.load(..., mmap_mode="r").shape leaves a mapping alive until the garbage
    collector gets to it, and on Windows a mapped file cannot be truncated -
    which is exactly what happens next when the caller overwrites it.  This
    reads the header through an ordinary handle and closes it.
    """
    readers = {(1, 0): np.lib.format.read_array_header_1_0,
               (2, 0): np.lib.format.read_array_header_2_0}
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        reader = readers.get(version)
        if reader is None:
            raise ValueError(f"unsupported .npy version {version} in {path}")
        shape, _, _ = reader(f)
    return shape


def create_memmap(path, shape, dtype=np.float32, attempts=6, pause=2.0):
    """np.lib.format.open_memmap(mode="w+"), retried, and unlinking first.

    open_memmap writes the .npy header, closes the file and reopens it to map.
    Two things go wrong on Windows.  Something can hold the new file open in
    between - a virus scanner picking it up is the usual culprit - and the
    reopen raises PermissionError even though the directory is writable.  And
    an existing file that is still mapped anywhere cannot be truncated, which
    surfaces as OSError EINVAL rather than anything more descriptive.

    So: remove the old file first, and retry either error after a pause.  Both
    have shown up on the first creation of a large output and gone away on the
    next attempt.
    """
    import time

    for attempt in range(1, attempts + 1):
        try:
            if os.path.isfile(path):
                os.remove(path)
        except OSError:
            pass                      # let open_memmap report it
        try:
            return np.lib.format.open_memmap(path, mode="w+", dtype=dtype,
                                             shape=shape)
        except (PermissionError, OSError) as exc:
            if attempt == attempts:
                raise
            print(f"  {type(exc).__name__} creating {os.path.basename(path)} "
                  f"(attempt {attempt}/{attempts}), retrying in {pause:g}s")
            time.sleep(pause)


class ToNumpy:
    def __call__(self, sample):
        return np.array(sample)


def get_project_root():
    return str(Path(__file__).parent.parent)


def read_binary(data_path, n1):
    with open(data_path, "rb") as file:
        field = np.fromfile(file, dtype=np.float32)
    data = field.reshape(-1, n1)
    data = data.transpose()
    return data


def load_data_from_file(data_path, n1, n2):
    f = open(data_path, "rb")
    data = np.zeros((n2, n1), dtype=np.float32)
    for ir in range(n1):
        f.seek(4 * n2 * ir)
        data[:, ir] = np.fromfile(f, dtype=np.float32, count=n2)
    f.close()
    maxval = np.max(data)
    minval = np.min(data)
    result = np.clip(data, minval, maxval)
    return result


def save_data(data, name):
    f = FortranFile(name, "w")
    f.write_record(data.transpose())


def normalize(data):
    result = (data - np.min(data)) / data.ptp()
    return result


def mask_using_threshold(data, threshold):
    mask = (data >= threshold).astype(np.float32)
    mask = np.max(mask, axis=1)
    start_index = np.argmax(mask)
    return data[start_index:, :]


def split(data, stride=100, nr=238):
    patches = []
    h, w = data.shape
    for start_index in range(0, h - nr, stride):
        patch = data[start_index:start_index + nr, :]
        patches.append(patch)
    patch = data[-nr:, :]
    patches.append(patch)
    return patches


def merge(patches, h, stride=100, nr=238):
    patch_index = 0
    data = np.zeros((h, nr), dtype=np.uint8)
    for start_index in range(0, h - nr, stride):
        data[start_index:start_index + nr, :] = patches[patch_index]
        patch_index += 1
    data[-nr:, :] = patches[-1]
    return data


def get_ae(data, aeparam):
    n0 = data.shape[0]
    d0 = aeparam[0]
    o0 = aeparam[1]
    s0 = -o0 * d0
    e0 = (n0 - 1 - o0) * d0

    n1 = data.shape[1]
    d1 = aeparam[2]
    o1 = aeparam[3]
    s1 = -o1 * d1
    e1 = (n1 - 1 - o1) * d1

    extent = [s1, e1, e0, s0]
    aspect = n0 * (e1 - s1) / (n1 * (e0 - s0))
    return extent, aspect


def show_2d_data(data, cval=None, perc=1.0, aeparam=None):
    extent = None
    aspect = None
    if aeparam is not None:
        extent, aspect = get_ae(data, aeparam)
    if cval is not None:
        maxval = cval[0]
        minval = cval[1]
    else:
        maxval = np.quantile(data, perc)
        minval = np.quantile(data, 1 - perc)
    plt.imshow(data, cmap="gray", vmax=maxval, vmin=minval, extent=extent, aspect=aspect)
    plt.show()


def show_trace(trace, sampling_rate):
    sampling = np.linspace(0, (trace.shape[0] - 1) * sampling_rate, trace.shape[0])
    plt.plot(sampling, trace)
    plt.show()


def show_2d_array(data, scale=100, is_adjust_top=False, cmap="seismic", vmin=-1.0, vmax=1.0, is_show=True, **kwargs):
    fig = plt.figure()
    top_cut = 0.384 if is_adjust_top else 0.0
    fig.set_size_inches((data.shape[1] / scale, data.shape[0] / scale - top_cut))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
    if is_show:
        plt.show()


class ValueTracker:
    def __init__(self, ema_coeff):
        self.ema_coeff = ema_coeff
        self.cur_value = 0.0
        self.bias = 1.0

    def initialize(self):
        self.__init__(self.ema_coeff)

    def feed(self, value):
        self.cur_value = self.ema_coeff * self.cur_value + (1.0 - self.ema_coeff) * value
        self.bias = self.ema_coeff * self.bias

    def val(self):
        return self.cur_value / (1 - self.bias) if self.bias < 1.0 else 0.0

