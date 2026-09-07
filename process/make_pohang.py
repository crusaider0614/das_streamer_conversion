import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import random
import time
from itertools import product, accumulate
from scipy.ndimage import convolve, gaussian_filter, sobel
from scipy import signal
from utils.data import get_project_root, show_2d_array
from joblib import Parallel, delayed


layer_ratio_avg = 0.03
layer_ratio_max = 0.05
sqrt_pi = np.sqrt(2 * np.pi)


def gaussian(x, mu, sig):
    return np.exp(-((x - mu) * (x - mu)) / (2 * sig * sig)) / (sig * sqrt_pi)


def fill_layer_gaussian(reflection, layer, value, width):
    nz, ny, nx = reflection.shape
    gradient = np.sqrt(sobel(layer, axis=0)**2 + sobel(layer, axis=1)**2)
    # gradient = gaussian_filter(gradient, sigma=1.0)
    min_val = 5.0
    max_val = 100.0
    gradient = np.clip(gradient, min_val, max_val)
    # gradient = np.log10(gradient)
    min_val = gradient.min()
    max_val = gradient.max()

    reflection_width = 1.0 + (6.0 / (max_val - min_val + 1e-10)) * (gradient - min_val)
    reflection_sigma = reflection_width * 0.4
    reflection_strength = value / reflection_width

    # reflection_width = width * np.ones_like(gradient)
    # reflection_sigma = reflection_width * 0.4
    # reflection_strength = value * np.ones_like(reflection_width) / width

    reflection_idx = np.zeros((ny, nx, 2), dtype=np.int32)
    reflection_idx[:, :, 0] = np.clip(np.floor(layer - reflection_width - 0.001), 0, nz - 1).astype(np.int32)
    reflection_idx[:, :, 1] = np.clip(np.ceil(layer + reflection_width + 0.001), 0, nz - 1).astype(np.int32)

    for iy, ix in product(range(ny), range(nx)):
        a = gaussian(
            np.linspace(reflection_idx[iy, ix, 0], reflection_idx[iy, ix, 1], reflection_idx[iy, ix, 1] - reflection_idx[iy, ix, 0] + 1),
            layer[iy, ix],
            reflection_sigma[iy, ix],
        )
        reflection[reflection_idx[iy, ix, 0]: reflection_idx[iy, ix, 1] + 1, iy, ix] = (
            reflection[reflection_idx[iy, ix, 0]: reflection_idx[iy, ix, 1] + 1, iy, ix] +
            reflection_strength[iy, ix] * a
        )
    return reflection


def fill_layer(reflection, layer, value):
    ny, nx = layer.shape
    layerf = np.floor(layer).astype(np.int32)
    layerc = np.ceil(layer).astype(np.int32)
    interp = layer - layerf
    interp[interp == 0] = 1.0
    for idx in iter((layerf[iy, ix], iy, ix) for iy, ix in product(range(ny), range(nx))):
        reflection[idx] = value * (1 - interp[idx[1:]])
    for idx in iter((layerc[iy, ix], iy, ix) for iy, ix in product(range(ny), range(nx))):
        reflection[idx] = value * interp[idx[1:]]
    return reflection


def interpolate_reflection(reflection, top_layer, bottom_layer, width):
    layer_width = np.abs(bottom_layer - top_layer)

    avg_range = int(np.floor((layer_width.mean() * 0.8 + layer_width.max() * 0.2) * layer_ratio_avg))
    max_range = int(np.ceil((layer_width.mean() * 0.5 + layer_width.max() * 0.5) * layer_ratio_max))
    layer_num = random.randrange(avg_range, max_range) if max_range > avg_range else avg_range

    # int random distribution
    rand_distribution = 0.5 + 0.5 * np.random.rand(layer_num)
    rand_distribution = rand_distribution / np.sum(rand_distribution)

    new_layers = np.zeros((0, top_layer.shape[0], top_layer.shape[1]), dtype=np.float32)
    for rate in accumulate(rand_distribution[:-1]):
        layer = top_layer + rate * (bottom_layer - top_layer)
        new_layers = np.append(new_layers, layer[None, :, :], axis=0)
        value = 0.1 + 0.4 * random.random()
        if random.random() > 0.5:
            value = -value
        fill_layer_gaussian(reflection, layer, value, width)
    return new_layers


def construct_reflection(reflection, layers):
    new_layers = np.zeros((0, layers.shape[1], layers.shape[2]), dtype=np.float32)
    for i in range(len(layers) - 1):
        if i == len(layers) - 3:
            width = 2.5
        elif i == len(layers) - 2:
            width = 4.0
        else:
            width = 1.0
        new_sublayers = interpolate_reflection(reflection, layers[i], layers[i + 1], width)
        new_layers = np.append(new_layers, new_sublayers, axis=0)
    for i in range(len(layers)):
        if i == len(layers) - 3:
            width = 2.5
        elif i == len(layers) - 2:
            width = 4.0
        else:
            width = 1.0
        value = 0.4 + 0.6 * random.random()
        if random.random() > 0.5:
            value = -value
        fill_layer_gaussian(reflection, layers[i], value, width)
        new_layers = np.append(new_layers, layers[i: i + 1], axis=0)
    return new_layers


def fdgaus(sig, mu, nt):
    x = np.linspace(0, nt - 1, nt)
    tmp = 1 / (2 * np.pi * sig**2)**0.5 * (-(x - mu) / sig**2) * np.exp(-(x - mu)**2 / (2 * sig**2))
    tmp /= tmp.max()
    return tmp


def convolvesource(reflection, source_len, source_co):
    kernel = signal.ricker(source_len, source_co)
    kernel = np.expand_dims(kernel, axis=1)
    kernel = np.expand_dims(kernel, axis=1)
    section = convolve(reflection, kernel)
    return section


def mute(data, ct, wt, is_up=True):
    if is_up:
        for it in range(ct):
            a = max(0.0, (it - ct) / wt + 1.0)
            data[:, it] = a * data[:, it]
    else:
        nt = data.shape[1]
        for it in range(ct, nt):
            a = max(0.0, -(it - ct) / wt + 1.0)
            data[:, it] = a * data[:, it]
    return data


def plot_layer(data, strata):
    for iy in range(25, 950, 50):
        print(iy)
        strata = np.clip(strata, 0, data.shape[0] - 1)

        a = data[:, iy, :]
        plt.figure()
        m1 = np.maximum(np.abs(np.percentile(a, 1)), np.abs(np.percentile(a, 99)))
        plt.imshow(a, cmap="gray", vmin=-m1, vmax=m1, aspect=0.5)
        b = strata[:, iy, :]
        for i in range(len(strata)):
            plt.plot(range(data.shape[2]), b[i, :], alpha=1, label=i)
        plt.show()


def plot_spot(data, spots):
    m1 = np.maximum(np.abs(np.percentile(data, 0)), np.abs(np.percentile(data, 100)))
    plt.imshow(data, cmap="gray", vmin=-m1, vmax=m1, aspect=1.0)
    active_spots = spots.copy()
    print(active_spots)
    plt.scatter(active_spots[:, 1], active_spots[:, 2], alpha=1, label=i, s=1)
    plt.show()


def interpolate_spots(spots):
    interpolated_spots = np.zeros((0, 3), dtype=np.float32)
    for px, py, pz in spots:
        condition = (
            (spots[:, 0] >= px) &
            (spots[:, 0] < px + 1.0) &
            (spots[:, 1] >= py) &
            (spots[:, 1] < py + 1.0)
        )
        near_spots = spots[condition]
        for near_spot in near_spots:
            if not (near_spot[0] == px and near_spot[1]) == py:
                new_spot = (near_spot + [px, py, pz]) / 2.0
                interpolated_spots = np.append(interpolated_spots, new_spot[None, :], axis=0)
    return interpolated_spots


def gaussian_3d(xc, yc, zc, mx, my, mz, sig):
    return np.exp(-((xc - mx) * (xc - mx) + (yc - my) * (yc - my) + (zc - mz) * (zc - mz)) / (2 * sig * sig))


def target_process(i, shape, strata):
    reflection = np.zeros(shape)
    new_layers = construct_reflection(reflection, strata)
    print(i, new_layers.shape)
    # new_layers = np.transpose(new_layers, (2, 0, 1))

    section = np.clip(reflection, -1.0, 1.0)
    section = convolvesource(section, 21, 1.5)
    # section = np.transpose(section, (2, 0, 1))

    np.save(os.path.join(get_project_root(), "data", "synt_sparse", "sections", str(i)), section)
    np.save(os.path.join(get_project_root(), "data", "synt_sparse", "layers", str(i)), new_layers)

    return i


if __name__ == "__main__":
    def show_in_plot(imgs, perc=100):
        plt.figure()
        for i, img in enumerate(imgs):
            if i == 0 or i == 2:
                img = np.transpose(img)
            vval = max(np.percentile(img, perc), -np.percentile(img, 100 - perc))
            plt.subplot(1, 4, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img, cmap="seismic", vmin=-vval, vmax=vval)
        plt.show()

    data = np.load(os.path.join(get_project_root(), "data", "pohang", "Eastsea_3D.npy"))

    # data = mute(data, 80, 10)
    # data_tp = np.transpose(data, (0, 2, 1))
    # shape = data_tp.shape
    # nonzero_depth = np.zeros((shape[0: 2]))
    # for ix in range(shape[0]):
    #     for iy in range(shape[1]):
    #         nonzero_idx = np.nonzero(data_tp[ix, iy, 0: 200])
    #         nonzero_depth[ix, iy] = nonzero_idx[0][0]
    # nonzero_depth[nonzero_depth < 70] = 70

    nz, nx, ny = data.shape
    strata = np.load(os.path.join(get_project_root(), "data", "pohang", "Eastsea_3D_layer.npy"))
    strata = strata[[0, 1, 3, 2, 4]]
    # data = data * 15
    # for iz in range(120, 600, 20):
    #     print(iz)
    #     show_2d_array(data[iz])
    print(np.mean(strata, axis=(1, 2)))
    print(strata[0].max(), strata[0].min())
    print(strata[1].max(), strata[1].min())
    print(strata[2].max(), strata[2].min())
    print(strata[3].max(), strata[3].min())
    print(strata[4].max(), strata[4].min())

    from skimage.restoration import denoise_tv_chambolle as tv_filter
    from scipy.ndimage import gaussian_filter
    # top_layer = 43.0 * np.ones((1, nx, ny), dtype=np.float32)
    # strata = np.append(top_layer, strata, axis=0)
    # top_layer = 875.0 * np.ones((1, nx, ny), dtype=np.float32)
    # strata = np.append(strata, top_layer, axis=0)
    # strata = np.clip(strata, 43.0, 875.0)
    # for i in range(strata.shape[0] - 1):
    #     upper_layer = strata[i]
    #     lower_layer = strata[i + 1]
    #     avg_layer = (upper_layer + lower_layer) / 2.0
    #     strata[i][upper_layer > lower_layer] = avg_layer[upper_layer > lower_layer]
    #     strata[i + 1][upper_layer > lower_layer] = avg_layer[upper_layer > lower_layer]
    for i in range(len(strata)):
        strata[i] = tv_filter(strata[i], weight=1.0)
        # strata[i] = gaussian_filter(strata[i], sigma=0.2)
    # for i in range(strata.shape[0] - 1):
    #     upper_layer = strata[i].copy()
    #     lower_layer = strata[i + 1].copy()
    #     strata[i][upper_layer > lower_layer] = lower_layer[upper_layer > lower_layer]
    #     strata[i + 1][upper_layer > lower_layer] = upper_layer[upper_layer > lower_layer]

    for ix in range(nx):
        strata_ix = strata[:, ix]
        error_idxs = np.where(strata_ix[1] > strata_ix[2])
        if len(error_idxs[0]) == 0:
            continue
        error_sx = error_idxs[0].min()
        error_ex = error_idxs[0].max() + 1
        strata_ix[2, error_sx: error_ex] = strata_ix[3, error_sx: error_ex] - strata_ix[3, error_ex] + strata_ix[2, error_sx: error_ex]
        strata[:, ix] = strata_ix
        pad_height = strata_ix[2, error_sx - 1] - strata_ix[2, error_sx]
        error_px = int(pad_height / 20)
        print(ix, pad_height, error_px)
        if error_px > 1:
            for ip in range(-error_px, 0):
                strata_ix[2, error_sx + ip] = strata_ix[2, error_sx + ip] - pad_height * (error_px + ip) / error_px

    strata[1][strata[1] > strata[2]] = strata[2][strata[1] > strata[2]]
    strata[0][strata[1] > strata[2]] = strata[2][strata[1] > strata[2]]
    strata = np.concatenate((strata, 900 * np.ones((1, nx, ny))), axis=0)

    # strata[0] = 43.0
    # strata[-1] = 875.0
    # for iy in range(0, 221, 10):
    #     for i in range(len(strata)):
    #         plt.plot(range(data.shape[2]), strata[i, iy, :], alpha=1, label=i)
    #     plt.gca().invert_yaxis()
    #     plt.show()
    # data = np.transpose(data, (2, 0, 1))
    # print(data.shape)
    data = data[100: 100 + 600]
    strata = strata - 100

    np.save(os.path.join(get_project_root(), "data", "pohang", "sections", "0"), data)
    np.save(os.path.join(get_project_root(), "data", "pohang", "layers", "0"), strata)
    plot_layer(data, strata)
    exit()

    # all_fault_spots = np.zeros((0, 3), dtype=np.float32)
    #
    # fault_spots = np.zeros((0, 3), dtype=np.float32)
    # with open(os.path.join(get_project_root(), "data", "fault", "Fault interpretation 1")) as f:
    #     while True:
    #         line = f.readline()
    #         if not line: break
    #         line_data = np.array([float(x) for x in line.split()], dtype=np.float32)[None, :]
    #         fault_spots = np.append(fault_spots, line_data, axis=0)
    #     fault_spots = (fault_spots - [[1598582.0, -184984.0, -2.0]]) / [[55.0, 55.0, 4]]
    #     fault_spots[:, 1] = 270 - fault_spots[:, 1]
    #     fault_spots = np.append(fault_spots, interpolate_spots(fault_spots), axis=0)
    #     all_fault_spots = np.append(all_fault_spots, fault_spots, axis=0)
    #
    # fault_spots = np.zeros((0, 3), dtype=np.float32)
    # with open(os.path.join(get_project_root(), "data", "fault", "Fault interpretation 2")) as f:
    #     while True:
    #         line = f.readline()
    #         if not line: break
    #         line_data = np.array([float(x) for x in line.split()], dtype=np.float32)[None, :]
    #         fault_spots = np.append(fault_spots, line_data, axis=0)
    #     fault_spots = (fault_spots - [[1598582.0, -184984.0, -2.0]]) / [[55.0, 55.0, 4]]
    #     fault_spots[:, 1] = 270 - fault_spots[:, 1]
    #     fault_spots = np.append(fault_spots, interpolate_spots(fault_spots), axis=0)
    #     all_fault_spots = np.append(all_fault_spots, fault_spots, axis=0)
    #
    # fault_spots = np.zeros((0, 3), dtype=np.float32)
    # with open(os.path.join(get_project_root(), "data", "fault", "Fault interpretation 3")) as f:
    #     while True:
    #         line = f.readline()
    #         if not line: break
    #         line_data = np.array([float(x) for x in line.split()], dtype=np.float32)[None, :]
    #         fault_spots = np.append(fault_spots, line_data, axis=0)
    #     fault_spots = (fault_spots - [[1598582.0, -184984.0, -2.0]]) / [[55.0, 55.0, 4]]
    #     fault_spots[:, 1] = 270 - fault_spots[:, 1]
    #     fault_spots = np.append(fault_spots, interpolate_spots(fault_spots), axis=0)
    #     all_fault_spots = np.append(all_fault_spots, fault_spots, axis=0)
    #
    # fault_spots = np.zeros((0, 3), dtype=np.float32)
    # with open(os.path.join(get_project_root(), "data", "fault", "Fault interpretation 4")) as f:
    #     while True:
    #         line = f.readline()
    #         if not line: break
    #         line_data = np.array([float(x) for x in line.split()], dtype=np.float32)[None, :]
    #         fault_spots = np.append(fault_spots, line_data, axis=0)
    #     fault_spots = (fault_spots - [[1598582.0, -184984.0, -2.0]]) / [[55.0, 55.0, 4]]
    #     fault_spots[:, 1] = 270 - fault_spots[:, 1]
    #     fault_spots = np.append(fault_spots, interpolate_spots(fault_spots), axis=0)
    #     all_fault_spots = np.append(all_fault_spots, fault_spots, axis=0)
    #
    # fault_spots = np.zeros((0, 3), dtype=np.float32)
    # with open(os.path.join(get_project_root(), "data", "fault", "Fault interpretation 5")) as f:
    #     while True:
    #         line = f.readline()
    #         if not line: break
    #         line_data = np.array([float(x) for x in line.split()], dtype=np.float32)[None, :]
    #         fault_spots = np.append(fault_spots, line_data, axis=0)
    #     fault_spots = (fault_spots - [[1598582.0, -184984.0, -2.0]]) / [[55.0, 55.0, 4]]
    #     fault_spots[:, 1] = 270 - fault_spots[:, 1]
    #     fault_spots = np.append(fault_spots, interpolate_spots(fault_spots), axis=0)
    #     all_fault_spots = np.append(all_fault_spots, fault_spots, axis=0)
    #
    # nx = 221
    # ny = 271
    # nz = 876
    #
    # xcoordi = np.linspace(0, nx - 1, nx)[:, None, None]
    # xcoordi = np.repeat(xcoordi, ny, axis=1)
    # xcoordi = np.repeat(xcoordi, nz, axis=2)
    #
    # ycoordi = np.linspace(0, ny - 1, ny)[None, :, None]
    # ycoordi = np.repeat(ycoordi, nx, axis=0)
    # ycoordi = np.repeat(ycoordi, nz, axis=2)
    #
    # zcoordi = np.linspace(0, nz - 1, nz)[None, None, :]
    # zcoordi = np.repeat(zcoordi, nx, axis=0)
    # zcoordi = np.repeat(zcoordi, ny, axis=1)
    #
    # fault_distribution = np.zeros((nx, ny, nz), dtype=np.float32)
    # kernel_size = 7
    # for fault_spot in all_fault_spots:
    #     px = fault_spot[0]
    #     py = fault_spot[1]
    #     pz = fault_spot[2]
    #     ix_s = max(int(round(px - kernel_size // 2)), 0)
    #     iy_s = max(int(round(py - kernel_size // 2)), 0)
    #     iz_s = max(int(round(pz - kernel_size // 2)), 0)
    #     ix_e = min(ix_s + kernel_size, nx)
    #     iy_e = min(iy_s + kernel_size, ny)
    #     iz_e = min(iz_s + kernel_size, nz)
    #
    #     # new_input = np.append(fault_distribution[None, ix_s: ix_e, iy_s: iy_e, iz_s: iz_e],
    #     #     gaussian_3d(
    #     #         xcoordi[ix_s: ix_e, iy_s: iy_e, iz_s: iz_e],
    #     #         ycoordi[ix_s: ix_e, iy_s: iy_e, iz_s: iz_e],
    #     #         zcoordi[ix_s: ix_e, iy_s: iy_e, iz_s: iz_e],
    #     #         px,
    #     #         py,
    #     #         pz,
    #     #         1.0
    #     #     )[None], axis=0
    #     # )
    #     # fault_distribution[ix_s: ix_e, iy_s: iy_e, iz_s: iz_e] = np.max(new_input, axis=0)
    #
    #     new_input = gaussian_3d(
    #         xcoordi[ix_s: ix_e, iy_s: iy_e, iz_s: iz_e],
    #         ycoordi[ix_s: ix_e, iy_s: iy_e, iz_s: iz_e],
    #         zcoordi[ix_s: ix_e, iy_s: iy_e, iz_s: iz_e],
    #         px,
    #         py,
    #         pz,
    #         1.0
    #     )
    #     fault_distribution[ix_s: ix_e, iy_s: iy_e, iz_s: iz_e] += new_input
    #
    # fault_distribution = np.transpose(fault_distribution, (2, 0, 1))
    # print(np.max(all_fault_spots[:, 0]), np.min(all_fault_spots[:, 0]))
    #
    # # for section_idx in range(10, 220):
    # #     a = fault_distribution[:, section_idx]
    # #     print(all_fault_spots[:, 0])
    # #     print(all_fault_spots[:, 0] > section_idx - 2.0)
    # #     print(all_fault_spots[:, 0] < section_idx + 2.0)
    # #     condition = np.logical_and(all_fault_spots[:, 0] > section_idx - 2.0, all_fault_spots[:, 0] < section_idx + 2.0)
    # #     section_fault_spots = all_fault_spots[condition]
    # #     plot_spot(a, section_fault_spots)
    #
    # np.save(os.path.join(get_project_root(), "data", "fault", "fault_image"), fault_distribution)

    results = Parallel(n_jobs=20)(delayed(target_process)(i, shape=(nz, nx, ny), strata=strata) for i in range(40))
    print(results)

    # for i in range(1):
    #     reflection = np.zeros((nz, nx, ny))
    #     print(data.shape, reflection.shape)
    #     new_layers = construct_reflection(reflection, strata)
    #     new_layers = np.transpose(new_layers, (2, 0, 1))
    #
    #     section = np.clip(reflection, -1.0, 1.0)
    #     # section = scipy.ndimage.gaussian_filter(section, 1.0)
    #     section = convolvesource(section, 21, 1.5)
    #
    #     reflection = np.transpose(reflection, (2, 0, 1))
    #     section = np.transpose(section, (2, 0, 1))
    #     data = np.transpose(data, (2, 0, 1))
    #
    #     # for ix in range(section.shape[0]):
    #     #     for iy in range(section.shape[2]):
    #     #         section[ix, 0: int(nonzero_depth[ix, iy]), iy] = 0.0
    #
    #     for j in range(1, 10):
    #         # data = data[:, 0: 640]
    #         # last_layer = np.ceil(stratum[-1]).astype(np.int32)
    #         # print(data.shape)
    #         # for ix in range(section.shape[0]):
    #         #     for iy in range(section.shape[2]):
    #         #         for iz in range(last_layer[iy, ix], 640):
    #         #             data[ix, iz, iy] = data[ix, iz, iy] * max(0.0, 1.0 - 0.1 * (iz - last_layer[iy, ix]))
    #
    #         coeff = j / 10
    #         point = [
    #             int(round(coeff * ny)),
    #             int(round(coeff * nz)),
    #             int(round(coeff * nx)),
    #         ]
    #
    #         xz_real = data[:, :, point[2]]
    #         zy_real = data[point[0], :, :]
    #         # print(data.shape)
    #         # np.save(os.path.join(get_project_root(), "data", "tb_cut", "sections", "1"), data)
    #
    #         # section = mute(section, 80, 10)
    #         xz_synt = section[:, :, point[2]]
    #         zy_synt = section[point[0], :, :]
    #
    #         imgs = [xz_real, zy_real, xz_synt, zy_synt]
    #         show_in_plot(imgs, perc=100)
    #     # np.save(os.path.join(get_project_root(), "data", "synt_new_tv", "sections", str(i)), section)
    #     # np.save(os.path.join(get_project_root(), "data", "synt_new_tv", "layers", str(i)), new_layers)
