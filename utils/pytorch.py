import os
import random

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn

from utils.data import get_project_root


def get_n_params(model):
    n_whole_param = 0
    for parameter in list(model.parameters()):
        n_param = 1
        for len_dim in list(parameter.size()):
            n_param = len_dim * n_param
        n_whole_param += n_param
    return n_whole_param


def set_droprate(model, droprate):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = droprate


def load_network_state_dict(network, state_dict):
    network_dict = network.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in network_dict}
    network_dict.update(state_dict)
    network.load_state_dict(network_dict)


def sample_latent(shape, nzg, nzh, nzl, global_latent=None, horizontal_latent=None, local_latent=None):
    batch_size, nz, nh, nw = shape
    assert nz == nzg + nzh + nzl

    if global_latent is not None:
        global_latent_channels = global_latent.copy()
    else:
        global_latent_channels = np.random.randn(batch_size, nzg).astype(np.float32)
        global_latent_channels = np.expand_dims(global_latent_channels, axis=(2, 3))
        global_latent_channels = np.repeat(global_latent_channels, nh, axis=2)
        global_latent_channels = np.repeat(global_latent_channels, nw, axis=3)

    if horizontal_latent is not None:
        horizontal_latent_channels = horizontal_latent.copy()
    else:
        horizontal_latent_channels = np.random.randn(batch_size, nzh, nw).astype(np.float32)
        horizontal_latent_channels = np.expand_dims(horizontal_latent_channels, axis=2)
        horizontal_latent_channels = np.repeat(horizontal_latent_channels, nh, axis=2)

    if local_latent is not None:
        local_latent_channels = local_latent.copy()
    else:
        local_latent_channels = np.random.randn(batch_size, nzl, nh, nw).astype(np.float32)

    z = np.concatenate((global_latent_channels, horizontal_latent_channels, local_latent_channels), axis=1)


def init_weights(module):
    if (
        type(module) == nn.Linear or
        type(module) == nn.Conv2d or
        type(module) == nn.ConvTranspose2d or
        type(module) == nn.Conv3d or
        type(module) == nn.ConvTranspose3d
    ):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


n_zero = 1
n_pad = 4

def mute_tb(data):
    data[:, :, :n_zero] = 0
    for i in range(n_pad):
        data[:, :, n_zero + i] *= i / n_pad
    data[:, :, -n_zero:] = 0
    for i in range(n_pad):
        data[:, :, -n_zero - i] *= i / n_pad
    return data


def mute_tb_numpy(data):
    data[:n_zero] = 0
    for i in range(n_pad):
        data[n_zero + i] *= i / n_pad
    data[-n_zero:] = 0
    for i in range(n_pad):
        data[-n_zero - i] *= i / n_pad
    return data


def coordi_tensor(d_center, d_delta, nz, nx):
    d_center_numpy = d_center.detach().cpu().numpy()
    d_delta_numpy = d_delta.detach().cpu().numpy()
    device = d_center.device

    d_ptp = (d_delta_numpy * (nz - 1))
    d_min = d_center_numpy - d_ptp / 2
    d_max = d_min + d_ptp

    dc = np.linspace(d_min, d_max, nz, dtype=np.float32, axis=1)
    dc = np.expand_dims(dc, axis=2).repeat(nx, axis=2)
    dc = np.expand_dims(dc, axis=1)
    dc = torch.tensor(dc, device=device, dtype=torch.float32, requires_grad=False)
    return dc


# def coordi_tensor(c_center, c_delta, nh, nw):
#     c_center_numpy = c_center.detach().cpu().numpy()
#     c_delta_numpy = c_delta.detach().cpu().numpy()
#     device = c_center.device
#
#     ch_ptp = c_delta_numpy[:, 0] * (nh - 1)
#     ch_min = c_center_numpy[:, 0] - ch_ptp / 2
#     ch_max = ch_min + ch_ptp
#
#     ch = np.linspace(ch_min, ch_max, nh, dtype=np.float32, axis=1)
#     ch = np.expand_dims(ch, axis=2).repeat(nw, axis=2)
#     ch = np.expand_dims(ch, axis=1)
#
#     cw_ptp = c_delta_numpy[:, 1] * (nw - 1)
#     cw_min = c_center_numpy[:, 1] - cw_ptp / 2
#     cw_max = cw_min + cw_ptp
#
#     cw = np.linspace(cw_min, cw_max, nw, dtype=np.float32, axis=1)
#     cw = np.expand_dims(cw, axis=2).repeat(nh, axis=2)
#     cw = np.expand_dims(cw, axis=1)
#     cw = np.transpose(cw, (0, 1, 3, 2))
#
#     cc = np.append(ch, cw, axis=1)
#     cc = torch.tensor(cc, device=device, dtype=torch.float32, requires_grad=False)
#     return cc


def get_horizontal_seed(tag):
    file = open(os.path.join(get_project_root(), "horizontal_seed", tag + ".dat"), "r")
    lines = file.readlines()
    horizontal_seed = []
    for line in lines:
        line = line.strip()
        horizontal_seed.append(float(line))
    return horizontal_seed


def get_horizontal_code(horizontal_seed, batch_size, nzh, nh, nw, vertical_expand=True):
    horizontal_seed_cycle = np.concatenate((horizontal_seed, horizontal_seed), axis=0)

    if vertical_expand:
        horizontal_code = np.zeros((batch_size, nzh, nw), dtype=np.float32)
        for batch_idx in range(batch_size):
            for channel_idx in range(nzh):
                start_idx = random.randint(0, len(horizontal_seed))
                code = horizontal_seed_cycle[start_idx: start_idx + nw]
                horizontal_code[batch_idx, channel_idx] = code
        horizontal_code = np.expand_dims(horizontal_code, axis=2)
        horizontal_code = np.repeat(horizontal_code, nh, axis=2)
    else:
        horizontal_code = np.zeros((batch_size, nzh, nh, nw), dtype=np.float32)
        for batch_idx in range(batch_size):
            for channel_idx in range(nzh):
                for ih in range(nh):
                    start_idx = random.randint(0, len(horizontal_seed))
                    code = horizontal_seed_cycle[start_idx: start_idx + nw]
                    horizontal_code[batch_idx, channel_idx, ih] = code
    return horizontal_code


def crop_sample(x, n_crop, nzs, nxs):
    batch_size, nc, nzo, nxo = x.size()
    device = x.device
    x_sample = torch.zeros((batch_size * n_crop, nc, nzs, nxs), dtype=torch.float32, device=device)
    for i_batch in range(batch_size):
        for i_crop in range(n_crop):
            i_sample = n_crop * i_batch + i_crop

            mz = (nzo - nzs) // 50 + 1
            mx = (nxo - nxs) // 50 + 1

            sz = random.randint(-mz, nzo - nzs + mz)
            sx = random.randint(-mx, nxo - nxs + mx)

            sz = np.clip(sz, 0, nzo - nzs)
            sx = np.clip(sx, 0, nxo - nxs)

            ez = sz + nzs
            ex = sx + nxs

            x_sample[i_sample] = x[i_batch, :, sz: ez, sx: ex]
    return x_sample


def crop_sample_3d(x, n_crop, nzs, nxs, nys):
    batch_size, nc, nzo, nxo, nyo = x.size()
    device = x.device
    x_sample = torch.zeros((batch_size * n_crop, nc, nzs, nxs, nys), dtype=torch.float32, device=device)
    for i_batch in range(batch_size):
        for i_crop in range(n_crop):
            i_sample = n_crop * i_batch + i_crop

            mz = (nzo - nzs) // 50 + 1
            mx = (nxo - nxs) // 50 + 1
            my = (nyo - nys) // 50 + 1

            sz = random.randint(-mz, nzo - nzs + mz)
            sx = random.randint(-mx, nxo - nxs + mx)
            sy = random.randint(-my, nyo - nys + my)

            sz = np.clip(sz, 0, nzo - nzs)
            sx = np.clip(sx, 0, nxo - nxs)
            sy = np.clip(sy, 0, nyo - nys)

            ez = sz + nzs
            ex = sx + nxs
            ey = sy + nys

            x_sample[i_sample] = x[i_batch, :, sz: ez, sx: ex, sy: ey]
    return x_sample


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    def show_in_plot(imgs, cval=None, perc=100, **kwargs):
        if cval is not None:
            maxval = cval[0]
            minval = cval[1]
        else:
            maxval = np.quantile(imgs, perc / 100)
            minval = np.quantile(imgs, 1 - perc / 100)
        plt.imshow(imgs, vmax=maxval, vmin=minval, **kwargs)
        plt.show()

    for i in range(100):
        data = np.load(os.path.join(get_project_root(), "data", "ccs_synt", "sections", str(i) + ".npy")).astype(np.float32)
        show_in_plot(data, aspect=6, cmap="seismic")
