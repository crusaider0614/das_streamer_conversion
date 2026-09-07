import torch
import yacs.config
import os

from module.dataset_pohang_shore import PohangShoreDataset
from network.pohang_shore_network_aniso import get_gen_model
from torch.utils.data import DataLoader

from utils.data import get_project_root, show_2d_array
from utils.process import calculate_norscale_inversion
from matplotlib import pyplot as plt
import numpy as np


def show_in_plot(imgs, perc=0.95):
    plt.figure()
    for i in range(imgs.shape[0]):
        plt.subplot(2, 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(imgs[i, :, :], cmap='gray', vmin=-1.0, vmax=1.0)
    plt.show()


epoch = 198

# Define
config_file = os.path.join(get_project_root(), "config", "pohang_shore_das_str_cut.yaml")
with open(config_file, "rt") as f_read:
    CF = yacs.config.load_cfg(f_read)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:5")
tag = CF.TAG
print("Tag:", tag)

# Dataset
num_data = 20
# num_data = 281
A_dataset = PohangShoreDataset(
    is_das=True,
    crop_size=None,
    total_length=num_data,
    is_flip=False,
    is_negative=False,
    noise=0.0,
    is_train=False,
)
B_dataset = PohangShoreDataset(
    is_das=False,
    crop_size=None,
    total_length=num_data,
    is_flip=False,
    is_negative=False,
    noise=0.0,
    is_train=False,
)

# Load
G_A2B = get_gen_model(CF, True).to(device)
state = torch.load(os.path.join(get_project_root(), "checkpoint", tag + "_" + str(epoch).zfill(3)), map_location=lambda storage, loc: storage)
G_A2B.load_state_dict(state["G_A2B"])
G_A2B = G_A2B.eval()

# The same two arrays process/logenv_process.py takes as its inputs: the DAS
# side band-passed (_freq), the streamer side straight off the geometry grid
# (_geom), which is already band-limited from its own pipeline.
A_nor = np.load(os.path.join(get_project_root(), "data", "pohang_shore", "numpy", "das_data_freq.npy"))[:, :4000]
B_nor = np.load(os.path.join(get_project_root(), "data", "pohang_shore", "numpy", "str_data_geom.npy"))[:, :4000]

das_base = 100
str_base = 1e-2
sigma = (5.0, 2.0)
eps = 1e-8

cval_A_nor = np.max(np.abs(A_nor)) / 20
cval_B_nor = np.max(np.abs(B_nor)) / 20

# Testing
for idx in range(0, 281, 10):
    with torch.no_grad():
        # for i in range(len(A_dataset)):
        # for i in [0,135,270]:
        import random

        real_A_image = A_dataset.__getitem__(idx)
        real_B_image = B_dataset.__getitem__(idx)

        real_A_nor = A_nor[idx]
        real_B_nor = B_nor[idx]

        real_A_image = torch.tensor(real_A_image[None], device=device, dtype=torch.float32)
        real_B_image = torch.tensor(real_B_image[None], device=device, dtype=torch.float32)

        gen_output = G_A2B(real_B_image)
        same_B_signal = gen_output[:, 0:1, :, :]
        same_B_noise = gen_output[:, 1:2, :, :]
        same_B_image = same_B_signal + same_B_noise

        gen_output = G_A2B(real_A_image)
        fake_B_signal = gen_output[:, 0:1, :, :]
        fake_B_noise = gen_output[:, 1:2, :, :]
        fake_B_image = fake_B_signal + fake_B_noise

        real_A_numpy = real_A_image.cpu().numpy().squeeze()
        fake_B_signal_numpy = fake_B_signal.cpu().numpy().squeeze()
        fake_B_noise_numpy = fake_B_noise.cpu().numpy().squeeze()
        fake_B_image_numpy = fake_B_image.cpu().numpy().squeeze()

        real_B_numpy = real_B_image.cpu().numpy().squeeze()
        same_B_signal_numpy = same_B_signal.cpu().numpy().squeeze()
        same_B_noise_numpy = same_B_noise.cpu().numpy().squeeze()
        same_B_image_numpy = same_B_image.cpu().numpy().squeeze()

        tb = -cval_A_nor * np.ones((real_A_numpy.shape[0], 4), dtype=np.float32)

        print(real_A_image.min(), real_A_image.max())
        print(fake_B_image.min(), fake_B_image.max())
        print(real_B_image.min(), real_B_image.max())
        print(same_B_image.min(), same_B_image.max())

        imgs = np.concatenate((
            real_A_numpy, tb,
            fake_B_signal_numpy, tb,
            fake_B_noise_numpy, tb,
            fake_B_image_numpy, tb,
            real_B_numpy, tb,
            same_B_signal_numpy, tb,
            same_B_noise_numpy, tb,
            same_B_image_numpy,
        ), axis=1)[:1000]

        show_2d_array(imgs, vmin=-1, vmax=1)

        continue

        # imgs = np.concatenate((
        #     real_A_nor, tb,
        #     real_A_est, tb,
        #     (real_A_nor - real_A_est),
        # ), axis=1)[:1000]
        #
        # show_2d_array(imgs, vmin=-cval_A_nor, vmax=cval_A_nor)

        _, fake_B_est = calculate_norscale_inversion(fake_B_numpy, log_base=str_base, smooth_sigma=sigma, eps=eps, iterations=20)
        # _, real_B_est = calculate_norscale_inversion(real_B_numpy, log_base=str_base, smooth_sigma=sigma, eps=eps)
        # _, same_B_est = calculate_norscale_inversion(same_B_numpy, log_base=str_base, smooth_sigma=sigma, eps=eps)

        real_A_nor = real_A_nor / cval_A_nor * cval_B_nor

        imgs = np.concatenate((
            real_A_nor, tb,
            4 * fake_B_est, tb,
            real_B_nor,
        ), axis=1)[:1000]

        show_2d_array(imgs, vmin=-cval_B_nor, vmax=cval_B_nor)


        # fake_imgs = fake_imgs.squeeze()
        # fig = plt.figure()
        # fig.set_size_inches((fake_imgs.shape[1] / 100, fake_imgs.shape[0] / 100 - 0.375))
        # ax = plt.Axes(fig, [0., 0., 1., 1.])
        # ax.set_axis_off()
        # fig.add_axes(ax)
        # plt.imshow(fake_imgs, cmap="gray", vmax=0.5, vmin=-0.5)
        # plt.show()

        # for ix in [10, 50, 90, 130, 170, 210]:
        #     fig = plt.figure()
        #     fig.set_size_inches((3 * fake_imgs.shape[2] / 100, fake_imgs.shape[0] / 100 - 0.375))
        #     ax = plt.Axes(fig, [0., 0., 1., 1.])
        #     ax.set_axis_off()
        #     fig.add_axes(ax)
        #     plt.imshow(np.concatenate((real_A_imgs[:, ix], real_B_imgs[:, ix], fake_imgs[:, ix]), axis=1), cmap="gray", vmax=0.5, vmin=-0.5)
        #     plt.show()

        # for ix in [10, 50, 90, 130, 170, 210]:
        #     real_B_image = torch.tensor(real_B_imgs[None, None, :, ix], dtype=torch.float32)
        #     print(real_B_image.shape)
        #     fake_A_image = G_B2A(real_B_image, B_center, c_delta)
        #     fig = plt.figure()
        #     fig.set_size_inches((3 * real_A_imgs.shape[2] / 100, real_A_imgs.shape[0] / 100 - 0.375))
        #     ax = plt.Axes(fig, [0., 0., 1., 1.])
        #     ax.set_axis_off()
        #     fig.add_axes(ax)
        #     plt.imshow(np.concatenate((real_A_imgs[:, ix], real_B_imgs[:, ix], fake_A_image.cpu().numpy().squeeze()), axis=1), cmap="gray", vmax=0.5, vmin=-0.5)
        #     plt.show()

