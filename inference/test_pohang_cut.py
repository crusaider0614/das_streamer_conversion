import torch
import yacs.config
import os

from module.dataset_pohang_shore import PohangShoreDataset
from network.pohang_shore_network_aniso import get_gen_model
from torch.utils.data import DataLoader

from utils.data import get_project_root, show_2d_array
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


epoch = 130

# Define
config_file = os.path.join(get_project_root(), "config", "pohang_shore_das_str_cut.yaml")
with open(config_file, "rt") as f_read:
    CF = yacs.config.load_cfg(f_read)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:9")
tag = CF.TAG
print("Tag:", tag)

# Dataset
# num_data = 20
num_data = 240
A_dataset = PohangShoreDataset(
    is_das=True,
    crop_size=None,
    total_length=num_data,
    is_flip=False,
    is_negative=False,
    noise=0.0,
    is_train=False,
    stage="rg_infer"
)
B_dataset = PohangShoreDataset(
    is_das=False,
    crop_size=None,
    total_length=num_data,
    is_flip=False,
    is_negative=False,
    noise=0.0,
    is_train=True,
)

A_loader = DataLoader(A_dataset, batch_size=1, shuffle=True, drop_last=True)
B_loader = DataLoader(B_dataset, batch_size=1, shuffle=True, drop_last=True)

# Load
G_A2B = get_gen_model(CF, True).to(device)
state = torch.load(os.path.join(get_project_root(), "checkpoint", tag + "_" + str(epoch).zfill(3)), map_location=lambda storage, loc: storage)
G_A2B.load_state_dict(state["G_A2B"])
G_A2B = G_A2B.eval()


# Testing
with torch.no_grad():
    # for i in range(len(A_dataset)):
    # for i in [0,135,270]:
    import random
    real_A_imgs = None
    real_B_imgs = None
    fake_imgs = None
    for (real_A_image, real_B_image) in zip(A_loader, B_loader):
        real_A_image = real_A_image.to(device)
        real_B_image = real_B_image.to(device)

        same_B_image = G_A2B(real_B_image)
        fake_B_image = G_A2B(real_A_image)

        real_A_numpy = real_A_image.cpu().numpy().squeeze()
        fake_B_numpy = fake_B_image.cpu().numpy().squeeze()
        real_B_numpy = real_B_image.cpu().numpy().squeeze()
        same_B_numpy = same_B_image.cpu().numpy().squeeze()
        tb = -np.ones((real_A_numpy.shape[0], 4), dtype=np.float32)

        imgs = np.concatenate((
            real_A_numpy, tb,
            fake_B_numpy, tb,
            real_B_numpy, tb,
            same_B_numpy,
        ), axis=1)

        print(real_A_image.min(), real_A_image.max())
        print(fake_B_image.min(), fake_B_image.max())
        print(real_B_image.min(), real_B_image.max())
        print(same_B_image.min(), same_B_image.max())

        show_2d_array(imgs, vmax=1.0, vmin=-1.0)

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

