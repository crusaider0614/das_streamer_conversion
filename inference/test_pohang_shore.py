import torch
import yacs.config
import os

from module.dataset_pohang_shore import PohangShoreDataset
from network.pohang_shore_network import get_gen_model
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


epoch = 200

# Define
config_file = os.path.join(get_project_root(), "config", "pohang_shore_das_str_oneway.yaml")
with open(config_file, "rt") as f_read:
    CF = yacs.config.load_cfg(f_read)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:2")
tag = CF.TAG
print("Tag:", tag)

# Dataset
num_data = 20
A_dataset = PohangShoreDataset(
    is_das=True,
    crop_size=None,
    total_length=num_data,
    is_flip=True,
    is_negative=True,
    noise=0.0,
    is_train=False,
)
B_dataset = PohangShoreDataset(
    is_das=False,
    crop_size=None,
    total_length=num_data,
    is_flip=True,
    is_negative=True,
    noise=0.0,
    is_train=True,
)

A_loader = DataLoader(A_dataset, batch_size=1, shuffle=False, drop_last=True)
B_loader = DataLoader(B_dataset, batch_size=1, shuffle=False, drop_last=True)

# Load
G_A2B = get_gen_model(CF, True).to(device)
G_B2A = get_gen_model(CF, False).to(device)

state = torch.load(os.path.join(get_project_root(), "checkpoint", tag + "_" + str(epoch).zfill(3)), map_location=lambda storage, loc: storage)
G_A2B.load_state_dict(state["G_A2B"])
G_B2A.load_state_dict(state["G_B2A"])

G_A2B = G_A2B.eval()
G_B2A = G_B2A.eval()


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

        same_A_image = G_B2A(real_A_image)
        same_B_image = G_A2B(real_B_image)
        fake_B_image = G_A2B(real_A_image)
        fake_A_image = G_B2A(real_B_image)
        reverse_A_image = G_B2A(fake_B_image)
        reverse_B_image = G_A2B(fake_A_image)

        imgs = np.concatenate((
            real_A_image.cpu().numpy().squeeze(),
            same_A_image.cpu().numpy().squeeze(),
            fake_B_image.cpu().numpy().squeeze(),
            reverse_A_image.cpu().numpy().squeeze(),
            real_B_image.cpu().numpy().squeeze(),
            same_B_image.cpu().numpy().squeeze(),
            fake_A_image.cpu().numpy().squeeze(),
            reverse_B_image.cpu().numpy().squeeze(),
        ), axis=1)[:1000]

        print(real_A_image.min(), real_A_image.max())
        print(same_A_image.min(), same_A_image.max())
        print(fake_B_image.min(), fake_B_image.max())
        print(reverse_A_image.min(), reverse_A_image.max())
        print(real_B_image.min(), real_B_image.max())
        print(same_B_image.min(), same_B_image.max())
        print(fake_A_image.min(), fake_A_image.max())
        print(reverse_B_image.min(), reverse_B_image.max())

        show_2d_array(imgs, vmax=0.5, vmin=-0.5)

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

