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


def show_four(panels, names=("real A", "fake B", "real B", "same B"), clip=1.0):
    """네 패널을 축 없이, 확대하면 같은 위치가 보이도록 공유 축으로."""
    fig, axs = plt.subplots(1, 4, figsize=(16, 6), sharex=True, sharey=True)
    for ax, img, name in zip(axs, panels, names):
        ax.imshow(img, cmap="seismic", vmin=-clip, vmax=clip,
                  aspect="auto", interpolation="nearest")
        ax.set_title(name, fontsize=9)
        ax.set_axis_off()
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01,
                        wspace=0.02)
    plt.show()


epoch = 200

# Define
config_file = os.path.join(get_project_root(), "config", "pohang_shore_das_str_cut_decay.yaml")
with open(config_file, "rt") as f_read:
    CF = yacs.config.load_cfg(f_read)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0")
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

        show_four((real_A_numpy, fake_B_numpy, real_B_numpy, same_B_numpy), clip=0.5)
