import os
import time
from itertools import chain

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yacs.config
from torch.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from module.dataset_pohang_shore import PohangShoreDataset
from module.ssim import SSIM
from network.pohang_shore_network import get_gen_model, get_dis_model
from utils.data import get_project_root, ValueTracker
from utils.parallel import setup, cleanup, run_target
from utils.pytorch import init_weights


def main(config):
    config_file = os.path.join(get_project_root(), "config", config)
    with open(config_file, "rt") as f_read:
        CF = yacs.config.load_cfg(f_read)

    if CF.DEVICE == "cuda":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(gpu_id) for gpu_id in CF.GPUS])

    world_size = torch.cuda.device_count() if CF.DEVICE == "cuda" else 1
    run_target(train, world_size, CF)


def train(rank, world_size, CF):
    setup(rank, world_size, 19467)

    tag = CF.TAG
    if rank == 0:
        print("Tag:", tag)
    cudnn.benchmark = CF.CUDNN_BENCHMARK
    is_parallel = False
    if CF.DEVICE == "cpu":
        device = torch.device("cpu")
    elif CF.DEVICE == "cuda":
        if len(CF.GPUS) == 1:
            device = torch.device("cuda:" + str(CF.GPUS[0]))
        elif len(CF.GPUS) > 1:
            is_parallel = True
            device = torch.device("cuda")
            torch.cuda.set_device(rank)
        else:
            exit(1)
    else:
        exit(1)

    if rank == 0:
        print("Number of GPU:", world_size)

    A_dataset = PohangShoreDataset(
        is_das=True,
        crop_size=CF.DATASET.CROP_SIZE,
        total_length=CF.DATASET.NUM_DATA,
        is_flip=True,
        is_negative=True,
        noise=0.0,
        is_train=True,
    )
    B_dataset = PohangShoreDataset(
        is_das=False,
        crop_size=CF.DATASET.CROP_SIZE,
        total_length=CF.DATASET.NUM_DATA,
        is_flip=True,
        is_negative=True,
        noise=0.0,
        is_train=True,
    )

    A_sampler = DistributedSampler(A_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    B_sampler = DistributedSampler(B_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)

    A_loader = DataLoader(
        A_dataset,
        shuffle=False,
        batch_size=CF.TRAIN.BATCH_SIZE // world_size,
        num_workers=1,
        pin_memory=True,
        sampler=A_sampler,
        persistent_workers=True,
    )
    B_loader = DataLoader(
        B_dataset,
        shuffle=False,
        batch_size=CF.TRAIN.BATCH_SIZE // world_size,
        num_workers=1,
        pin_memory=True,
        sampler=B_sampler,
        persistent_workers=True,
    )

    n_batch = max(len(A_loader), len(B_loader))
    if rank == 0:
        print("Length:", len(A_loader), len(B_loader))

    # Train
    G_A2B = get_gen_model(CF, True).to(rank)
    G_B2A = get_gen_model(CF, False).to(rank)
    D_B = get_dis_model(CF, False).to(rank)
    if is_parallel:
        G_A2B = DDP(G_A2B, device_ids=[rank], find_unused_parameters=True)
        G_B2A = DDP(G_B2A, device_ids=[rank], find_unused_parameters=True)
        D_B = DDP(D_B, device_ids=[rank], find_unused_parameters=True)

    # Loss
    value_criterion = nn.L1Loss().to(rank)
    ssim_criterion = SSIM().to(rank)

    # Optimizer
    optimizer_G = optim.AdamW(chain(G_A2B.parameters(), G_B2A.parameters()), lr=CF.TRAIN.GEN_LR, betas=(CF.TRAIN.BETA1, CF.TRAIN.BETA2), weight_decay=1e-4)
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=CF.TRAIN.DIS_LR, betas=(CF.TRAIN.BETA1, CF.TRAIN.BETA2))

    lr_scheduler_G   = optim.lr_scheduler.ExponentialLR(optimizer_G,   gamma=1.0)
    lr_scheduler_D_B = optim.lr_scheduler.ExponentialLR(optimizer_D_B, gamma=1.0)

    identity_losses = []
    cycle_losses = []
    G_losses = []
    D_B_losses = []
    G_B_losses = []
    if CF.PRETRAIN.LOAD:
        load_tag = CF.PRETRAIN.TAG
        state = torch.load(os.path.join(get_project_root(), "checkpoint", load_tag + "_" + str(CF.PRETRAIN.LOAD_EPOCH).zfill(3)), map_location=lambda storage, loc: storage)

        if is_parallel:
            G_A2B.module.load_state_dict(state["G_A2B"])
            G_B2A.module.load_state_dict(state["G_B2A"])
            D_B.module.load_state_dict(state["D_B"])
        else:
            G_A2B.load_state_dict(state["G_A2B"])
            G_B2A.load_state_dict(state["G_B2A"])
            D_B.load_state_dict(state["D_B"])

        if CF.PRETRAIN.LOAD_OPTIMIZER:
            optimizer_G.load_state_dict(state["optimizer_G"])
            optimizer_D_B.load_state_dict(state["optimizer_D_B"])

            lr_scheduler_G.load_state_dict(state["lr_scheduler_G"])
            lr_scheduler_D_B.load_state_dict(state["lr_scheduler_D_B"])

        # identity_losses = state["identity_loss"]
        # cycle_losses = state["cycle_loss"]
        # G_losses = state["G_loss"]
        # D_B_losses = state["D_B_loss"]
        # G_B_losses = state["G_B_loss"]
    else:
        G_A2B.apply(init_weights)
        G_B2A.apply(init_weights)
        D_B.apply(init_weights)

    # Training
    real_labels = torch.full((CF.TRAIN.BATCH_SIZE // world_size, 1, 1, 1), 1.0, requires_grad=False).to(rank)
    fake_labels = torch.full((CF.TRAIN.BATCH_SIZE // world_size, 1, 1, 1), -1.0, requires_grad=False).to(rank)

    # for param_group in optimizer_D_B.param_groups:
    #     param_group['lr'] = 5e-6
    # lr_scheduler_D_B = torch.optim.lr_scheduler.ExponentialLR(optimizer_D_B, gamma=1.0)

    scaler = GradScaler(device="cuda")
    noise_level = CF.DATASET.NOISE
    ema_coeff = 0.99
    avg_identity_loss = ValueTracker(ema_coeff)
    avg_cycle_loss = ValueTracker(ema_coeff)
    avg_G_loss = ValueTracker(ema_coeff)
    avg_D_B_loss = ValueTracker(ema_coeff)
    avg_G_B_loss = ValueTracker(ema_coeff)
    grad_ratio = CF.TRAIN.LAMBDA_ADV
    for i_epoch in range(CF.TRAIN.BEGIN_EPOCH, CF.TRAIN.END_EPOCH):
        if rank == 0:
            lr_G = lr_scheduler_G.get_last_lr()
            lr_D_B = lr_scheduler_D_B.get_last_lr()
            print(f"epoch: {i_epoch + 1:4d}, learning rate: {lr_G[0]} {lr_D_B[0]}")
        start_time = time.time()
        avg_identity_loss.initialize()
        avg_cycle_loss.initialize()
        avg_G_loss.initialize()
        avg_D_B_loss.initialize()
        avg_G_B_loss.initialize()
        for i_iter in range(CF.TRAIN.ITER_PER_EPOCH):
            for i_batch, (real_A_images, real_B_images) in enumerate(zip(A_loader, B_loader)):
                real_A_images = real_A_images.to(rank, non_blocking=True)
                real_B_images = real_B_images.to(rank, non_blocking=True)

                # Generator
                for param in G_A2B.parameters():
                    param.requires_grad_(True)
                for param in G_B2A.parameters():
                    param.requires_grad_(True)

                for param in D_B.parameters():
                    param.requires_grad_(False)

                with autocast(device_type="cuda"):
                    # Identity Loss
                    same_A_images = G_B2A(real_A_images, is_check=True)
                    same_B_images = G_A2B(real_B_images, is_check=True)

                    identity_loss = (
                        value_criterion(real_A_images, same_A_images) +
                        value_criterion(real_B_images, same_B_images) +
                        (CF.TRAIN.LAMBDA_SSIM * (
                            1.0 - ssim_criterion(real_A_images, same_A_images) +
                            1.0 - ssim_criterion(real_B_images, same_B_images)
                        ) if CF.TRAIN.LAMBDA_SSIM != 0.0 else 0.0)
                    ) / 2.0

                    # Cycle Consistency Loss
                    fake_B_images = G_A2B(real_A_images, is_check=True)
                    fake_A_images = G_B2A(real_B_images, is_check=True)

                    reverse_A_images = G_B2A(fake_B_images, is_check=True)
                    reverse_B_images = G_A2B(fake_A_images, is_check=True)

                    cycle_loss = (
                        value_criterion(real_A_images, reverse_A_images) +
                        value_criterion(real_B_images, reverse_B_images) +
                        (CF.TRAIN.LAMBDA_SSIM * (
                            1.0 - ssim_criterion(real_A_images, reverse_A_images) +
                            1.0 - ssim_criterion(real_B_images, reverse_B_images)
                        ) if CF.TRAIN.LAMBDA_SSIM != 0.0 else 0.0)
                    ) / 2.0

                    value_loss = CF.TRAIN.LAMBDA_I * identity_loss + CF.TRAIN.LAMBDA_C * cycle_loss

                    fake_B_logits = D_B(fake_B_images)

                    # GAN_loss = adversarial_criterion(fake_B_logits, real_labels) + adversarial_criterion(fake_A_logits, real_labels)
                    GAN_loss = -fake_B_logits.mean()

                    # G_loss = CF.TRAIN.LAMBDA_I * identity_loss + CF.TRAIN.LAMBDA_C * cycle_loss + GAN_loss
                    G_loss = value_loss / (1 + grad_ratio) + grad_ratio * GAN_loss / (1 + grad_ratio)

                avg_identity_loss.feed(identity_loss.detach().item())
                avg_cycle_loss.feed(cycle_loss.detach().item())
                avg_G_B_loss.feed(fake_B_logits.detach().mean().item())
                avg_G_loss.feed(G_loss.detach().item())

                optimizer_G.zero_grad()
                scaler.scale(G_loss).backward()
                scaler.step(optimizer_G)

                for param in G_A2B.parameters():
                    param.requires_grad_(False)
                for param in G_B2A.parameters():
                    param.requires_grad_(False)

                for param in D_B.parameters():
                    param.requires_grad_(True)

                # Discriminator Losses
                with torch.no_grad():
                    real_A_images = real_A_images + noise_level * torch.rand_like(real_A_images)
                    real_B_images = real_B_images + noise_level * torch.rand_like(real_B_images)

                    fake_A_images = fake_A_images + noise_level * torch.rand_like(fake_A_images)
                    fake_B_images = fake_B_images + noise_level * torch.rand_like(fake_B_images)

                with autocast(device_type="cuda"):
                    real_B_logits = D_B(real_B_images.detach())
                    fake_B_logits = D_B(fake_B_images.detach())

                    # D_B_loss = adversarial_criterion(fake_B_logits, fake_labels) + adversarial_criterion(real_B_logits, real_labels)
                    D_B_loss = nn.ReLU()(1.0 + fake_B_logits).mean() + nn.ReLU()(1.0 - real_B_logits).mean()

                avg_D_B_loss.feed(real_B_logits.detach().mean().item())

                optimizer_D_B.zero_grad()
                scaler.scale(D_B_loss).backward()
                scaler.step(optimizer_D_B)

                scaler.update()

                if rank == 0 and ((i_batch + 1) % 10 == 0 or (i_batch + 1) == n_batch):
                    print("epoch: {:4}, batch: {:4}, identity_loss: {:8.3e}, cycle_loss: {:8.3e}, G_loss: {:8.3e}, B_score: {:7.4f}, {:7.4f}".format(
                        i_epoch + 1,
                        i_batch + 1,
                        avg_identity_loss.val(),
                        avg_cycle_loss.val(),
                        avg_G_loss.val(),
                        avg_D_B_loss.val(),
                        avg_G_B_loss.val(),
                    ))

        lr_scheduler_G.step()
        lr_scheduler_D_B.step()
        noise_level *= CF.DATASET.NOISE_DECAY

        if rank == 0:
            if True:
                identity_losses.append(avg_identity_loss.val())
                cycle_losses.append(avg_cycle_loss.val())
                G_losses.append(avg_G_loss.val())
                D_B_losses.append(avg_D_B_loss.val())
                G_B_losses.append(avg_G_B_loss.val())

                state = {
                    "G_A2B": G_A2B.module.state_dict() if is_parallel else G_A2B.state_dict(),
                    "G_B2A": G_B2A.module.state_dict() if is_parallel else G_B2A.state_dict(),
                    "D_B": D_B.module.state_dict() if is_parallel else D_B.state_dict(),
                    "optimizer_G": optimizer_G.state_dict(),
                    "optimizer_D_B": optimizer_D_B.state_dict(),
                    "lr_scheduler_G": lr_scheduler_G.state_dict(),
                    "lr_scheduler_D_B": lr_scheduler_D_B.state_dict(),
                    "identity_loss": identity_losses,
                    "cycle_loss": cycle_losses,
                    "G_loss": G_losses,
                    "D_B_loss": D_B_losses,
                    "G_B_loss": G_B_losses,
                }

                print("epoch: {:4}, save training state".format(i_epoch + 1))
                torch.save(state, os.path.join(get_project_root(), "checkpoint", tag + "_" + str(i_epoch + 1).zfill(3)))
                if i_epoch % CF.TRAIN.CHECK_EPOCH != 0:
                    os.system("rm " + os.path.join(get_project_root(), "checkpoint", tag + "_" + str(i_epoch).zfill(3)))

            print("epoch: {:4}, execution time: {:6.2f}".format(
                i_epoch + 1,
                time.time() - start_time,
            ))

    cleanup()

if __name__ == "__main__":
    main("pohang_shore_das_str_oneway.yaml")
