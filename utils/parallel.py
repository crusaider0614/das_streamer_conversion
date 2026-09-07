import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from matplotlib import pyplot as plt
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)

    dist.init_process_group("nccl", init_method="tcp://127.0.0.1:" + str(port), rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run_target(target_subroutine, world_size, CF):
    mp.spawn(
        target_subroutine,
        args=(world_size, CF),
        nprocs=world_size,
        join=True
    )


class ToyMpModel(nn.Module):
    def __init__(self):
        super(ToyMpModel, self).__init__()
        self.network = nn.Sequential(
            torch.nn.Linear(1001, 1001),
            torch.nn.ReLU(),
            torch.nn.Linear(1001, 1001),
            torch.nn.ReLU(),
            torch.nn.Linear(1001, 1001),
        )

    def forward(self, x):
        return self.network(x)


from torch.utils.data import Dataset
class TestDataset(Dataset):
    def __init__(self, len):
        self.len = len
        self.x = torch.range(-1.0, 1.0, 0.002)
        self.y = torch.sin(5 * self.x)
        print(self.x.shape, self.y.shape)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.x, self.y


def demo_model_parallel(rank, world_size):
    print(f"Running DDP with model parallel example on rank {rank}.")
    setup(rank, world_size)

    mp_model = ToyMpModel().to(rank)
    mp_model = DDP(mp_model, device_ids=[rank])

    dataset = TestDataset(100)
    sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=1, num_workers=0, pin_memory=True, sampler=sampler)
    optimizer = optim.Adam(mp_model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss().to(rank)

    for i, (x, y) in enumerate(dataloader):
        x = x.to(rank)
        y = y.to(rank)
        print(i, x.shape, y.shape)
        optimizer.zero_grad()
        o = mp_model(x)
        loss_fn(o, y).backward()
        optimizer.step()
        if (i + 1) % 10 == 0 and rank == 0:
            xnp = x.detach().cpu().numpy()
            ynp = y.detach().cpu().numpy()
            onp = o.detach().cpu().numpy()

            plt.scatter(xnp, ynp)
            plt.scatter(xnp, onp)
            plt.show()

    cleanup()


def demo_model(rank, world):
    # 작업을 위한 mp_model 및 장치 설정
    dev0 = "cuda:0"
    dev1 = "cuda:1"
    mp_model = ToyMpModel(dev0, dev1)
    # mp_model = DDP(mp_model)

    dataset = TestDataset((1, 1, 1), 10000)
    # sampler = torch.utils.data.DistributedSampler(dataset, drop_last=True)
    dataloader = torch.utils.data.DataLoader(dataset, drop_last=True, batch_size=2, num_workers=0)

    for i, (x, y) in enumerate(dataloader):
        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(mp_model.parameters(), lr=0.001)

        optimizer.zero_grad()
        outputs = mp_model(x)
        labels = torch.randn(1, 10000).to(dev1)
        loss_fn(outputs, labels).backward()
        optimizer.step()
        print(i, outputs.shape)


if __name__ == "__main__":
    gpus = [8, 9]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(gpu_id) for gpu_id in gpus])
    print(torch.cuda.device_count())

    # demo_model()

    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    run_target(demo_model_parallel, world_size)
