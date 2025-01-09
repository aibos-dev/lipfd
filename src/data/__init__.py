import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from .datasets import AVLip


def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1.0 / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights)
    )
    return sampler


def create_dataloader(opt):
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    dataset = AVLip(opt)

    sampler = get_bal_sampler(dataset) if opt.class_bal else None

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        sampler=sampler,
        num_workers=int(opt.num_threads),
    )
    return data_loader


def create_distributed_dataloader(opt, num_workers, pin_memory, rank, dataset=None):
    if dataset is None:
        dataset = AVLip(opt)

    sampler = DistributedSampler(
        dataset,
        num_replicas=opt.world_size,
        rank=opt.local_rank,
        shuffle=not opt.serial_batches if opt.isTrain else False
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4 // opt.world_size,
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
    )

    return data_loader, sampler
