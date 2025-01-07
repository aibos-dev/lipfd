# import torch
# import numpy as np
# from torch.utils.data.sampler import WeightedRandomSampler
# from .datasets import AVLip


# def get_bal_sampler(dataset):
#     targets = []
#     for d in dataset.datasets:
#         targets.extend(d.targets)

#     ratio = np.bincount(targets)
#     w = 1.0 / torch.tensor(ratio, dtype=torch.float)
#     sample_weights = w[targets]
#     sampler = WeightedRandomSampler(
#         weights=sample_weights, num_samples=len(sample_weights)
#     )
#     return sampler


# def create_dataloader(opt):
#     shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
#     dataset = AVLip(opt)

#     sampler = get_bal_sampler(dataset) if opt.class_bal else None

#     data_loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=opt.batch_size,
#         shuffle=True,
#         sampler=sampler,
#         num_workers=int(opt.num_threads),
#     )
#     return data_loader

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
        batch_size=opt.batch_size,
        shuffle=True,
        sampler=sampler,
        num_workers=int(opt.num_threads),
    )
    return data_loader


def create_distributed_dataloader(opt, num_workers, pin_memory, rank, dataset=None):
    """Creates a distributed dataloader for multi-GPU training"""
    if dataset is None:
        dataset = AVLip(opt)  # Ensure this dataset can be created correctly

    # Ensure that DistributedSampler is initialized properly
    sampler = DistributedSampler(
        dataset,
        num_replicas=opt.world_size,  # Pass world_size explicitly if needed
        rank=opt.local_rank,  # Pass local_rank explicitly
        shuffle=not opt.serial_batches if opt.isTrain else False
    )

    # Ensure data_loader is configured with appropriate parameters
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size // opt.world_size,  # Account for distributed batch size
        sampler=sampler,
        num_workers=4,
        pin_memory=True,  # Try False if you encounter memory issues
    )

    return data_loader, sampler
