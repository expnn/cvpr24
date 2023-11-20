# -*- coding: utf-8 -*-
import os
import json
import math
import torch
import numpy as np
import torch.distributed as dist
from torch.utils.data import Sampler
from monai import transforms, data

from gkai.config import DATA_DIR


class BraTS21Dataset(data.Dataset):
    def __init__(self, subdir, spatial_shape, training, fold=0, mode='production', train_size=10, valid_size=3):
        self.base_dir = os.path.join(DATA_DIR, subdir)
        self.json_list = os.path.join(self.base_dir, "brats21_folds.json")

        train_files, valid_files = self.read_datafold(datalist=self.json_list, basedir=self.base_dir, fold=fold)
        if mode == 'debug':
            train_files = train_files[:train_size]
            valid_files = valid_files[:valid_size]

        if training:
            transform = transforms.Compose(
                [
                    transforms.LoadImaged(keys=["image", "label"], image_only=False),
                    lambda item: {'image': item['image'], 'label': item['label']},
                    transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                    transforms.CropForegroundd(
                        keys=["image", "label"], source_key="image", k_divisible=spatial_shape
                    ),
                    transforms.RandSpatialCropd(
                        keys=["image", "label"], roi_size=spatial_shape, random_size=False
                    ),
                    transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                    transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                    transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                    transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                    transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                    transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
                    transforms.ToTensord(keys=["image", "label"], track_meta=False),
                ]
            )
            super().__init__(data=train_files, transform=transform)
        else:
            transform = transforms.Compose(
                [
                    transforms.LoadImaged(keys=["image", "label"], image_only=False),
                    lambda item: {'image': item['image'], 'label': item['label']},
                    transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                    transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                    transforms.ToTensord(keys=["image", "label"], track_meta=False),
                ]
            )
            super().__init__(data=valid_files, transform=transform)

    def __getitem__(self, item):
        value = super().__getitem__(item)
        return value['image'], value['label']

    @staticmethod
    def read_datafold(datalist, basedir, fold=0, key="training"):
        with open(datalist) as f:
            json_data = json.load(f)

        json_data = json_data[key]

        for d in json_data:
            for k, v in d.items():
                if isinstance(d[k], list):
                    d[k] = [os.path.join(basedir, iv) for iv in d[k]]
                elif isinstance(d[k], str):
                    d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

        tr = []
        val = []
        for d in json_data:
            if "fold" in d and d["fold"] == fold:
                val.append(d)
            else:
                tr.append(d)

        return tr, val


class BraTS21Sampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        super().__init__(dataset)
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank: self.total_size: self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
