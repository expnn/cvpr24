# -*- coding: utf-8 -*-
import os
import torch
from torch.utils.data import Dataset, default_collate
from torchvision.datasets.folder import ImageFolder
from timm.data import create_transform, Mixup
from gkai.config import DATA_DIR


class ImageNetDataset(Dataset):
    def __init__(
            self,
            subdir: str = 'imagenet',
            training: bool = False,
            input_size: int = 256,
            auto_augment: str = 'rand-m9-mstd0.5-inc1',
            color_jitter: float = 0.4,
            interpolation: str = 'bicubic',
            re_prob: float = 0.0,
            re_mode: str = 'pixel',
            re_count: int = 1,
            scale: tuple = (0.5, 1.0),
            ratio: tuple = (3. / 4., 4. / 3.),
            hflip=0.5,
            vflip=0.,
            mixup=None,
    ):
        super(ImageNetDataset, self).__init__()
        if training:
            self.root = os.path.join(DATA_DIR, subdir, 'train')
        else:
            self.root = os.path.join(DATA_DIR, subdir, 'val')
        transform = create_transform(
            input_size=input_size,
            is_training=training,
            color_jitter=color_jitter,
            auto_augment=auto_augment,
            interpolation=interpolation,
            re_prob=re_prob,
            re_mode=re_mode,
            re_count=re_count,
            scale=scale,
            ratio=ratio,
            hflip=hflip,
            vflip=vflip,
        )
        self.data = ImageFolder(self.root, transform=transform)

        if mixup is not None and training:
            self.mixup = Mixup(**mixup)
        else:
            self.mixup = None
        self.training = training

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        batch = default_collate(batch)
        if self.mixup:
            return self.mixup(*batch)
        else:
            return batch


class ImageNetForEnsembleClassifier(ImageNetDataset):
    def __init__(self, *args, ensemble_heads=(16, 16), **kwargs):
        super().__init__(*args, **kwargs)
        self.ensemble_heads = tuple(ensemble_heads)

    def collate_fn(self, batch):
        batch = super().collate_fn(batch)
        if not self.training:
            return batch
        else:
            x, y = batch
            s = tuple(y.shape)
            y = torch.reshape(y, s + (1, 1))
            y = y.expand(s + self.ensemble_heads).clone()
            return x, y


if __name__ == '__main__':
    def demo():
        dataset = ImageNetDataset(os.path.join(DATA_DIR, 'imagenet'), training=True)
        print(len(dataset))
        print(dataset[0])
        print(dataset[0][0].shape)
        print("=" * 60)
        dataset = ImageNetForEnsembleClassifier(os.path.join(DATA_DIR, 'imagenet'), training=True)
        batch = [dataset[i] for i in range(10)]
        batch = dataset.collate_fn(batch)
        print(batch[0].shape)
        print(batch[1].shape)

    demo()
