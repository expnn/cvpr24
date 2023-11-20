# -*- coding: utf-8 -*-
import functools
import numpy as np
import os.path
from glob import glob
from torch.utils.data import Dataset, DataLoader
from gkai.config import DATA_DIR


class ColaDataset(Dataset):
    def __init__(
            self,
            subdir: str,
            file_pat='[0-9][0-9][0-9]-*.npz',
            cache_size=30,
            choose_file_by_seed=lambda x: True
    ):
        root = os.path.join(DATA_DIR, subdir)
        self.cache_size = cache_size
        self.load_file = functools.lru_cache(self.cache_size)(np.load)

        if isinstance(choose_file_by_seed, str):
            choose_file_by_seed = eval(choose_file_by_seed)
        self.choose_file_by_seed = choose_file_by_seed

        self.files = []
        self.size = 0
        self.index = []
        start = 0
        for f in os.scandir(root):
            if not f.is_dir():
                continue

            try:
                seed = int(f.name)
            except ValueError:
                continue
            if not choose_file_by_seed(seed):
                continue

            files = glob(os.path.join(f.path, file_pat))
            size = len(files) - 1
            self.size += size
            self.files.extend(files)
            self.files.append(None)
            self.index.extend(range(start, start + size))
            start += size + 2

        assert len(self.index) == self.size
        for i in self.index:
            assert self.files[i] is not None and self.files[i + 1] is not None

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        i = self.index[item]
        x = self.files[i]
        y = self.files[i + 1]
        # add a channel dimension
        return self.load_file(x)['state'][np.newaxis, ...], self.load_file(y)['state'][np.newaxis, ...]


if __name__ == '__main__':
    def demo1():
        dataset = ColaDataset(os.path.join(DATA_DIR, 'cosmo-simu', 'sim-cola-preprocessed-128'))
        print(len(dataset))
        x, y = dataset[0]
        print(x.shape)
        print(y.shape)
        dataloader = DataLoader(dataset, batch_size=3, shuffle=False)
        x, y = next(iter(dataloader))
        print(x.shape)
        print(y.shape)

    demo1()
