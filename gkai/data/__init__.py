
from .cola import ColaDataset
from .brats import BraTS21Dataset
from .imgnet import ImageNetDataset, ImageNetForEnsembleClassifier

from gkai.utils import import_class
from gkai.base.config import CfgNode


def create_dataloader(config: CfgNode, training: bool, **dataset_kwargs):
    dataset_cls = import_class(config.data.name, 'gkai.data')
    kwargs = dict(config.data.args)
    kwargs.update(dataset_kwargs)
    if training:
        kwargs.update(config.data.train_args)
    else:
        kwargs.update(config.data.valid_args)
    dataset = dataset_cls(**kwargs)

    shuffle = training
    if config.run.gpus > 1:
        sampler_cls = import_class(config.dataloader.sampler.name, 'gkai.data')
        sampler_args = dict(shuffle=shuffle)
        sampler_args.update(**config.dataloader.sampler.args)
        sampler = sampler_cls(dataset, **sampler_args)
        shuffle = None
    else:
        sampler = None

    if "pin_memory" not in config.dataloader.args:
        overrides = {"pin_memory": True}
    else:
        overrides = {}
    if hasattr(dataset, 'collate_fn'):
        overrides['collate_fn'] = dataset.collate_fn

    loader_cls = import_class(config.dataloader.name, 'gkai.data')
    dataloader = loader_cls(dataset, sampler=sampler, shuffle=shuffle, **overrides, **config.dataloader.args)
    return dataloader
