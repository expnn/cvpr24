# -*- coding: utf-8 -*-

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from monai.networks.nets import SwinUNETR

from gkai.base.config import CfgNode
from gkai.config import CACHE_DIR
from gkai.utils import get_clean_path
from gkai.utils import get_logger
from gkai.utils import import_class

from .afno import AFNONet
from .swin_afno import SwinAFNO
from .infer import Inferencer

__all__ = [
    'SwinUNETR', 'AFNONet',
    'create_model', 'create_loss', 'create_metrics', 'create_inferencer'
]


def create_model(config: CfgNode):
    logger = get_logger('gkai.model')
    model_cls = import_class(config.model.name, 'gkai.model')
    model = model_cls(**config.model.args)
    rank = 0
    if config.run.gpus > 1:
        rank = dist.get_rank()

    pretrained_path = get_clean_path(config.model.pretrained_path)
    if rank == 0 and config.model.pretrained_path:
        pretrained_path = os.path.join(CACHE_DIR, config.model.model_save_subdir, pretrained_path)
        logger.info("Load weights from %s", pretrained_path)
        weights = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(weights)

    if config.run.gpus > 1:
        device_id = rank % torch.cuda.device_count()
        if getattr(config.model.args, "norm_name", None) == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.to(device_id)
        model = DistributedDataParallel(model, device_ids=[device_id])
    elif config.run.gpus == 1:
        model.to('cuda:0')
    else:
        raise ValueError("number of gpus should be a positive number. ")

    return model


def create_loss(config: CfgNode):
    loss_cls = import_class(config.loss.name, 'gkai.model.loss')
    loss = loss_cls(**config.loss.args)
    return loss


def create_metrics(config: CfgNode):
    metrics = {}
    for name, metric in config.validator.metrics.items():
        metrics_cls = import_class(metric.name, 'gkai.model.metric')
        metrics[name] = metrics_cls(**metric.args)
    return metrics


def create_inferencer(config: CfgNode, model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(config.validator, 'inferencer'):
        infer_cfg = config.validator.inferencer
        inferencer_cls = import_class(infer_cfg.name, 'gkai.model.infer')
        return inferencer_cls(model, **infer_cfg.args)
    else:
        return Inferencer(model)
