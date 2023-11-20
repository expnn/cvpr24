# -*- coding: utf-8 -*-

import os
import string
import sys
import random
import argparse
import warnings
import torch.cuda
from typing import Union
from faker import Faker
from gkai.base.config import CfgNode


__SCRIPT_PATH = os.path.abspath(__file__)
__SCRIPT_DIR = os.path.dirname(__SCRIPT_PATH)
PROJECT_DIR = os.path.dirname(__SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
CACHE_DIR = os.path.join(PROJECT_DIR, 'cache')


__C = CfgNode()

__C.project = "gkai"

__C.data = CfgNode(new_allowed=True)
__C.data.name = 'ColaDataset'
__C.data.spatial_shape = (256, 256, 256)
__C.data.in_channels = 1
__C.data.out_channels = 1

__C.data.args = CfgNode(new_allowed=True)
__C.data.train_args = CfgNode(new_allowed=True)
__C.data.valid_args = CfgNode(new_allowed=True)

__C.model = CfgNode(new_allowed=True)
__C.model.name = "SwinUNETR"
__C.model.args = CfgNode(new_allowed=True)
__C.model.model_save_subdir = 'cosmo-simu-model'
__C.model.pretrained_path = None

__C.run = CfgNode(new_allowed=True)
__C.run.id = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
__C.run.seed = ''.join(random.choices(string.ascii_letters + string.digits, k=40))
__C.run.epochs = 5000
__C.run.gpus = -1
__C.run.eval_every_steps = 100
__C.run.report_interval = 10
__C.run.visualize_interval = 30
__C.run.save_interval = 100
__C.run.rank = 0
__C.run.local_rank = 0
__C.run.skip_initial_validation = False

__C.gradient_clipper = CfgNode()
__C.gradient_clipper.max_norm = 1.0
__C.gradient_clipper.norm_type = 2.0
__C.gradient_clipper.enabled = False
__C.gradient_clipper.log_name = 'norm_grad'

__C.loss = CfgNode(new_allowed=True)
__C.loss.name = 'torch.nn.MSELoss'
__C.loss.args = CfgNode(new_allowed=True)

__C.validator = CfgNode(new_allowed=True)
__C.validator.metrics = CfgNode(new_allowed=True)
__C.validator.main_metric = None
__C.validator.larger_better = False

__C.optimizer = CfgNode()
__C.optimizer.name = "torch.optim.AdamW"
__C.optimizer.args = CfgNode(new_allowed=True)
__C.optimizer.freeze_params = None
__C.optimizer.freeze_epochs = 1

__C.lr_scheduler = CfgNode()
__C.lr_scheduler.name = "LinearWarmupCosineAnnealingLR"
__C.lr_scheduler.arg_adaptor = CfgNode()
__C.lr_scheduler.arg_adaptor.args = CfgNode(new_allowed=True)
__C.lr_scheduler.arg_adaptor.name = ''
__C.lr_scheduler.args = CfgNode(new_allowed=True)

__C.wandb = CfgNode(new_allowed=True)
__C.wandb.enabled = False
__C.wandb.args = CfgNode(new_allowed=True)
__C.wandb.args.project = __C.project
__C.wandb.args.job_type = "train"
__C.wandb.args.entity = None
__C.wandb.args.name = None
__C.wandb.args.group = None
__C.wandb.args.tags = None
__C.wandb.args.notes = None
__C.wandb.args.config_exclude_keys = []
# optional watch entry.
# __C.wandb.watch = ...

__C.dataloader = CfgNode()
__C.dataloader.name = "torch.utils.data.DataLoader"
__C.dataloader.args = CfgNode(new_allowed=True)
__C.dataloader.args.batch_size = 1
__C.dataloader.args.num_workers = 2
__C.dataloader.args.prefetch_factor = 2
__C.dataloader.sampler = CfgNode()
__C.dataloader.sampler.name = "torch.utils.data.distributed.DistributedSampler"
__C.dataloader.sampler.args = CfgNode(new_allowed=True)

__C.dist = CfgNode()
__C.dist.backend = 'nccl'
__C.dist.init_method = 'env://'

__C.register_renamed_key('data.channels', 'data.in_channels')
__C.register_deprecated_key('data.channels')


def get_default_config():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return __C.clone()


def get_base_config(cmdline: Union[str, tuple, list, None] = None,
                    default: Union[CfgNode, None] = None) -> CfgNode:
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=[], action='append', help='config files')
    parser.add_argument('-X', '--overrides', nargs='*', default=(),
                        help='override multiple config field, '
                             f'e.g.: {sys.argv[0]} -X model.drop_rate=4.5 data.spatial_shape="(10,10,10)"')
    parser.add_argument('-x', '--override', nargs='+', action='append', default=[],
                        help='override a single config field. '
                             f'e.g.: {sys.argv[0]} -x model.drop_rate=4.5 -x data.spatial_shape="(10,10,10)"')

    if isinstance(cmdline, str):
        cmds = cmdline.split()
    elif isinstance(cmdline, (tuple, list)):
        cmds = cmdline
    else:
        cmds = cmdline
    args = parser.parse_args(cmds)

    if default is None:
        default = get_default_config()

    for cfg in args.config:
        default.merge_from_file(cfg)

    fields = []
    for field in args.overrides:
        try:
            key, value = field.split('=', maxsplit=1)
        except ValueError:
            raise SyntaxError(f"bad command line at `{field}` in the -X option")
        fields.append(key)
        fields.append(value)
    default.merge_from_list(fields)

    fields.clear()
    for kv in args.override:
        if len(kv) == 1:
            try:
                kv = kv[0].split('=', maxsplit=1)
            except ValueError:
                raise SyntaxError(f"bad command line at `-x {kv[0]}`")

        if len(kv) == 2:
            fields.extend(kv)
        else:
            raise SyntaxError(f"bad command line at `-x {''.join(kv)}`")

    default.merge_from_list(fields)

    return default


_VALID_TYPES = {tuple, list, str, int, float, bool, type(None)}


def convert_to_dict(cfg_node, key_list=()):
    """ Convert a config node to dictionary """
    if not isinstance(cfg_node, CfgNode):
        if type(cfg_node) not in _VALID_TYPES:
            print("Key {} with value {} is not a valid type; valid types: {}".format(
                ".".join(key_list), type(cfg_node), _VALID_TYPES), )
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v, key_list + (k, ))
        return cfg_dict


def get_config(cmdline: Union[str, tuple, list, None] = None,
               default: Union[CfgNode, None] = None) -> CfgNode:
    """ init the config with `default` if given, or the system builtin, and then override fields by command line.
    :param cmdline: command line specified fields to override the defaults.
    :param default: the default config
    :return: the config.
    :rtype: CfgNode
    """
    cfg = __C.clone()
    if default is not None:
        cfg.merge_from_other_cfg(default)
    cfg = get_base_config(cmdline, cfg)

    if cfg.run.gpus < 0:
        cfg.run.gpus = int(os.environ.get("WORLD_SIZE", 1 if torch.cuda.is_available() else 0))
    cfg.run.rank = int(os.environ.get("RANK", 0))
    cfg.run.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if cfg.run.gpus > 0:
        cfg.run.device = f'cuda:{cfg.run.local_rank}'

    if cfg.data.name in ('ColaDataset', 'ColaWithRandomWindowDataset'):
        if hasattr(cfg.data.args, 'subdir'):
            cfg.data.args.subdir = cfg.data.args.subdir or 'cosmo-simu/sim-cola-preprocessed-128'
        else:
            cfg.data.args.subdir = 'cosmo-simu/sim-cola-preprocessed-128'
        cfg.data.train_args.choose_file_by_seed = "lambda seed: seed != 24"
        cfg.data.valid_args.choose_file_by_seed = "lambda seed: seed == 24"

    if cfg.data.name in ('ColaDataset', 'ColaWithRandomWindowDataset'):
        metrics = cfg.validator.metrics
        metrics.mae = CfgNode()
        metrics.mse = CfgNode()
        metrics.mae.name = 'torchmetrics.MeanAbsoluteError'
        metrics.mae.args = CfgNode(new_allowed=True)
        metrics.mse.name = 'torchmetrics.MeanSquaredError'
        metrics.mse.args = CfgNode(new_allowed=True)

    # wandb.args.group
    if cfg.wandb.args.group is None:
        cfg.wandb.args.group = class_name_to_tag(cfg.data.name) or cfg.wandb.args.group

    # wandb.args.tags
    fix_list_cfg_field(cfg.wandb.args, "tags")
    for cls_name in (cfg.data.name, cfg.model.name):
        tag = class_name_to_tag(cls_name)
        if tag is not None and tag not in cfg.wandb.args.tags:
            cfg.wandb.args.tags.append(tag)

    # wandb.args.name
    if cfg.wandb.args.name is None:
        faker = Faker()
        cfg.wandb.args.name = f"{cfg.model.name}-{faker.word('noun')}"
    elif cfg.model.name not in cfg.wandb.args.name:
        candidate_model_tag = class_name_to_tag(cfg.model.name)
        if not (candidate_model_tag is not None and candidate_model_tag in cfg.wandb.args.name):
            cfg.wandb.args.name = f"{cfg.model.name}-{cfg.wandb.args.name}"

    check_config(cfg)
    cfg.freeze()
    return cfg


def fix_list_cfg_field(cfg: CfgNode, key):
    if key in cfg:
        if isinstance(cfg[key], str):
            cfg[key] = cfg[key].split(',')
        elif cfg[key] is None:
            cfg[key] = []
        else:
            cfg[key] = list(cfg[key] or [])


__REGISTERED_CLASS_NAME_TO_TAG = {
    "ColaDataset": "cola",
    "ColaWithRandomWindowDataset": "cola",
    "BraTS21WithRandomWindowDataset": "brats21",
    "BraTS21Dataset": "brats21",
    "SwinAFNO": "swin-afno",
    "RwinAFNO": "rwin-afno",
    "SwinUNETR": "swin-unetr",
    "AFNONet": "afno",
    "AFNO": "afno",
}


def class_name_to_tag(name):
    name = name.split('.', maxsplit=1)[-1]
    return __REGISTERED_CLASS_NAME_TO_TAG.get(name, None)


def check_config(cfg: CfgNode):
    run_config = cfg.run  # type: CfgNode
    if run_config.eval_every_steps % run_config.visualize_interval != 0:
        warnings.warn("`run.eval_every_steps` is not evenly divisible by `run.eval_every_steps`")

    if run_config.save_interval % run_config.eval_every_steps != 0:
        warnings.warn("`run.eval_every_steps` is not evenly divisible by `run.save_interval`, "
                      "which will cause the save function fail. ")
        raise ValueError("bad configuration: run.eval_every_steps` is not evenly divisible by `run.save_interval`")

    metric_keys = list(cfg.validator.metrics.keys())
    if cfg.validator.main_metric is None and len(metric_keys) == 1:
        cfg.validator.main_metric = metric_keys[0]
    cfg.validator.main_metric = None if cfg.validator.main_metric is None else cfg.validator.main_metric.upper()

    if cfg.optimizer.freeze_params is not None:
        assert isinstance(cfg.optimizer.freeze_params, str)
        cfg.optimizer.freeze_params = cfg.optimizer.freeze_params.strip()


def _demo():
    import sys
    from gkai.utils import flatten_toplevel
    cfg = get_config(sys.argv[1:] + ["-x", "wandb.args.tags", "a,b,c", "-x", "model.pretrained_path=None",
                                     "-x", "data.train_args.mixup.re_prob=0.2"])
    # cfg.merge_from_list(["wandb.args.tags", "[\"gkai\", \"demo\"]"])
    cfg.freeze()
    print(cfg)
    print(type(cfg.model.pretrained_path), cfg.model.pretrained_path)
    print(dict(**cfg.optimizer.args))

    cfg_dict = convert_to_dict(cfg)
    cfg_dict = flatten_toplevel(cfg_dict)
    print(cfg_dict)


if __name__ == '__main__':
    _demo()
