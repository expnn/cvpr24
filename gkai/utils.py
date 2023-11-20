# -*- coding: utf-8 -*-

import sys
import types
import logging
import importlib
import torch.distributed as dist
from itertools import zip_longest
import wandb

from gkai.config import convert_to_dict, CfgNode


__ALL_LOGGERS = {}


class Formatter(logging.Formatter):
    debug_fmt = logging.Formatter(
        "[%(asctime)s %(levelname)s %(name)s %(pathname)s:%(lineno)d]: %(message)s")

    def __init__(self, fmt="[%(asctime)s %(levelname)s %(name)s]: %(message)s"):
        super().__init__(fmt)

    def format(self, record):
        if record.levelno == logging.DEBUG:
            result = self.debug_fmt.format(record)
        else:
            # Call the original formatter class to do the grunt work
            result = logging.Formatter.format(self, record)
        return result


def group(n, iterable, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)


def setup(config: CfgNode):
    import random
    import numpy as np
    import torch
    import wandb

    rank = 0
    if config.run.gpus > 1:
        kwargs = convert_to_dict(config.dist)
        dist.init_process_group(**kwargs)
        rank = dist.get_rank()
    # see https://discuss.pytorch.org/t/extra-10gb-memory-on-gpu-0-in-ddp-tutorial/118113
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()

    seed_str = config.run.seed
    part_len = len(seed_str) // 4
    if len(seed_str) % 4 != 0:
        part_len += 1
    parts = list(map(lambda x: ''.join(x), group(part_len, seed_str, '')))

    random.seed(hash(parts[0]) % 2 ** 32)
    np.random.seed(hash(parts[1]) % 2 ** 32)
    torch.manual_seed(hash(parts[2]) % 2 ** 32)
    torch.cuda.manual_seed_all(hash(parts[3]) % 2 ** 32)

    get_logger('gkai')

    if config.wandb.enabled:
        if (config.run.gpus > 1 and rank == 0) or config.run.gpus <= 1:
            cfg_dict = convert_to_dict(config)
            cfg_dict.pop('dist', None)
            cfg_dict.pop('wandb', None)
            wandb.init(config=cfg_dict, **config.wandb.args)


def cleanup(config: CfgNode):
    logging.shutdown()

    rank = 0
    if config.run.gpus > 1:
        rank = dist.get_rank()

    if config.wandb.enabled:
        if (config.run.gpus > 1 and rank == 0) or config.run.gpus <= 1:
            wandb.finish()

    if config.run.gpus > 1:
        dist.destroy_process_group()


def get_logger(name, level=logging.INFO):
    global __ALL_LOGGERS
    if name in __ALL_LOGGERS:
        return __ALL_LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(level=level)
    logger.propagate = False

    hdlr = logging.StreamHandler(sys.stdout)
    hdlr.setFormatter(Formatter())
    logger.addHandler(hdlr)

    __ALL_LOGGERS[name] = logger

    return logger


def flatten_toplevel(dic: dict):
    new_dic = {}
    for k, v in dic.items():
        if isinstance(v, dict):
            for sub_key, sub_val in v.items():
                new_dic[f"{k}_{sub_key}"] = sub_val
        else:
            new_dic[k] = v
    return new_dic


def import_class(path, package=None):
    mod_cls = path.rsplit('.', maxsplit=1)
    if len(mod_cls) == 2:
        mod, cls = mod_cls
        mod = mod or '.'
    else:
        mod = '.'
        cls = mod_cls[0]

    mod = importlib.import_module(mod, package)  # 经测试, 当mod是绝对路径时, 不会使用 package 的值.
    return getattr(mod, cls)


def get_clean_path(path):
    if path is None or not isinstance(path, str):
        return None
    return path.strip()


def map_structure(fn, struct, leafs=(str, int, float, type(None),
                                     types.FunctionType, types.LambdaType,
                                     types.BuiltinFunctionType),
                  context=None, logger=None):
    if logger is None:
        class TmpLogger(object):
            pass
        logger = TmpLogger()
        logger.warning = print

    if isinstance(struct, list):
        if type(struct) not in leafs:
            return [map_structure(fn, x, leafs, context) for x in struct]
        else:
            return struct
    elif isinstance(struct, tuple):
        if type(struct) in leafs:
            return struct
        else:
            return tuple(map_structure(fn, x, leafs, context) for x in struct)
    elif isinstance(struct, dict):
        if type(struct) in leafs:
            return struct
        else:
            tmp = {}
            for k, v in struct.items():
                tmp[k] = map_structure(fn, v, leafs, context)
            return tmp
    else:
        if not isinstance(struct, leafs):
            logger.warning("Find type {} in structure which is neither one of [dict, list, tuple] "
                           "nor one of leafs (a.k.a {}). Treat it as a leaf".format(type(struct), list(leafs)))
            return struct
        else:
            if context:
                return fn(struct, context)
            else:
                return fn(struct)


def _demo():
    # get_logger('gkai')
    logger = get_logger('gkai.utils', logging.DEBUG)
    logger.info("info message: %s", logging.INFO)
    logger.debug("debug message: %s", logging.DEBUG)
    logger.warning('warning message: %s', logging.WARNING)
    logger.error('error message: %s', logging.ERROR)

    scheduler = import_class('LinearWarmupCosineAnnealingLR', 'gkai.optim')
    print(scheduler.__name__)
    scheduler = import_class('.LinearWarmupCosineAnnealingLR', 'gkai.optim')
    print(scheduler.__name__)
    scheduler = import_class('gkai.model.ToyModel', 'gkai.model')
    print(scheduler.__name__)


if __name__ == '__main__':
    _demo()
