# -*- coding: utf-8 -*-

import re
import math
import warnings
from typing import List

import torch.nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ConstantLR
from torch.optim.lr_scheduler import LRScheduler
from gkai.utils import import_class, get_logger


class LinearWarmupCosineAnnealingLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_steps: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_steps (int): Maximum number of iterations for linear warmup
            max_steps (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Compute learning rate using chainable form of the scheduler
        """
        # noinspection PyUnresolvedReferences
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.", UserWarning
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        elif self.last_epoch < self.warmup_steps:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_steps - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif self.last_epoch == self.warmup_steps:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.max_steps) % (2 * (self.max_steps - self.warmup_steps)) == 0:
            return [
                group["lr"] +
                (base_lr - self.eta_min) * (1 - math.cos(math.pi / (self.max_steps - self.warmup_steps))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps) / (self.max_steps - self.warmup_steps))) /
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps - 1) / (self.max_steps - self.warmup_steps))) *
            (group["lr"] - self.eta_min) + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """
        Called when epoch is passed as a param to the `step` function of the scheduler.
        """
        if self.last_epoch < self.warmup_steps:
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_steps - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps) / (self.max_steps - self.warmup_steps)))
            for base_lr in self.base_lrs
        ]


class LinearWarmupCosineAnnealingLRArgAdaptor(object):
    def __init__(self, warmup_portion: float, **kwargs):
        self.warmup_portion = warmup_portion
        self.kwargs = kwargs

    def __call__(self, total_steps: int, **kwargs):
        extra_args = dict(**self.kwargs)
        extra_args.update(kwargs)
        warmup_steps = int(total_steps * self.warmup_portion)
        return {
            'warmup_steps': warmup_steps,
            'max_steps': total_steps,
            **extra_args,
        }


class CosineAnnealingLRArgAdaptor(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, total_steps: int, **kwargs):
        extra_args = dict(**self.kwargs)
        extra_args.update(kwargs)
        return {
            'T_max': total_steps,
            **extra_args,
        }


def create_optimizer(config, model: torch.nn.Module, *adaptor_args, **adaptor_kwargs):
    logger = get_logger('gkai.optim')
    optim_cls = import_class(config.optimizer.name, 'gkai.optim')
    args = config.optimizer.args or {}

    # https://discuss.pytorch.org/t/best-practice-for-freezing-layers/58156/2
    frozen_params = []
    frozen_params_names = []
    if config.optimizer.freeze_params:
        parameters = []
        freeze_pat = re.compile(config.optimizer.freeze_params.strip())
        for name, p in model.named_parameters():
            if isinstance(model, DistributedDataParallel):
                name = name.split('.', maxsplit=1)[-1]
            if freeze_pat.match(name):
                if config.run.rank == 0:
                    logger.info(f"Freezes '{name}'")
                frozen_params.append(p)
                frozen_params_names.append(name)
                continue
            parameters.append(p)
    else:
        parameters = model.parameters()
    optimizer = optim_cls(parameters, **args)

    def unfreeze(epoch):
        if epoch == config.optimizer.freeze_epochs and frozen_params_names:
            optimizer.add_param_group({'params': frozen_params})
            if config.run.rank == 0:
                for name_ in frozen_params_names:
                    logger.info(f"Unfreezes '{name_}'")

    adaptor_cls_name = config.lr_scheduler.arg_adaptor.name.strip()
    kwargs = {}
    if adaptor_cls_name:
        arg_adaptor_cls = import_class(adaptor_cls_name, 'gkai.optim')
        arg_adaptor = arg_adaptor_cls(**config.lr_scheduler.arg_adaptor.args)
        kwargs = arg_adaptor(*adaptor_args, **adaptor_kwargs)
    scheduler_name = config.lr_scheduler.name.strip() if config.lr_scheduler.name else None
    if scheduler_name:
        lr_scheduler_cls = import_class(scheduler_name, 'gkai.optim')
        kwargs.update(config.lr_scheduler.args)
        lr_scheduler = lr_scheduler_cls(optimizer, **kwargs)
    else:
        lr_scheduler = ConstantLR(optimizer, 1.0, 1)

    return optimizer, lr_scheduler, unfreeze
