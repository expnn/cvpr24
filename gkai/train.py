# -*- coding: utf-8 -*-

import os
import sys
import torch
import wandb
import json
import math
import numpy as np

from torch import nn
from time import time, strftime
from datetime import timedelta
import torch.distributed as dist
from functools import partial
from torch.nn.parallel import DistributedDataParallel
from collections import deque
from monai.utils import ensure_tuple_rep

from gkai.model import create_model, create_loss, create_metrics, create_inferencer
from gkai.config import get_config
from gkai.utils import setup, cleanup, get_logger, map_structure
from gkai.data import create_dataloader
from gkai.optim import create_optimizer
from gkai.config import CACHE_DIR, CfgNode


def main():
    cfg = get_config(sys.argv[1:])
    setup(cfg)
    logger = get_logger('gkai.train')

    device_id = 0
    rank = 0

    if cfg.run.gpus > 1:
        pid = os.getpid()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device_id = rank % torch.cuda.device_count()

        logger.info(f'pid: {pid} - rank: {rank} - world size: {world_size}')
        assert cfg.run.gpus == world_size
        if rank == 0:
            print(cfg)
    else:
        print(cfg)
    device = torch.device(f'cuda:{device_id}')

    model = create_model(cfg)
    train_dataloader = create_dataloader(cfg, training=True)
    valid_dataloader = create_dataloader(cfg, training=False)

    total_steps = cfg.run.epochs * len(train_dataloader)
    validator = Validator(cfg, model, device_id, valid_dataloader, total_steps)

    save_path = os.path.join(CACHE_DIR, cfg.model.model_save_subdir, f"{strftime('%F-%H.%M.%S')}-{cfg.run.id}")
    saver = Saver(model, save_path, cfg.run.save_interval, main_metric=cfg.validator.main_metric,
                  larger_better=cfg.validator.larger_better)

    if rank == 0 and cfg.wandb.enabled:
        dump_cfg_path = os.path.join(save_path, 'config.yaml')
        with open(dump_cfg_path, 'w', encoding='utf8') as f:
            f.write(cfg.dump())
        artifact = wandb.Artifact('frozen_config', type='config')
        artifact.add_file(local_path=dump_cfg_path)
        wandb.log_artifact(artifact)

    loss_fn = create_loss(cfg)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        logger.info("Total parameters count: %d", total_params)
        logger.info("Total epochs: %d, dataset size: %d, total steps: %d",
                    cfg.run.epochs, len(train_dataloader), total_steps)
        if cfg.wandb.enabled:
            wandb.run.summary['num_parameters'] = total_params

    optimizer, lr_scheduler, unfreeze = create_optimizer(cfg, model, total_steps)
    clipper = GradientClipper(cfg, model.parameters(), rank)

    global_step = 0
    loss = 1e10
    metrics = None
    if not cfg.run.skip_initial_validation:
        metrics = validator.validate(step=0)
    start_time = time()
    visualizer = Visualizer(cfg, rank, start_time, total_steps, cfg.run.epochs, logger, model, validator.metrics)
    visualizer.log(metrics, step=0, epoch=0)

    model.train()
    for epoch in range(1, cfg.run.epochs+1):
        if cfg.run.gpus > 1:
            # noinspection PyUnresolvedReferences
            train_dataloader.sampler.set_epoch(epoch)
        for data in train_dataloader:
            x, y = map_structure(
                fn=partial(copy_to_device, device=device),
                struct=data,
                leafs=(torch.Tensor, np.ndarray)
            )
            y_hat = model(x)
            optimizer.zero_grad()
            loss = loss_fn(y_hat, y)
            if not torch.isfinite(loss):
                logger.error("Loss is not finite")
                break
            loss.backward()
            metrics = clipper()
            optimizer.step()
            lr_scheduler.step()

            global_step += 1
            metrics.update(validator.validate(global_step))
            saver.save('model-{epoch:04d}-{step:06d}.pt', metrics, step=global_step, epoch=epoch)
            metrics['loss'] = loss.item()
            metrics['lr'] = lr_scheduler.get_last_lr()[-1]
            visualizer.log(metrics, step=global_step, epoch=epoch)
        clipper.summarize()
        if not torch.isfinite(loss):
            break
        unfreeze(epoch)

    cleanup(cfg)


class GradientClipper(object):
    def __init__(self, cfg: CfgNode, parameters, rank):
        self.parameters = list(parameters)
        self.wandb_enabled = cfg.wandb.enabled and rank == 0
        self.max_norm = cfg.gradient_clipper.max_norm
        self.norm_type = cfg.gradient_clipper.norm_type
        self.enabled = cfg.gradient_clipper.enabled
        self.log_name = cfg.gradient_clipper.log_name
        self.all_norms = []

    def __call__(self):
        if self.enabled:
            total_norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.max_norm, self.norm_type)
            val = total_norm.item()
            self.all_norms.append(val)
            return {self.log_name: val}
        else:
            return {}

    def reset(self):
        self.all_norms.clear()

    def summarize(self):
        if self.wandb_enabled and len(self.all_norms) > 0:
            # wandb.run.summary[f"{self.log_name}_stats"] = np.array(self.all_norms)
            arr = np.array(self.all_norms)
            hist = np.histogram(arr, bins=40)
            percentiles = np.percentile(arr, [10, 25, 50, 75, 90])
            norm_grad_stats = {
                "10%": percentiles[0],
                "25%": percentiles[1],
                "50%": percentiles[2],
                "75%": percentiles[3],
                "90%": percentiles[4],
                "min": np.min(arr),
                "max": np.max(arr),
                "var": np.var(arr),
                "mean": np.mean(arr),
            }
            wandb.run.summary[f"{self.log_name}_stats"] = norm_grad_stats
            wandb.log({f"{self.log_name}_hist": wandb.Histogram(np_histogram=hist)})


class Visualizer:
    def __init__(self, config, rank, start_time, total_steps, total_epochs, logger, model, metrics):
        self.model = model
        self.wandb_enabled = config.wandb.enabled
        self.report_interval = config.run.report_interval
        self.visualize_interval = config.run.visualize_interval
        self.start_time = start_time
        self.logger = logger
        self.total_steps = total_steps
        self.total_epochs = total_epochs
        self.rank = rank

        if self.wandb_enabled and rank == 0:
            wandb.define_metric("loss", summary="min")
            for key, m in metrics.items():
                # 有时一个指标会返回多个值. 可以使用 value_names 为每个值定义一个名字
                cfg = config.validator.metrics[key]
                if hasattr(cfg, 'value_names'):
                    for name in cfg.value_names:
                        wandb.define_metric(
                            f"{key}_{name}".upper(),
                            summary="max" if m.higher_is_better else "min")
                    if hasattr(cfg, 'reduction'):
                        wandb.define_metric(
                            key.upper(),
                            summary="max" if m.higher_is_better else "min")
                else:
                    wandb.define_metric(
                        key.upper(),
                        summary="max" if m.higher_is_better else "min")
            if hasattr(config.wandb, 'watch'):
                wandb.watch(model, **config.wandb.watch)

    def log(self, data: dict, epoch, step):
        if self.rank != 0 or not bool(data):
            return

        def format_value(val):
            if isinstance(val, float):
                return f"{val:.4g}"
            else:
                return f"{val}"

        if self.wandb_enabled and step % self.visualize_interval == 0:
            wandb.log(data, step=step)

        if step % self.report_interval == 0 or step == self.total_steps:
            current = time()
            speed = step / (current - self.start_time + 1e-10)
            if speed != 0:
                eta = (self.total_steps - step) / speed
                data['eta'] = timedelta(seconds=eta)
            else:
                data['eta'] = 'NaN'
            message = ' - '.join(map(lambda kv: f"{str(kv[0]).upper()}: {format_value(kv[1])}", data.items()))
            self.logger.info(f"EPOCH: {epoch}/{self.total_epochs} - STEP: {step}/{self.total_steps} - {message}")


class Validator:
    def __init__(self, config: CfgNode, model: nn.Module, device_id, loader, total_steps):
        self.loader = loader
        self.device = torch.device(f'cuda:{device_id}')
        self.freq = config.run.eval_every_steps
        self.total_steps = total_steps
        self.inferencer = create_inferencer(config, model)
        self.metric_configs = config.validator.metrics
        self.metrics = nn.ModuleDict(create_metrics(config))
        self.metrics.to(self.device)

    def validate(self, step):
        if step % self.freq != 0 and step != self.total_steps:
            return {}
        self.inferencer.eval()
        assert not self.inferencer.model.training
        with torch.no_grad():
            metrics = self._validate()
        self.inferencer.train()
        assert self.inferencer.model.training
        return metrics

    def _validate(self):
        for data in self.loader:
            x, target = map_structure(
                fn=partial(copy_to_device, device=self.device),
                struct=data,
                leafs=(torch.Tensor, np.ndarray)
            )
            y_pred = self.inferencer(x)
            for m in self.metrics.values():
                m.update(y_pred, target)
        metrics = {}
        for key, m in self.metrics.items():
            metric_values = m.compute().cpu().numpy().ravel()

            # 有时一个指标会返回多个值. 可以使用 value_names 为每个值定义一个名字
            cfg = self.metric_configs[key]
            if hasattr(cfg, 'value_names'):
                metric_names = ensure_tuple_rep(cfg.value_names, len(metric_values))
                for name, val in zip(metric_names, metric_values):
                    metrics[f"{key}_{name}".upper()] = float(val)
                if hasattr(cfg, 'reduction'):
                    metrics[key.upper()] = float(self.reduce_array(metric_values, 'mean'))
            else:
                assert len(metric_values) == 1
                metrics[key.upper()] = float(metric_values[0])
            m.reset()
        return metrics

    @staticmethod
    def reduce_array(val, method):
        if method == 'mean':
            return val.mean()
        elif method == 'max':
            return
        elif method == 'min':
            return val.min()
        elif method == 'sum':
            return val.sum()
        elif method == 'none' or method is None:
            return val.mean()


class SizedFIFOQueue:
    def __init__(self, iterable=(), maxlen: int = None):
        if maxlen is None:
            maxlen = len(iterable)
        self.buffer = deque(iterable, maxlen)

    def append(self, item):
        ret = None
        if len(self.buffer) >= self.buffer.maxlen:
            ret = self.buffer.popleft()
        self.buffer.append(item)
        return ret


class Saver:
    def __init__(self, model, root, freq, main_metric=None, larger_better=True, max_checkpoint=5):
        if isinstance(model, (DistributedDataParallel, torch.nn.parallel.DataParallel)):
            self.model = model.module
        else:
            self.model = model
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.distributed = isinstance(model, DistributedDataParallel)
        self.freq = freq
        self.best_metric = -math.inf if larger_better else math.inf
        self.larger_better = larger_better
        self.main_metric = main_metric
        self.max_checkpoint = max_checkpoint

        self.saved_paths = SizedFIFOQueue([], maxlen=max_checkpoint)
        self.best_model = None

    def better(self, cur_metric):
        if self.larger_better:
            return cur_metric > self.best_metric
        else:
            return cur_metric < self.best_metric

    def should_save(self):
        if not self.distributed:
            return True
        return dist.get_rank() == 0

    def save(self, path: str, metrics: dict, /, *arg, step: int, **kwargs):
        if step % self.freq != 0:
            return

        if self.should_save():
            kwargs['step'] = step
            path = path.format(*arg, **kwargs)
            path = os.path.join(self.root, path)
            torch.save(self.model.state_dict(), path)

            oldest_path = self.saved_paths.append(path)
            if oldest_path:
                os.remove(oldest_path)

            if self.main_metric and self.main_metric in metrics:
                cur_metric = metrics[self.main_metric]
                if self.better(cur_metric):
                    if self.best_model and os.path.exists(self.best_model):
                        os.unlink(self.best_model)
                    best_dir, best_base = os.path.split(path)
                    best_path = os.path.join(best_dir, f"best-{best_base}")
                    self.best_model = best_path
                    os.link(path, best_path)
                    self.best_metric = cur_metric
                    with open(os.path.join(best_dir, 'best-info.json'), 'wt', encoding='utf8') as fp:
                        json.dump({
                            'step': step,
                            'main_metric': self.main_metric,
                            'metrics': metrics
                        }, fp)
        if self.distributed:
            dist.barrier()


def copy_to_device(tensor, device):
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device, non_blocking=True)
    else:
        return torch.tensor(tensor).to(device, non_blocking=True)


if __name__ == '__main__':
    main()
