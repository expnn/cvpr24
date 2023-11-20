# -*- coding: utf-8 -*-

import os
import sys
import torch
from time import time
import torch.distributed as dist

from gkai.model import create_model
from gkai.config import get_config
from gkai.utils import setup, cleanup, get_logger
from gkai.data import create_dataloader
from gkai.train import Validator


def main():
    cfg = get_config(sys.argv[1:] + ["-x", "wandb.enabled=False"])
    setup(cfg)
    logger = get_logger('gkai.eval')

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

    model = create_model(cfg)
    valid_dataloader = create_dataloader(cfg, training=False)
    validator = Validator(cfg, model, device_id, valid_dataloader, total_steps=0)
    dataset_size = len(valid_dataloader) * cfg.dataloader.args.batch_size

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        logger.info("Total parameters count: %d", total_params)

    vis = Visualizer(rank, logger)
    start = time()
    metrics = validator.validate(step=0)
    end = time()
    logger.info(f"total examples: {dataset_size} - throughput: {dataset_size / (end - start):.3f}")
    vis.log(metrics)

    cleanup(cfg)


class Visualizer:
    def __init__(self, rank, logger):
        self.logger = logger
        self.rank = rank

    def log(self, data: dict):
        if self.rank != 0 or not bool(data):
            return

        def format_value(val):
            if isinstance(val, float):
                return f"{val:.4g}"
            else:
                return f"{val}"
        message = ' - '.join(map(lambda kv: f"{str(kv[0]).upper()}: {format_value(kv[1])}", data.items()))
        self.logger.info(message)


if __name__ == '__main__':
    main()
