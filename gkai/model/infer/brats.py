# -*- coding: utf-8 -*-

import torch
from torch import nn
from abc import ABC, abstractmethod
from functools import partial
from typing import Tuple
from monai.inferers import sliding_window_inference


class BraTS21BaseInferencer(nn.Module, ABC):
    def __init__(
            self,
            model: nn.Module,
            inference_size: Tuple[int],
            sw_batch_size: int = 4,
            infer_overlap: float = 0.5,
            threshold: float = 0.0):
        super().__init__()
        self.model = model
        self.inferencer = partial(
            sliding_window_inference,
            roi_size=inference_size,
            sw_batch_size=sw_batch_size,
            predictor=self.model,
            overlap=infer_overlap,
        )
        self.threshold = threshold

    @abstractmethod
    def forward(self, x) -> torch.Tensor:
        pass


class BraTS21MultiLabelInferencer(BraTS21BaseInferencer):
    def forward(self, x) -> torch.Tensor:
        with torch.no_grad():
            x = self.inferencer(x)
            return x > self.threshold


class BraTS21DiceInferencer(BraTS21BaseInferencer):
    def __init__(
            self,
            model: nn.Module,
            inference_size: Tuple[int],
            sw_batch_size: int = 4,
            infer_overlap: float = 0.5,
            threshold: float = 0.5,
    ):
        super().__init__(model, inference_size, sw_batch_size, infer_overlap, threshold)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x) -> torch.Tensor:
        with torch.no_grad():
            x = self.inferencer(x)
            x = self.sigmoid(x)
            return x > self.threshold
