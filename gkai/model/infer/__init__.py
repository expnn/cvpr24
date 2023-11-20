# -*- coding: utf-8 -*-
import torch
from torch import nn
from .brats import BraTS21DiceInferencer
from .brats import BraTS21MultiLabelInferencer
from .imgnet import ImageNetEnsembleClassifier


class Inferencer(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        with torch.no_grad():
            return self.model(x)
