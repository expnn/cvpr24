# -*- coding: utf-8 -*-

import einops
import math
import torch
from torch import nn
from abc import ABC, abstractmethod
from typing import Union, Sequence, Literal
from monai.networks.blocks import Convolution
from monai.utils import ensure_tuple_rep
from einops import rearrange

from gkai.utils import import_class
from gkai.model.layers.basic import Permute, MeanPool, MaxPool


class Head(nn.Module, ABC):
    def __init__(self, ndim, in_channels):
        super().__init__()
        self.ndim = ndim
        self.in_channels = in_channels

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass


class PredictionHead(Head):
    def __init__(self, ndim, in_channels, out_channels, patch_size, num_patches):
        super().__init__(ndim, in_channels)
        self.patch_size = ensure_tuple_rep(patch_size, ndim)
        self.num_patches = ensure_tuple_rep(num_patches, ndim)
        self.head = nn.Linear(in_channels, out_channels * math.prod(self.patch_size), bias=False)

    def forward(self, x):
        x = self.head(x)
        x = self.postprocess_output(x, self.num_patches, self.patch_size)
        return x

    @staticmethod
    def postprocess_output(x, num_patches, patch_size):
        ndim = x.ndim - 2
        if ndim == 1:
            spec = "b l (c p) -> b c (l p)"
            kwargs = {
                "p": patch_size[0],
                "l": num_patches[0],
            }
        elif ndim == 2:
            spec = "b h w (c p1 p2) -> b c (h p1) (w p2)"
            kwargs = {
                "p1": patch_size[0],
                "p2": patch_size[1],
                "h": num_patches[0],
                "w": num_patches[1],
            }
        else:
            spec = "b h w d (c p1 p2 p3) -> b c (h p1) (w p2) (d p3)"
            kwargs = {
                "p1": patch_size[0],
                "p2": patch_size[1],
                "p3": patch_size[2],
                "h": num_patches[0],
                "w": num_patches[1],
                "d": num_patches[2],
            }

        return rearrange(x, spec, **kwargs)


class ConvClassificationHeadV1(Head):
    def __init__(
            self,
            ndim: int,
            in_channels: int,
            out_channels: Sequence[int],
            num_classes: int,
            kernel_sizes: Union[Sequence[int], int] = 2,
            flattened_size: int = 4,
            activation: Union[tuple, str, None] = "PRELU",
            norm_layer: Union[str, tuple, None] = "INSTANCE",
            dropout: Union[float, None] = None,
            pool_method: Union[Literal['avg', 'mean', 'max', 'none'], None] = 'avg',
    ):
        super().__init__(ndim, in_channels)
        num_layers = len(out_channels)
        kernel_sizes = ensure_tuple_rep(kernel_sizes, num_layers)
        strides = kernel_sizes

        in_channels = (in_channels, *out_channels[:-1])
        pool_method = pool_method or 'none'
        if pool_method == 'none':
            pool_layer = nn.Flatten()
        elif pool_method in ('avg', 'mean'):
            pool_layer = MeanPool(dims=tuple(range(2, 2 + ndim)))
            flattened_size = 1
        else:
            assert pool_method == 'max', f'Expecting pool_method in ("avg", "mean", "max", "none"), got {pool_method}'
            pool_layer = MaxPool(dims=tuple(range(2, 2 + ndim)))
            flattened_size = 1

        # noinspection PyTypeChecker
        self.head = nn.Sequential(*[
            Permute(dims=(0, ndim + 1, *range(1, 1 + ndim)), make_contiguous=True),
            *[Convolution(
                spatial_dims=ndim,
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=kernel_sizes[i],
                strides=strides[i],
                act=activation,
                norm=norm_layer,
                dropout=dropout,
                padding='valid',
            ) for i in range(num_layers)],
            pool_layer,
            nn.Linear(flattened_size * out_channels[-1], num_classes),
        ])

    def forward(self, x):
        return self.head(x)


class ConvClassificationHeadV2(Head):
    def __init__(
            self,
            ndim: int,
            in_channels: int,
            out_channels: Sequence[int],
            num_classes: int,
            kernel_sizes: Union[Sequence[int], int] = 2,
            activation: Union[tuple, str, None] = "PRELU",
            norm_layer: Union[str, tuple, None] = "INSTANCE",
            dropout: Union[float, None] = None,
            ada_pool_size: Union[int, Sequence[int]] = 1,
            ada_pool_method: Literal['avg', 'mean', 'max'] = 'avg',
    ):
        super().__init__(ndim, in_channels)
        ada_pool_size = ensure_tuple_rep(ada_pool_size, ndim)
        num_layers = len(out_channels)
        kernel_sizes = ensure_tuple_rep(kernel_sizes, num_layers)
        strides = kernel_sizes
        in_channels = (in_channels, *out_channels[:-1])

        self.permute = Permute(dims=(0, ndim + 1, *range(1, 1 + ndim)), make_contiguous=True)
        # noinspection PyTypeChecker
        self.conv_blocks = nn.Sequential(*[Convolution(
            spatial_dims=ndim,
            in_channels=in_channels[i],
            out_channels=out_channels[i],
            kernel_size=kernel_sizes[i],
            strides=strides[i],
            act=activation,
            norm=norm_layer,
            dropout=dropout,
            padding='valid',
        ) for i in range(num_layers)])

        if ada_pool_method in ('avg', 'mean'):
            pool_layer = import_class(f"AdaptiveAvgPool{ndim}d", "torch.nn.modules.pooling")
        else:
            assert ada_pool_method == 'max'
            pool_layer = import_class(f"AdaptiveMaxPool{ndim}d", "torch.nn.modules.pooling")
        self.avgpool = pool_layer(ada_pool_size)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(math.prod(ada_pool_size) * out_channels[-1], num_classes)

    def forward(self, x):
        x = self.permute(x)
        x = self.conv_blocks(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class PoolClassificationHead(Head):
    def __init__(self, ndim, in_channels, num_classes: int,
                 pool_method: Literal['avg', 'mean', 'max'] = 'avg'):
        super().__init__(ndim, in_channels)
        self.pool_method = pool_method
        if pool_method in ('avg', 'mean'):
            self.pool = MeanPool(dims=tuple(range(1, 1 + ndim)))
        else:
            assert pool_method == 'max'
            self.pool = MaxPool(dims=tuple(range(1, 1 + ndim)))
        self.head = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.head(x)
        return x


class EnsembleClassificationHead(Head):
    def __init__(self, ndim, in_channels,
                 num_classes: int, ada_pool_size: Union[Sequence[int], int] = 16,
                 ada_pool_method: Literal['avg', 'mean', 'max'] = 'avg', shared: bool = True):
        super().__init__(ndim, in_channels)
        ada_pool_size = ensure_tuple_rep(ada_pool_size, ndim)
        if ada_pool_method in ('avg', 'mean'):
            pool_layer = import_class(f"AdaptiveAvgPool{ndim}d", "torch.nn.modules.pooling")
        else:
            assert ada_pool_method == 'max'
            pool_layer = import_class(f"AdaptiveMaxPool{ndim}d", "torch.nn.modules.pooling")

        self.permute = Permute(dims=(0, ndim + 1, *range(1, 1 + ndim)), make_contiguous=True)
        self.pool = pool_layer(ada_pool_size)
        if shared:
            self.head = Convolution(
                spatial_dims=ndim,
                in_channels=in_channels,
                out_channels=num_classes,
                kernel_size=1,
                strides=1,
                bias=True,
                act=None,
                norm=None,
                conv_only=True,
            )
        else:
            self.head = PointwiseClassifiers(ada_pool_size, in_channels, num_classes)

    def forward(self, x):
        x = self.permute(x)
        x = self.pool(x)
        x = self.head(x)
        return x


class PointwiseClassifiers(nn.Module):
    def __init__(self, spatial_shape, feat_size, num_class):
        super().__init__()
        s = math.prod(spatial_shape)
        w = torch.empty(num_class, s, feat_size)
        for i in range(s):
            nn.init.orthogonal_(w[:, i, :])
        self.weights = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(num_class, s))
        self.spatial_shape = spatial_shape

    def forward(self, x):
        # x: (B, d, H, W)
        x = torch.flatten(x, start_dim=2, end_dim=-1)  # (B, d, S)
        x = einops.einsum(x, self.weights, 'b d s, c s d -> b c s')  # (B, C, S)
        x = x + self.bias
        x = torch.unflatten(x, dim=-1, sizes=self.spatial_shape)  # (B, C, H, W)
        return x


class MixPoolClassificationHead(Head):
    def __init__(self, ndim, in_channels, num_classes: int, avg_ratio: float = 0.5):
        super().__init__(ndim, in_channels)
        self.avg_channels = int(avg_ratio * in_channels)
        self.max_channels = in_channels - self.avg_channels
        self.avg_pool = MeanPool(dims=tuple(range(1, 1 + ndim)))
        self.max_pool = MaxPool(dims=tuple(range(1, 1 + ndim)))
        self.tanh = nn.Tanh()
        self.head = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        n, *_, c = x.shape
        y = torch.zeros([n, c], device=x.device)
        y[:, :self.avg_channels] = self.avg_pool(x[..., :self.avg_channels])
        y[:, self.avg_channels:] = self.max_pool(x[..., self.avg_channels:])
        y = self.tanh(y)
        return self.head(y)


class CenterPoolClassificationHead(Head):
    def __init__(self, ndim, in_channels, num_classes: int):
        super().__init__(ndim, in_channels)
        self.head = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        _, *spatial_size, _ = x.shape
        index = tuple(spatial_size[i] // 2 for i in range(self.ndim))
        index = (slice(None), *index)
        x = x[index]
        return self.head(x)


def _demo():
    classifier = PointwiseClassifiers((4, 4), 8, 10)
    x = torch.randn(7, 8, 4, 4)
    y = classifier(x)
    print(y.shape)
    assert y.shape == (7, 10, 4, 4)


if __name__ == '__main__':
    _demo()
