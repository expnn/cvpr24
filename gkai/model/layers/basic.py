# -*- coding: utf-8 -*-

import torch
from torch import nn


class Permute(nn.Module):
    def __init__(self, dims, make_contiguous=False):
        super().__init__()
        self.dims = dims
        self.make_contiguous = make_contiguous

    def forward(self, x):
        x = torch.permute(x, self.dims)
        if self.make_contiguous:
            x = x.contiguous()
        return x


class MeanPool(nn.Module):
    def __init__(self, dims, keepdim=False):
        super().__init__()
        self.dims = dims
        self.keepdim = keepdim

    def forward(self, x):
        return torch.mean(x, dim=self.dims, keepdim=self.keepdim)


class MaxPool(nn.Module):
    def __init__(self, dims, keepdim=False):
        super().__init__()
        self.dims = dims
        self.keepdim = keepdim

    def forward(self, x):
        return torch.amax(x, dim=self.dims, keepdim=self.keepdim)


class PatchEmbed(nn.Module):
    def __init__(self, spatial_size=(224, 224), patch_size=(16, 16), in_channels=3, embed_dim=768):
        super().__init__()
        self.ndims = ndims = len(spatial_size)
        assert ndims == len(patch_size)
        self.spatial_size = tuple(spatial_size)
        self.patch_size = patch_size
        self.num_patches = tuple(x // y for x, y in zip(spatial_size, patch_size))
        self.embed_dim = embed_dim

        if ndims == 2:
            self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        elif ndims == 3:
            self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        elif ndims == 1:
            self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        else:
            raise ValueError(f"{self.__class__.__name__} only support 1D/2D/3D inputs")

    def forward(self, x):
        b, d, *spatial_dims = x.shape  # (B, Ci, H, W, ...)
        assert tuple(spatial_dims) == self.spatial_size, \
            f"Input image size '{spatial_dims}' doesn't match model '{self.spatial_size}'."
        x = self.proj(x)  # (B, C_o, H/p[0], W/p[1], ...)
        b, d, *spatial_dims = x.shape
        assert d == self.embed_dim
        x = (
            x.flatten(2)      # (B, Co, H/p[0] * W/p[1] * ...)
            .transpose(1, 2)  # (B, H/p[0] * W/p[1] * ..., Co)
            .reshape(b, *spatial_dims, self.embed_dim)  # (B, H/p[0], W/p[1], ..., Co)
        )
        return x
