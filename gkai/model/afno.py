# -*- coding: utf-8 -*-
import math
import torch
import numpy as np
import torch.nn as nn
from typing import Union
from torch.utils.checkpoint import checkpoint_sequential
import torch.nn.functional as functor
from functools import partial
from typing import Callable
from timm.models.layers import DropPath, trunc_normal_
import torch.fft
from monai.utils import ensure_tuple_rep

from gkai.model.layers import create_head, PatchEmbed


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ComplexReLU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        real = functor.relu(x.real, inplace=self.inplace)
        imag = functor.relu(x.imag, inplace=self.inplace)
        x = torch.stack([real, imag], dim=-1)
        return torch.view_as_complex(x)


def apply_complex(fr, fi, x):
    if torch.is_complex(x):
        return torch.complex(fr(x.real) - fi(x.imag), fr(x.imag) + fi(x.real))
    else:
        return torch.complex(fr(x), fi(x))


class ComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None):
        super(ComplexLinear, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.fc_r = nn.Linear(in_features, out_features, bias=False, **factory_kwargs)
        self.fc_i = nn.Linear(in_features, out_features, bias=False, **factory_kwargs)
        if bias:
            self.bias = nn.Parameter(torch.zeros((out_features, 2), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self._init_weight()

    def _init_weight(self):
        req_grad = self.fc_r.weight.requires_grad
        self.fc_r.weight.requires_grad = False
        self.fc_i.weight.requires_grad = False
        scale = math.sqrt(0.5)
        self.fc_i.weight.mul_(scale)
        self.fc_r.weight.mul_(scale)
        self.fc_r.weight.requires_grad = req_grad
        self.fc_i.weight.requires_grad = req_grad

    def forward(self, x):
        y = self.multiply(x)
        if self.bias is not None:
            y += torch.view_as_complex(self.bias)
        return y

    def multiply(self, x):
        if not torch.is_complex(x):
            return torch.complex(self.fc_r(x), self.fc_i(x))

        a = self.fc_r.weight  # (out, in)
        b = self.fc_i.weight  # (out, in)
        c = x.real  # (*, in)
        d = x.imag  # (*, in)
        # k1 = (a + b) * c
        k1 = functor.linear(c, a + b, None)
        # k2 = a * (d - c)
        k2 = functor.linear(d - c, a, None)
        # k3 = b * (c + d)
        k3 = functor.linear(c + d, b, None)
        return torch.complex(k1 - k3, k1 + k2)


class ComplexSoftshrink(nn.Module):
    def __init__(self, threshold: float = 0.5, by_magnitude=True, eps=1e-8):
        super().__init__()
        assert threshold > 0.0
        self.threshold = threshold
        self.softshrink = nn.Softshrink(threshold)
        self.by_magnitude = by_magnitude
        self.eps = eps

    def forward(self, x):
        if self.by_magnitude:
            y = x.abs()
            return torch.where(
                y > self.threshold,
                x * (1 - self.threshold / (y + self.eps)),
                torch.asarray(0.0 + 0.0j))
        else:
            x = torch.view_as_real(x)
            x = self.softshrink(x)
            return torch.view_as_complex(x)


class AFNOLayer(nn.Module):
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1.0,
                 hidden_size_factor=1.0, shrink_by_magnitude=False, direct_skip=True):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor

        if not direct_skip:
            self.proj_bias = nn.Linear(hidden_size, hidden_size)
        else:
            self.proj_bias = None

        hidden = int(self.hidden_size_factor * self.block_size)
        self.mlp = nn.Sequential(
            ComplexLinear(self.block_size, hidden),
            ComplexReLU(),
            ComplexLinear(hidden, self.block_size)
        )

        self.softshrink = ComplexSoftshrink(self.sparsity_threshold, by_magnitude=shrink_by_magnitude)

    # def forward_channel_last(self, x):
    def forward(self, x):
        # input_shape = x.shape  # ==> (N, L, C) or (N, H, W, C) or (N, H, W, D, C)
        ndims = x.ndim - 2
        fft_dims = tuple(range(1, 1 + ndims))

        if self.proj_bias is not None:
            bias = self.proj_bias(x)
        else:
            bias = x
        b, *spatial_dims, d = x.shape
        x = torch.fft.rfftn(x, s=spatial_dims, dim=fft_dims, norm="ortho")
        _, *fft_shape, d = x.shape
        x = x.reshape(b, *fft_shape, self.num_blocks, self.block_size)
        x = self.mlp(x)
        x = x.reshape(b, *fft_shape, d)
        x = self.softshrink(x)
        x = torch.fft.irfftn(x, s=spatial_dims, dim=fft_dims, norm="ortho")
        x = x + bias
        return x


class AFNOBlock(nn.Module):
    def __init__(
            self,
            dim,
            mlp_ratio=4.,
            num_blocks=8,
            drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
            double_skip=True,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0,
            hidden_size_factor=1.0,
            shrink_by_magnitude=False,
            direct_skip: bool = True,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = AFNOLayer(dim, num_blocks, sparsity_threshold, hard_thresholding_fraction,
                                hidden_size_factor, shrink_by_magnitude, direct_skip)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.double_skip = double_skip

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        return x


class AFNONet(nn.Module):
    def __init__(
            self,
            spatial_size=(720, 1440),
            patch_size=(16, 16),
            in_channels=1,
            embed_dim=768,
            add_position_embed=True,
            num_blocks=8,
            depth=12,
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0,
            shrink_by_magnitude=False,
            checkpoints: Union[int, None] = None,
            head: str = None,
            head_kwargs: dict = None,
    ):
        super().__init__()
        check_input_shapes(spatial_size, patch_size)
        self.ndims = len(spatial_size)
        self.spatial_size = spatial_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_features = self.embed_dim = embed_dim
        self.num_blocks = ensure_tuple_rep(num_blocks, depth)
        self.use_checkpoint = checkpoints is not None
        self.checkpoint_segments = checkpoints
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(spatial_size=spatial_size, patch_size=self.patch_size,
                                      in_channels=self.in_channels, embed_dim=embed_dim)

        if add_position_embed:
            num_patches = self.patch_embed.num_patches
            # self.pos_embed = nn.Parameter(torch.empty(1, embed_dim, *num_patches))
            self.pos_embed = nn.Parameter(torch.empty(1, *num_patches, embed_dim))
            trunc_normal_(self.pos_embed, std=.02)
        else:
            self.register_parameter('pos_embed', None)
        self.embed_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            AFNOBlock(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                      num_blocks=self.num_blocks[i], sparsity_threshold=sparsity_threshold,
                      hard_thresholding_fraction=hard_thresholding_fraction, shrink_by_magnitude=shrink_by_magnitude)
            for i in range(depth)])

        self.head = create_head(head, self.ndims, embed_dim, head_kwargs)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            if not torch.is_complex(m.weight):
                trunc_normal_(m.weight, std=.02)
            if m.bias is not None and not torch.is_complex(m.bias):
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x):
        # input shape: (B, Ci, H, W, ...)
        x = self.patch_embed(x)   # (B, H/p[0], W/p[1], ..., Co)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.embed_drop(x)

        if not self.use_checkpoint:
            x = self.blocks(x)  # (B, H/p[0], W/p[1], ..., Co)
        else:
            x = checkpoint_sequential(self.blocks, self.checkpoint_segments, x)

        if self.head is not None:
            x = self.head(x)
        return x


def check_input_shapes(spatial_size, patch_size):
    if len(spatial_size) not in (1, 2, 3):
        raise ValueError("only 1D/2D/3D data are supported. ")
    if len(spatial_size) != len(patch_size):
        raise ValueError("spatial dimension differs from patch dimension. ")
    if not np.array([s % p == 0 for s, p in zip(spatial_size, patch_size)], dtype=np.bool_).all():
        raise ValueError("spatial size is not divisible by patch size. ")


def demo():
    batch_size = 10
    dim = 64
    a = torch.randn((batch_size, dim, 96, 96))
    ffta = torch.fft.rfftn(a, s=(96, 96), dim=(2, 3))
    print(ffta.shape)
    a_hat = torch.fft.irfftn(ffta, s=(96, 96), dim=(2, 3))
    torch.testing.assert_close(a_hat, a)

    layer = AFNOLayer(hidden_size=dim)
    a = torch.randn((batch_size, 96, 96, dim))
    b = layer(a)
    print(b.shape)

    layer = PatchEmbed(spatial_size=(128, 128), patch_size=(4, 4), in_channels=3, embed_dim=64)
    a = torch.randn(batch_size, 3, 128, 128)
    b = layer(a)
    print(b.shape)
    assert b.shape == (batch_size, 128//4, 128//4, 64)

    net = AFNONet(spatial_size=(128, 128, 128), patch_size=(2, 2, 2), in_channels=1, embed_dim=256, depth=4,
                  head='PredictionHead', head_kwargs={'out_channels': 1, 'patch_size': 2, 'num_patches': 64})
    a = torch.randn(1, 1, 128, 128, 128)
    b = net(a)
    print(b.shape)
    assert b.shape == (1, 1, 128, 128, 128)


def demo1():
    layer = ComplexLinear(in_features=64, out_features=32)
    x = torch.randn(1, 64, 2)
    x = torch.view_as_complex(x)
    y1 = layer(x)
    y2 = apply_complex(layer.fc_r, layer.fc_i, x)
    assert torch.allclose(y1, y2)


if __name__ == "__main__":
    # model = AFNONet(img_size=(720, 1440), patch_size=(4, 4), in_chans=3, out_chans=10)
    # sample = torch.randn(1, 3, 720, 1440)
    # result = model(sample)
    # print(result.shape)
    # print(torch.norm(result))
    # demo()
    demo1()
