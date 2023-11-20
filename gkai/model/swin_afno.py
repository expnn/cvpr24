# -*- coding: utf-8 -*-

import torch
import itertools
from torch import nn
from torch.utils.checkpoint import checkpoint
from typing import Sequence, Union, Callable
from itertools import zip_longest
from monai.networks.blocks import Convolution
from monai.utils import ensure_tuple_rep
from timm.models.layers import trunc_normal_

from gkai.model.layers import Permute, PatchEmbed, create_head
from gkai.model.afno import AFNOBlock


class SwinAFNO(nn.Module):
    def __init__(
            self,
            spatial_size: Sequence[int],
            in_channels: int,
            embed_dim: int,
            window_size: Union[int, Sequence[int]],
            patch_size: Union[int, Sequence[int]],
            depths: Sequence[int],
            num_heads: Union[int, Sequence[int]],
            add_position_embed: bool = False,
            mlp_ratio: float = 4.0,
            drop_rate: float = 0.0,
            drop_path_rate: float = 0.0,
            sparsity_threshold: float = 0.01,
            hard_thresholding_fraction: float = 1.0,
            shrink_by_magnitude: bool = False,
            direct_skip: bool = True,
            hidden_size_factor: float = 1.0,
            norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
            lift_feature_sizes: Sequence[int] = None,
            lift_kernel_sizes: Union[Sequence[int], int] = 1,
            lift_strides: Union[Sequence[int], int] = None,
            lift_activation: Union[tuple, str, None] = "PRELU",
            lift_norm_layer: Union[str, tuple, None] = "INSTANCE",
            lift_dropout: Union[float, None] = None,
            use_checkpoint: bool = False,
            head: str = None,
            head_kwargs: dict = None,
    ) -> None:
        super().__init__()
        ndim = len(spatial_size)
        self.num_stages = len(depths)
        self.num_heads = ensure_tuple_rep(num_heads, self.num_stages)
        self.embed_dim = embed_dim
        self.window_size = ensure_tuple_rep(window_size, ndim)
        self.patch_size = ensure_tuple_rep(patch_size, ndim)
        self.shift_size = tuple(map(lambda x: x // 2, self.window_size))
        self.no_shift_size = ensure_tuple_rep(0, ndim)
        self.lift_feature_sizes = ensure_tuple_rep(lift_feature_sizes, self.num_stages - 1)
        self.lift_kernel_sizes = ensure_tuple_rep(lift_kernel_sizes, self.num_stages - 1)
        lift_strides = lift_strides or self.lift_kernel_sizes
        self.lift_strides = ensure_tuple_rep(lift_strides, self.num_stages - 1)
        self.use_checkpoint = use_checkpoint
        assert all(map(lambda x: x > 0, self.shift_size))

        self.patch_embed = PatchEmbed(
            spatial_size=spatial_size,
            patch_size=self.patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        if add_position_embed:
            num_patches = self.patch_embed.num_patches
            self.pos_embed = nn.Parameter(torch.empty(1, *num_patches, embed_dim))
            trunc_normal_(self.pos_embed, std=1/embed_dim)
        else:
            self.register_parameter('pos_embed', None)

        self.embed_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()
        shift = False
        for packed in zip_longest(range(self.num_stages), self.lift_feature_sizes,
                                  self.lift_kernel_sizes, self.lift_strides):
            stage, lift_size, lift_kernel_size, lift_stride = packed
            stage_dpr = dpr[sum(depths[:stage]):sum(depths[:stage + 1])]
            assert len(stage_dpr) == depths[stage]

            blocks = nn.Sequential()
            for i in range(depths[stage]):
                blocks.append(
                    SwinAFNOBlock(
                        window_size=self.window_size,
                        shift_size=self.shift_size if shift else self.no_shift_size,
                        dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=stage_dpr[i],
                        norm_layer=norm_layer,
                        num_blocks=self.num_heads[stage], sparsity_threshold=sparsity_threshold,
                        hard_thresholding_fraction=hard_thresholding_fraction, hidden_size_factor=hidden_size_factor,
                        shrink_by_magnitude=shrink_by_magnitude, direct_skip=direct_skip)
                )
                shift = not shift

            if lift_size is not None:
                # noinspection PyTypeChecker
                blocks.append(nn.Sequential(*[
                    Permute(dims=(0, ndim + 1, *range(1, 1 + ndim)), make_contiguous=True),
                    Convolution(
                        spatial_dims=ndim,
                        in_channels=embed_dim,
                        out_channels=lift_size,
                        kernel_size=lift_kernel_size,
                        strides=lift_stride,
                        act=lift_activation,
                        norm=lift_norm_layer,
                        dropout=lift_dropout,
                        padding='valid',
                    ),
                    Permute(dims=(0, *range(2, 2 + ndim), 1), make_contiguous=True),
                ]))
                embed_dim = lift_size
            self.stages.append(blocks)

        self.head = create_head(head, ndim, embed_dim, head_kwargs)

    # def forward(self, x, normalize=True):
    def forward(self, x):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.embed_drop(x)
        for stage in self.stages:
            if self.use_checkpoint:
                x = checkpoint(stage, x)
            else:
                x = stage(x)
        if self.head is not None:
            x = self.head(x)
        return x


class SwinAFNOBlock(nn.Module):
    """
    将输入的数据, 通过 shift window 划分为多个块, 然后在每个块上执行 AFNOBlock 运算.
    """

    def __init__(
            self,
            window_size: Sequence[int],
            shift_size: Sequence[int],
            dim: int,
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

        self.window_size = window_size
        self.shift_size = shift_size
        self.afno_block = AFNOBlock(
            dim=dim, mlp_ratio=mlp_ratio, num_blocks=num_blocks, drop=drop, drop_path=drop_path, act_layer=act_layer,
            norm_layer=norm_layer, double_skip=double_skip, sparsity_threshold=sparsity_threshold,
            hard_thresholding_fraction=hard_thresholding_fraction, hidden_size_factor=hidden_size_factor,
            shrink_by_magnitude=shrink_by_magnitude, direct_skip=direct_skip
        )

    def forward(self, x):
        x_shape = x.size()  # (B, *S, C), where S is spatial dimensions
        assert len(x_shape) in (3, 4, 5), \
            f"Only 1D/2D/3D input data are supported, got input data of shape {x_shape[1:-1]}"

        dest = torch.zeros_like(x)
        for w, recover_shape, index in block_partition(x, self.window_size, self.shift_size):
            y = self.afno_block(w)
            y = window_reverse(y, recover_shape)
            dest[index] = y
        return dest


def block_partition(x, window_sizes, shift_sizes):
    def create_slices(shape_, window_, shift_):
        if shape_ <= window_:  # 如果某一维的长度小于等于 window 大小, 则 shift 毫无意义.
            window_ = shape_
            shift_ = 0

        slices_ = []
        if shift_ > 0:
            slices_.append(slice(0, shift_))
        else:
            assert shift_ == 0

        rest = (shape_ - shift_) % window_
        if rest != 0:
            slices_.extend([slice(shift_, -rest), slice(-rest, None)])
        else:
            slices_.append(slice(shift_, None))
        return slices_

    spatial_shape = x.shape[1:-1]
    slices = [create_slices(shape, win, shift) for shape, win, shift in zip(spatial_shape, window_sizes, shift_sizes)]

    blocks = []
    for index in itertools.product([slice(None)], *slices, [slice(None)]):
        y = x[index]
        if y.numel() > 0:
            blocks.append(window_partition(y, window_sizes) + (index,))

    return blocks


def window_partition(x, window_sizes):
    x_shape = x.size()
    assert len(x_shape) in (5, 4), f"Only 2D/3D windows are supported, got {x_shape[1:-1]}"

    b, *spatial_sizes, c = x_shape
    new_shape = [b]
    for w, s in zip(window_sizes, spatial_sizes):
        if s >= w:
            new_shape.extend([s // w, w])
        else:
            new_shape.extend([1, s])
    new_shape.append(c)
    x = x.view(*new_shape)

    if len(x_shape) == 5:
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        recover_shape = x.shape
        windows = x.flatten(0, 3)  # (#windows, *spatial_dims, channels)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        recover_shape = x.shape
        windows = x.flatten(0, 2)  # (#windows, *spatial_dims, channels)
    return windows, recover_shape


def window_reverse(windows, recover_shape):
    assert windows.ndim in (4, 5), f"Only 2D/3D windows are supported, got {windows.ndim - 2}D data"
    x = windows.view(*recover_shape)
    if windows.ndim == 5:
        x = (
            x.permute(0, 1, 4, 2, 5, 3, 6, 7)
            .contiguous()
            .flatten(5, 6)  # ! 必须逆序写, 避免因为 flatten 导致维数变化
            .flatten(3, 4)
            .flatten(1, 2)
        )  # (B, D, H, W, C)
    else:
        x = (
            x.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .flatten(3, 4)  # ! 必须逆序写, 避免因为 flatten 导致维数变化
            .flatten(1, 2)
        )
    return x


def demo():
    a = torch.rand(1, 32, 40, 16)
    window_sizes = [7, 7]
    shift_sizes = [3, 3]

    afno_layer = AFNOBlock(16, num_blocks=2)
    dest = torch.zeros_like(a)
    torch.fill_(dest, 100.0)
    # noinspection PyTypeChecker
    for w, recover_shape, index in block_partition(a, window_sizes, shift_sizes):
        print(w.shape, recover_shape, index)
        y = afno_layer(w)
        y = window_reverse(y, recover_shape)
        dest[index] = torch.nn.functional.sigmoid(y)
    print(dest[..., 0])
    assert not torch.any(torch.eq(dest, 100.0))  # 确认每一个元素都被写到了, 没有遗漏.
    loss = dest.mean()
    loss.backward()

    model = SwinAFNO(
        spatial_size=(256, 256),
        in_channels=3,
        embed_dim=64,
        window_size=32,
        patch_size=2,
        depths=(2, 2, 2, 2),
        num_heads=(4, 4, 8, 8),
        lift_feature_sizes=(128, 256, 512),
        lift_kernel_sizes=2,
        head="ConvClassificationHeadV1",
        head_kwargs=dict(
            num_classes=100,
            pool_method=None,
            flattened_size=4,
            out_channels=(1024, 1024, 1024),
        )
    )
    x = torch.rand(1, 3, 256, 256)
    y = model(x)
    print(y.shape)

    model = SwinAFNO(
        spatial_size=(256, 256),
        in_channels=3,
        embed_dim=64,
        window_size=32,
        patch_size=2,
        depths=(2, 2, 2, 2, 2, 2),
        num_heads=(4, 4, 4, 8, 8, 8),
        lift_feature_sizes=(None, None, 128, 256, 512),
        lift_kernel_sizes=2,
        head="ConvClassificationHeadV2",
        head_kwargs=dict(
            num_classes=100,
            out_channels=(1024, ),
            kernel_sizes=4,
            ada_pool_size=2,
            ada_pool_method='max'
        )
    )
    x = torch.rand(1, 3, 256, 256)
    y = model(x)
    print(y.shape)


if __name__ == '__main__':
    demo()
