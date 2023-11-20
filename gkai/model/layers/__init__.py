from .basic import Permute, MeanPool, MaxPool, PatchEmbed
from gkai.utils import import_class


def create_head(name: str, ndim: int, in_channels: int, kwargs: dict):
    if name is not None:
        head = import_class(name, 'gkai.model.layers.head')(ndim, in_channels, **kwargs)
    else:
        head = None
    return head
