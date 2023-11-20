import torch
from torch import nn


class ImageNetEnsembleClassifier(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.flatten = nn.Flatten(start_dim=2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        with torch.no_grad():
            x = self.model(x)
            x = self.flatten(x)
            x = self.log_softmax(x)
            x = torch.sum(x, dim=2)
            return torch.argmax(x, dim=1)


class ImageNetEnsembleCountClassifier(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.flatten = nn.Flatten(start_dim=2)

    def forward(self, x):
        with torch.no_grad():
            x = self.model(x)  # (B, C, *S)
            x = self.flatten(x)  # (B, C, S)
            x = torch.argmax(x, dim=1)  # (B, S)
            x, _ = torch.mode(x, dim=-1)
            return x


def _demo():
    x = torch.randn((10, 40, 8, 8))
    model = ImageNetEnsembleClassifier(nn.Identity())
    y = model(x)
    print(y.shape)
    assert y.shape == (10,)


if __name__ == '__main__':
    _demo()
