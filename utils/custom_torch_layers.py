import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor):
        return x.view(x.shape[0], -1)
