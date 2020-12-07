from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

from utils import custom_torch_layers

import torch
import torch.nn as nn

import constants


class QNetwork(nn.Module):
    def __init__(
        self,
        state_dim: Tuple[int, int, int],
        num_actions: int,
        layer_specifications: List[Dict[str, Any]],
    ):
        self._state_dim = state_dim
        self._num_actions = num_actions
        self._layer_specifications = layer_specifications

        super().__init__()

        self._construct_layers()

    def _construct_layers(self):

        self._layers = nn.ModuleList([])

        for layer_specification in self._layer_specifications:

            layer_type = list(layer_specification.keys())[0]
            layer_info = list(layer_specification.values())[0]
            layer_nonlinearity = layer_info.get(constants.Constants.NONLINEARITY)

            if layer_type == constants.Constants.CONV:
                layer = nn.Conv2d(
                    in_channels=layer_info[constants.Constants.IN_CHANNELS],
                    out_channels=layer_info[constants.Constants.NUM_FILTERS],
                    kernel_size=layer_info[constants.Constants.KERNEL_SIZE],
                    stride=layer_info.get(constants.Constants.STRIDE, 1),
                    padding=layer_info.get(constants.Constants.PADDING, 0),
                )
            elif layer_type == constants.Constants.FC:
                layer = nn.Linear(
                    in_features=layer_info.get(constants.Constants.IN_FEATURES),
                    out_features=layer_info.get(
                        constants.Constants.OUT_FEATURES, self._num_actions
                    ),
                )
            elif layer_type == constants.Constants.FLATTEN:
                layer = custom_torch_layers.Flatten()
            else:
                raise ValueError(f"Layer type {layer_type} not recognised")

            if layer_nonlinearity == constants.Constants.RELU:
                nonlinearity = nn.ReLU()
            elif layer_nonlinearity == constants.Constants.IDENTITY:
                nonlinearity = nn.Identity()
            else:
                raise ValueError(f"Non-linearity {layer_nonlinearity} not recognised")

            self._layers.append(layer)
            self._layers.append(nonlinearity)

    def forward(self, x: torch.Tensor):

        for layer in self._layers:
            x = layer(x)

        return x