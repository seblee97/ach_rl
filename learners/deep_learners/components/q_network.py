from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import constants
import torch
import torch.nn as nn
from utils import custom_torch_layers


class QNetwork(nn.Module):
    """Q-value network."""

    def __init__(
        self,
        state_dim: Tuple[int, int, int],
        num_actions: int,
        layer_specifications: List[Dict[str, Any]],
        shared_layers: Union[List[int], None] = None,
        num_branches: int = 0,
        copy_initialisation: bool = False,
    ):
        """Class constructor.

        Args:
            state_dim: dimensions of the state.
            num_actions: number of possible actions.
            layer_specifications: hidden layer specifications.
        """
        self._state_dim = state_dim
        self._num_actions = num_actions
        self._layer_specifications = layer_specifications
        self._shared_layers = shared_layers or list(
            range(len(self._layer_specifications))
        )
        self._num_branches = num_branches
        self._copy_initialisation = copy_initialisation

        super().__init__()

        self._construct_layers()

    def _construct_layers(self):
        """Method to setup network architecture."""
        self._core_layers = nn.ModuleList([])
        self._branched_layers = nn.ModuleList(
            [nn.ModuleList([]) for _ in range(self._num_branches)]
        )

        for layer_index, layer_specification in enumerate(self._layer_specifications):

            layer, nonlinearity = self._construct_layer(
                layer_specification=layer_specification
            )

            self._core_layers.append(layer)
            self._core_layers.append(nonlinearity)

    def _construct_layer(
        self, layer_specification: Dict
    ) -> Tuple[nn.Module, nn.Module]:
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

        # initialise weights (only those with parameters)
        if [p for p in layer.parameters()]:
            self._initialise_weights(
                layer.weight,
                initialisation=layer_info.get(
                    constants.Constants.WEIGHT_INITIALISATION
                ),
            )
            self._initialise_weights(
                layer.bias,
                initialisation=layer_info.get(constants.Constants.BIAS_INITIALISATION),
            )

        return layer, nonlinearity

    def _initialise_weights(
        self, layer_weights: nn.Parameter, initialisation: Union[str, None]
    ) -> None:
        """Initialise weights of network layer according to specification.

        Initialisation is in-place.

        Args:
            layer_weights: un-initialised weights.
        """
        if initialisation is None:
            pass
        elif initialisation == constants.Constants.ZEROS:
            nn.init.zeros_(layer_weights)
        elif initialisation == constants.Constants.NORMAL:
            nn.init.normal_(layer_weights)
        elif initialisation == constants.Constants.XAVIER_NORMAL:
            nn.init.xavier_normal_(layer_weights)
        elif initialisation == constants.Constants.XAVIER_UNIFORM:
            nn.init.xavier_uniform_(layer_weights)

    def forward(self, x: torch.Tensor):
        """Forward pass through network."""
        for layer in self._core_layers:
            x = layer(x)

        return x
