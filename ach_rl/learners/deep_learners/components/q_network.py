import copy
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn
from ach_rl import constants
from ach_rl.utils import custom_torch_layers


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

            if layer_index in self._shared_layers:
                layer, nonlinearity = self._construct_layer(
                    layer_specification=layer_specification
                )
                self._core_layers.append(layer)
                self._core_layers.append(nonlinearity)
            else:
                if self._copy_initialisation:
                    layer, nonlinearity = self._construct_layer(
                        layer_specification=layer_specification
                    )
                    for i in range(self._num_branches):
                        self._branched_layers[i].append(copy.deepcopy(layer))
                        self._branched_layers[i].append(copy.deepcopy(nonlinearity))
                else:
                    for i in range(self._num_branches):
                        layer, nonlinearity = self._construct_layer(
                            layer_specification=layer_specification
                        )
                        self._branched_layers[i].append(layer)
                        self._branched_layers[i].append(nonlinearity)

    def _construct_layer(
        self, layer_specification: Dict
    ) -> Tuple[nn.Module, nn.Module]:
        layer_type = list(layer_specification.keys())[0]
        layer_info = list(layer_specification.values())[0]
        layer_nonlinearity = layer_info.get(constants.NONLINEARITY)

        if layer_type == constants.CONV:
            layer = nn.Conv2d(
                in_channels=layer_info[constants.IN_CHANNELS],
                out_channels=layer_info[constants.NUM_FILTERS],
                kernel_size=layer_info[constants.KERNEL_SIZE],
                stride=layer_info.get(constants.STRIDE, 1),
                padding=layer_info.get(constants.PADDING, 0),
            )
        elif layer_type == constants.FC:
            layer = nn.Linear(
                in_features=layer_info.get(constants.IN_FEATURES),
                out_features=layer_info.get(constants.OUT_FEATURES, self._num_actions),
            )

        elif layer_type == constants.FLATTEN:
            layer = custom_torch_layers.Flatten()
        else:
            raise ValueError(f"Layer type {layer_type} not recognised")

        if layer_nonlinearity == constants.RELU:
            nonlinearity = nn.ReLU()
        elif layer_nonlinearity == constants.IDENTITY:
            nonlinearity = nn.Identity()
        else:
            raise ValueError(f"Non-linearity {layer_nonlinearity} not recognised")

        # initialise weights (only those with parameters)
        if [p for p in layer.parameters()]:
            self._initialise_weights(
                layer.weight,
                initialisation=layer_info.get(constants.WEIGHT_INITIALISATION),
            )
            self._initialise_weights(
                layer.bias,
                initialisation=layer_info.get(constants.BIAS_INITIALISATION),
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
        elif initialisation == constants.ZEROS:
            nn.init.zeros_(layer_weights)
        elif initialisation == constants.NORMAL:
            nn.init.normal_(layer_weights)
        elif initialisation == constants.XAVIER_NORMAL:
            nn.init.xavier_normal_(layer_weights)
        elif initialisation == constants.XAVIER_UNIFORM:
            nn.init.xavier_uniform_(layer_weights)

    def forward_all_heads(self, x: torch.Tensor):
        """Method to get output through every head for an input tensor, x."""
        for layer in self._core_layers:
            x = layer(x)

        if self._num_branches > 0:
            all_outputs = []

            for branch in range(self._num_branches):
                branch_output = x
                for layer in self._branched_layers[branch]:
                    branch_output = layer(branch_output)
                all_outputs.append(branch_output)

            return torch.stack(all_outputs)
        else:
            return x.unsqueeze(0)

    def forward(self, x: torch.Tensor, branch: Union[int, None] = None):
        """Forward pass through network."""

        for layer in self._core_layers:
            x = layer(x)

        if branch is not None:
            for layer in self._branched_layers[branch]:
                x = layer(x)

        return x
