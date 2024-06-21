import warnings
from typing import Any, List, Optional

import torch
import torch.nn as nn

from ptmelt.layers import MELTBatchNorm
from ptmelt.utils import get_activation, get_initializer


class MELTBlock(nn.Module):
    def __init__(
        self,
        input_features: int,
        node_list: List[int],
        activation: Optional[str] = "relu",
        dropout: Optional[float] = 0.0,
        batch_norm: Optional[bool] = False,
        batch_norm_type: Optional[str] = "ema",
        use_batch_renorm: Optional[bool] = False,
        initializer: Optional[str] = "glorot_uniform",
        **kwargs: Any,
    ):
        super(MELTBlock, self).__init__(**kwargs)

        self.input_features = input_features
        self.node_list = node_list
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.batch_norm_type = batch_norm_type
        self.use_batch_renorm = use_batch_renorm
        self.initializer = initializer

        # Get the initializer function
        self.initializer_fn = get_initializer(self.initializer)

        # Create layer dictionary
        self.layer_dict = nn.ModuleDict()

        # Number of layers in the block
        self.num_layers = len(self.node_list)

        # Validate dropout value
        if self.dropout is not None:
            assert 0.0 <= self.dropout < 1.0, "Dropout must be in the range [0, 1)."

        # Get the activation layers
        if self.activation:
            self.layer_dict.update(
                {
                    f"activation_{i}": get_activation(self.activation)
                    for i in range(self.num_layers)
                }
            )

        # Optional dropout layers
        if self.dropout > 0:
            self.layer_dict.update(
                {
                    f"dropout_{i}": nn.Dropout(p=self.dropout)
                    for i in range(self.num_layers)
                }
            )

        # Optional batch normalization layers
        if self.batch_norm:
            if self.batch_norm_type == "pytorch":
                self.layer_dict.update(
                    {
                        f"batch_norm_{i}": nn.BatchNorm1d(
                            num_features=self.node_list[i],
                            affine=True,
                            track_running_stats=True,
                            momentum=1e-2,
                            eps=1e-3,
                        )
                        for i in range(self.num_layers)
                    }
                )
            else:
                self.layer_dict.update(
                    {
                        f"batch_norm_{i}": MELTBatchNorm(
                            num_features=self.node_list[i],
                            affine=True,
                            track_running_stats=True,
                            average_type=self.batch_norm_type,
                            momentum=1e-2,
                            eps=1e-3,
                        )
                        for i in range(self.num_layers)
                    }
                )


class DenseBlock(MELTBlock):
    def __init__(
        self,
        **kwargs: Any,
    ):
        super(DenseBlock, self).__init__(**kwargs)

        # Initialize dense layers
        self.layer_dict.update(
            {
                f"dense_{i}": nn.Linear(
                    in_features=(
                        self.input_features if i == 0 else self.node_list[i - 1]
                    ),
                    out_features=self.node_list[i],
                )
                for i in range(self.num_layers)
            }
        )
        # Initialize the weights
        [
            self.initializer_fn(self.layer_dict[f"dense_{i}"].weight)
            for i in range(self.num_layers)
        ]

    def forward(self, inputs: torch.Tensor):
        """Perform the forward pass of the dense block."""
        x = inputs

        for i in range(self.num_layers):
            # dense -> batch norm -> activation -> dropout
            x = self.layer_dict[f"dense_{i}"](x)
            x = self.layer_dict[f"batch_norm_{i}"](x) if self.batch_norm else x
            x = self.layer_dict[f"activation_{i}"](x) if self.activation else x
            x = self.layer_dict[f"dropout_{i}"](x) if self.dropout > 0 else x

        return x


class ResidualBlock(MELTBlock):
    def __init__(
        self,
        layers_per_block: Optional[int] = 2,
        pre_activation: Optional[bool] = False,
        post_add_activation: Optional[bool] = False,
        **kwargs: Any,
    ):
        super(ResidualBlock, self).__init__(**kwargs)

        self.layers_per_block = layers_per_block
        self.pre_activation = pre_activation
        self.post_add_activation = post_add_activation

        # Warning if the number of layers is not divisible by layers_per_block
        if self.num_layers % self.layers_per_block != 0:
            warnings.warn(
                f"Warning: Number of layers {self.num_layers} is not divisible by "
                f"layers_per_block ({self.layers_per_block}), so the last block will "
                f"have {self.num_layers % self.layers_per_block} layers."
            )

        # Initialize dense layers
        self.layer_dict.update(
            {
                f"dense_{i}": nn.Linear(
                    in_features=(
                        self.input_features if i == 0 else self.node_list[i - 1]
                    ),
                    out_features=self.node_list[i],
                )
                for i in range(self.num_layers)
            }
        )
        # Initialize the weights
        [
            self.initializer_fn(self.layer_dict[f"dense_{i}"].weight)
            for i in range(self.num_layers)
        ]

        # Optional activation layer after addition
        if self.post_add_activation:
            self.layer_dict.update(
                {
                    f"post_add_act_{i}": get_activation(self.activation)
                    for i in range(self.num_layers // 2)
                }
            )

    def forward(self, inputs: torch.Tensor):
        """Perform the forward pass of the residual block."""
        x = inputs

        for i in range(self.num_layers):
            y = x

            # dense -> (pre-activation) -> batch norm -> dropout -> (post-activation)
            x = self.layer_dict[f"dense_{i}"](x)
            x = self.layer_dict[f"activation_{i}"](x) if self.pre_activation else x
            x = self.layer_dict[f"batch_norm_{i}"](x) if self.batch_norm else x
            x = self.layer_dict[f"dropout_{i}"](x) if self.dropout > 0 else x
            x = self.layer_dict[f"activation_{i}"](x) if not self.pre_activation else x

            # Add the residual connection when reaching the end of a residual block
            if (i + 1) % self.layers_per_block == 0 or i == self.num_layers - 1:
                x = x + y
                x = (
                    self.layer_dict[f"post_add_act_{i // self.layers_per_block}"](x)
                    if self.post_add_activation
                    else x
                )

        return x


class DefaultOutput(nn.Module):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        activation: Optional[str] = "linear",
        initializer: Optional[str] = "glorot_uniform",
        **kwargs: Any,
    ):
        super(DefaultOutput, self).__init__(**kwargs)

        self.input_features = input_features
        self.output_features = output_features
        self.activation = activation
        self.initializer = initializer

        # Get the initializer function
        self.initializer_fn = get_initializer(self.initializer)

        # Initialize output layer
        self.output_layer = nn.Linear(
            in_features=self.input_features, out_features=self.output_features
        )
        # Initialize the weights
        self.initializer_fn(self.output_layer.weight)

        # Initialize activation layer
        self.activation_layer = get_activation(self.activation)

    def forward(self, inputs: torch.Tensor):
        """Perform the forward pass of the default output layer."""
        x = self.output_layer(inputs)
        x = self.activation_layer(x)

        return x
