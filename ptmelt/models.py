import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ptmelt.layers import MELTBatchNorm, MELTDropout


class MELTModel(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_outputs: int,
        width: Optional[int] = 32,
        depth: Optional[int] = 2,
        act_fun: Optional[str] = "relu",
        dropout: Optional[float] = 0.0,
        input_dropout: Optional[float] = 0.0,
        batch_norm: Optional[bool] = False,
        batch_norm_type: Optional[str] = "ema",
        use_batch_renorm: Optional[bool] = False,
        output_activation: Optional[str] = None,
        initializer: Optional[str] = "glorot_uniform",
        l1_reg: Optional[float] = 0.0,
        l2_reg: Optional[float] = 0.0,
        **kwargs,
    ):
        """
        PT-MELT Base model.

        Args:
            num_features (int): The number of input features.
            num_outputs (int): The number of output units.
            width (int, optional): The width of the hidden layers.
            depth (int, optional): The number of hidden layers.
            act_fun (str, optional): The activation function to use.
            dropout (float, optional): The dropout rate.
            input_dropout (float, optional): The input dropout rate.
            batch_norm (bool, optional): Whether to use batch normalization.
            batch_norm_type (str, optional): The type of batch normalization to use.
            use_batch_renorm (bool, optional): Whether to use batch renormalization.
            output_activation (str, optional): The activation function for the output layer.
            initializer (str, optional): The weight initializer to use.
            l1_reg (float, optional): The L1 regularization strength.
            l2_reg (float, optional): The L2 regularization strength.
            **kwargs: Additional keyword arguments.

        """
        super(MELTModel, self).__init__(**kwargs)

        self.num_features = num_features
        self.num_outputs = num_outputs
        self.width = width
        self.depth = depth
        self.act_fun = act_fun
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.batch_norm = batch_norm
        self.batch_norm_type = batch_norm_type
        self.use_batch_renorm = use_batch_renorm
        self.output_activation = output_activation
        self.initializer = initializer
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

        # Initialize flags for layers
        self.has_batch_norm = False
        self.has_dropout = False
        self.has_input_dropout = False
        self.has_output_activation = False

        # Create layer dictionary
        self.layer_dict = nn.ModuleDict()

    def build(self):
        """Build the model."""
        self.initialize_layers()

    def initialize_layers(self):
        """Initialize the layers of the model."""
        self.create_initializer()
        self.create_regularizer()
        self.create_dropout_layers()
        self.create_batch_norm_layers()
        self.create_input_layer()
        self.create_output_layer()

        # Set attribute flags based on which layers are present
        self.has_batch_norm = hasattr(self.layer_dict, "batch_norm_0")
        self.has_dropout = hasattr(self.layer_dict, "dropout_0")
        self.has_input_dropout = hasattr(self.layer_dict, "input_dropout")
        self.has_output_activation = hasattr(self.layer_dict, "output_act")

    def create_regularizer(self):
        """Create the regularization function."""
        pass

    def create_initializer(self):
        """Create the initializer function."""
        if self.initializer == "glorot_uniform":
            self.initializer_function = nn.init.xavier_uniform_
        elif self.initializer == "glorot_normal":
            self.initializer_function = nn.init.xavier_normal_
        elif self.initializer == "he_uniform":
            self.initializer_function = nn.init.kaiming_uniform_
        elif self.initializer == "he_normal":
            self.initializer_function = nn.init.kaiming_normal_
        else:
            raise ValueError(f"Unsupported initializer {self.initializer}")

    def create_dropout_layers(self):
        """Create the dropout layers."""
        # Dropout for Hidden Layers
        if self.dropout > 0:
            self.layer_dict.update(
                {f"dropout_{i}": nn.Dropout(p=self.dropout) for i in range(self.depth)}
            )

        # Input Dropout Layer
        if self.input_dropout > 0:
            self.layer_dict.update({"input_dropout": nn.Dropout(p=self.input_dropout)})

    def create_batch_norm_layers(self):
        """Create the batch normalization layers based on the type."""
        # TODO: Extract batch norm parameters to class attributes

        # Batch Normalization
        if self.batch_norm:
            if self.batch_norm_type == "pytorch":
                self.layer_dict.update(
                    {
                        f"batch_norm_{i}": nn.BatchNorm1d(
                            num_features=self.width,
                            affine=True,
                            track_running_stats=True,
                            momentum=1e-2,
                            eps=1e-3,
                        )
                        for i in range(self.depth + 1)
                    }
                )
            else:
                self.layer_dict.update(
                    {
                        f"batch_norm_{i}": MELTBatchNorm(
                            num_features=self.width,
                            affine=True,
                            track_running_stats=True,
                            average_type=self.batch_norm_type,
                            momentum=1e-2,
                            eps=1e-3,
                        )
                        for i in range(self.depth + 1)
                    }
                )

    def create_input_layer(self):
        """Create the input layer and associated activation."""
        self.layer_dict.update({"input2bulk": nn.Linear(self.num_features, self.width)})
        # initialize weights
        self.initializer_function(self.layer_dict["input2bulk"].weight)

        # Input Activation
        self.layer_dict.update({"input2bulk_act": self.get_activation(self.act_fun)})

    def create_output_layer(self):
        """Create the output layer and associated activation."""
        # Output Layer
        self.layer_dict.update({"output": nn.Linear(self.width, self.num_outputs)})
        # initialize weights
        self.initializer_function(self.layer_dict["output"].weight)

        # Output Activation
        if self.output_activation:
            self.layer_dict.update(
                {"output_act": self.get_activation(self.output_activation)}
            )

    def compute_jacobian(self, x):
        """Compute the Jacobian of the model with respect to the input."""
        pass

    def get_activation(self, act_name):
        """Utility method to get activation based on its name."""
        if act_name == "relu":
            return nn.ReLU()
        elif act_name == "sigmoid":
            return nn.Sigmoid()
        elif act_name == "tanh":
            return nn.Tanh()
        elif act_name == "linear":
            return nn.Identity()
        else:
            raise ValueError(f"Unsupported activation function {act_name}")

    def l1_regularization(self, lambda_l1):
        """Compute the L1 regularization term for use in the loss function."""
        l1_norm = sum(
            p.abs().sum()
            for name, p in self.named_parameters()
            if p.requires_grad and "weight" in name
        )
        return lambda_l1 * l1_norm

    def l2_regularization(self, lambda_l2):
        """Compute the L2 regularization term for use in the loss function."""
        l2_norm = sum(
            p.pow(2.0).sum()
            for name, p in self.named_parameters()
            if p.requires_grad and "weight" in name
        )
        return 0.5 * lambda_l2 * l2_norm

    def fit(self, train_dl, val_dl, optimizer, criterion, num_epochs):
        """Perform the model training loop."""

        # Create history dictionary
        if not hasattr(self, "history"):
            self.history = {"loss": [], "val_loss": []}

        for epoch in tqdm(range(num_epochs)):
            # Put model in training mode
            self.train()
            running_loss = 0.0

            for x_in, y_in in train_dl:

                # Forward pass
                pred = self(x_in)
                loss = criterion(pred, y_in)

                # Add L1 and L2 regularization if present
                if self.l1_reg > 0:
                    loss += self.l1_regularization(lambda_l1=self.l1_reg)
                if self.l2_reg > 0:
                    loss += self.l2_regularization(lambda_l2=self.l2_reg)

                # Zero the parameter gradients
                optimizer.zero_grad()
                # Backward pass
                loss.backward()
                # Optimize
                optimizer.step()
                # Print statistics
                running_loss += loss.item()

            # Normalize loss
            running_loss /= len(train_dl)

            # Put model in evaluation mode
            self.eval()
            # Compute validation loss
            running_val_loss = 0.0
            with torch.no_grad():
                for x_val, y_val in val_dl:
                    pred_val = self(x_val)
                    val_loss = criterion(pred_val, y_val)
                    running_val_loss += val_loss.item()

            running_val_loss /= len(val_dl)

            # Print statistics
            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}, Loss: {running_loss:.4f}, "
                    f"Val Loss: {running_val_loss:.4f}"
                )

            # Save history
            self.history["loss"].append(running_loss)
            self.history["val_loss"].append(running_val_loss)


class ArtificialNeuralNetwork(MELTModel):
    def __init__(
        self,
        **kwargs,
    ):
        """
        Artificial Neural Network (ANN) model.

        Args:
            **kwargs: Additional keyword arguments.

        """
        super(ArtificialNeuralNetwork, self).__init__(**kwargs)

    def initialize_layers(self):
        """Initialize the layers of the ANN."""
        super(ArtificialNeuralNetwork, self).initialize_layers()

        # Bulk layers
        self.layer_dict.update(
            {f"bulk_{i}": nn.Linear(self.width, self.width) for i in range(self.depth)}
        )
        # initialize weights
        for i in range(self.depth):
            self.initializer_function(self.layer_dict[f"bulk_{i}"].weight)

        self.layer_dict.update(
            {
                f"bulk_act_{i}": self.get_activation(self.act_fun)
                for i in range(self.depth)
            }
        )

    def forward(self, inputs):
        """Perform the forward pass of the ANN."""
        x = self.layer_dict["input2bulk"](inputs)
        x = self.layer_dict["batch_norm_0"](x) if self.has_batch_norm else x
        x = self.layer_dict["input2bulk_act"](x)
        x = self.layer_dict["input_dropout"](x) if self.has_input_dropout else x

        for i in range(self.depth):
            x = self.layer_dict[f"bulk_{i}"](x)
            x = self.layer_dict[f"batch_norm_{i + 1}"](x) if self.has_batch_norm else x
            x = self.layer_dict[f"bulk_act_{i}"](x)
            x = self.layer_dict[f"dropout_{i}"](x) if self.has_dropout else x

        output = self.layer_dict["output"](x)
        output = (
            self.layer_dict["output_act"](output)
            if self.has_output_activation
            else output
        )

        return output


class ResidualNeuralNetwork(MELTModel):
    def __init__(
        self,
        layers_per_block: Optional[int] = 2,
        pre_activation: Optional[bool] = True,
        post_add_activation: Optional[bool] = False,
        **kwargs,
    ):
        """
        Residual Neural Network (ResNet) model.

        Args:
            layers_per_block (int, optional): The number of layers per residual block.
            pre_activation (bool, optional): Whether to use pre-activation.
            post_add_activation (bool, optional): Whether to use activation after
                                                  addition.
            **kwargs: Additional keyword arguments.

        """
        super(ResidualNeuralNetwork, self).__init__(**kwargs)

        self.layers_per_block = layers_per_block
        self.pre_activation = pre_activation
        self.post_add_activation = post_add_activation

    def build(self):
        """Build the model."""
        if self.depth % self.layers_per_block != 0:
            warnings.warn(
                f"Warning: depth {self.depth} is not divisible by layers_per_block "
                f"({self.layers_per_block}), so the last block will have "
                f"{self.depth % self.layers_per_block} layers."
            )

        self.initialize_layers()
        super(ResidualNeuralNetwork, self).build()

    def initialize_layers(self):
        """Initialize the layers of the ResNet."""
        super(ResidualNeuralNetwork, self).initialize_layers()

        # ResNet Bulk layers
        self.layer_dict.update(
            {
                f"resnet_bulk_{i}": nn.Linear(self.width, self.width)
                for i in range(self.depth)
            }
        )
        # initialize weights
        for i in range(self.depth):
            self.initializer_function(self.layer_dict[f"resnet_bulk_{i}"].weight)

        # ResNet Activation layers
        self.layer_dict.update(
            {
                f"resnet_bulk_act_{i}": self.get_activation(self.act_fun)
                for i in range(self.depth)
            }
        )

        # Optional activation layer after addition
        if self.post_add_activation:
            self.layer_dict.update(
                {
                    f"post_add_act{i}": self.get_activation(self.act_fun)
                    for i in range(self.depth // 2)
                }
            )

    def forward(self, inputs):
        """Perform the forward pass of the ResNet."""
        x = self.layer_dict["input2bulk"](inputs)
        x = self.layer_dict["input2bulk_act"](x) if self.pre_activation else x
        x = self.layer_dict["batch_norm_0"](x) if self.has_batch_norm else x
        x = self.layer_dict["input_dropout"](x) if self.has_input_dropout else x
        x - self.layer_dict["input2bulk_act"](x) if not self.pre_activation else x

        # Apply bulk layers with residual connections
        for i in range(self.depth):
            y = x

            # Apply bulk layer:
            # dense -> (pre-activation) -> batch norm -> dropout -> (post-activation)
            x = self.layer_dict[f"resnet_bulk_{i}"](x)
            x = self.layer_dict[f"resnet_bulk_act_{i}"](x) if self.pre_activation else x
            x = self.layer_dict[f"batch_norm_{i + 1}"](x) if self.has_batch_norm else x
            x = self.layer_dict[f"dropout_{i}"](x) if self.has_dropout else x
            x = (
                self.layer_dict[f"resnet_bulk_act_{i}"](x)
                if not self.pre_activation
                else x
            )

            # Add residual connection when reaching the end of a block
            if (i + 1) % self.layers_per_block == 0 or i == self.depth - 1:
                x = x + y
                x = (
                    self.layer_dict[f"post_add_act{i // self.layers_per_block}"](x)
                    if self.post_add_activation
                    else x
                )

        output = self.layer_dict["output"](x)
        output = (
            self.layer_dict["output_act"](output)
            if self.has_output_activation
            else output
        )

        return output
