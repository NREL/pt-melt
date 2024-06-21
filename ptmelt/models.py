import warnings
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from ptmelt.blocks import DefaultOutput, DenseBlock, ResidualBlock


class MELTModel(nn.Module):
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
        node_list: Optional[list] = None,
        **kwargs,
    ):
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
        self.node_list = node_list

        # Determine if network should be defined based on depth/width or node_list
        if self.node_list:
            self.num_layers = len(self.node_list)
            self.layer_width = self.node_list
        else:
            self.num_layers = self.depth
            self.layer_width = [self.width for i in range(self.depth)]

        # Create list for storing names of sub-layers
        self.sub_layer_names = []

        # Create layer dictionary
        self.layer_dict = nn.ModuleDict()

    def build(self):
        """Build the model."""
        self.initialize_layers()

    def initialize_layers(self):
        """Initialize the layers of the model."""
        self.create_dropout_layers()
        self.create_output_layer()

    def create_dropout_layers(self):
        """Create the dropout layers."""
        if self.input_dropout > 0:
            self.layer_dict.update({"input_dropout": nn.Dropout(p=self.input_dropout)})

    def create_output_layer(self):
        """Create the output layer."""
        self.layer_dict.update(
            {
                "output": DefaultOutput(
                    input_features=self.layer_width[-1],
                    output_features=self.num_outputs,
                    activation=self.output_activation,
                    initializer=self.initializer,
                )
            }
        )
        self.sub_layer_names.append("output")

    def compute_jacobian(self, x):
        """Compute the Jacobian of the model with respect to the input."""
        pass

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
    """
    Artificial Neural Network (ANN) model.

    Args:
        **kwargs: Additional keyword arguments.

    """

    def __init__(
        self,
        **kwargs,
    ):
        super(ArtificialNeuralNetwork, self).__init__(**kwargs)

    def initialize_layers(self):
        """Initialize the layers of the ANN."""
        super(ArtificialNeuralNetwork, self).initialize_layers()

        # Bulk layers
        self.layer_dict.update(
            {
                "dense_block": DenseBlock(
                    input_features=self.num_features,
                    node_list=self.layer_width,
                    activation=self.act_fun,
                    dropout=self.dropout,
                    batch_norm=self.batch_norm,
                    batch_norm_type=self.batch_norm_type,
                    use_batch_renorm=self.use_batch_renorm,
                    initializer=self.initializer,
                )
            }
        )
        self.sub_layer_names.append("dense_block")

    def forward(self, inputs):
        """Perform the forward pass of the ANN."""
        # Apply input dropout
        x = (
            self.layer_dict["input_dropout"](inputs)
            if self.input_dropout > 0
            else inputs
        )

        # Apply dense block
        x = self.layer_dict["dense_block"](x)

        # Apply the output layer(s) and return
        return self.layer_dict["output"](x)


class ResidualNeuralNetwork(MELTModel):
    """
    Residual Neural Network (ResNet) model.

    Args:
        layers_per_block (int, optional): The number of layers per residual block.
        pre_activation (bool, optional): Whether to use pre-activation.
        post_add_activation (bool, optional): Whether to use activation after
                                              addition.
        **kwargs: Additional keyword arguments.

    """

    def __init__(
        self,
        layers_per_block: Optional[int] = 2,
        pre_activation: Optional[bool] = True,
        post_add_activation: Optional[bool] = False,
        **kwargs,
    ):
        super(ResidualNeuralNetwork, self).__init__(**kwargs)

        self.layers_per_block = layers_per_block
        self.pre_activation = pre_activation
        self.post_add_activation = post_add_activation

    def build(self):
        """Build the model."""
        if self.depth % self.layers_per_block != 0:
            warnings.warn(
                f"Warning: depth {self.num_layers} is not divisible by "
                f"layers_per_block ({self.layers_per_block}), so the last block will "
                f"have {self.depth % self.layers_per_block} layers."
            )

        self.initialize_layers()
        super(ResidualNeuralNetwork, self).build()

    def initialize_layers(self):
        """Initialize the layers of the ResNet."""
        super(ResidualNeuralNetwork, self).initialize_layers()

        # Create the Residual Block
        self.layer_dict.update(
            {
                "residual_block": ResidualBlock(
                    layers_per_block=self.layers_per_block,
                    pre_activation=self.pre_activation,
                    post_add_activation=self.post_add_activation,
                    input_features=self.num_features,
                    node_list=self.layer_width,
                    activation=self.act_fun,
                    dropout=self.dropout,
                    batch_norm=self.batch_norm,
                    batch_norm_type=self.batch_norm_type,
                    use_batch_renorm=self.use_batch_renorm,
                    initializer=self.initializer,
                )
            }
        )
        self.sub_layer_names.append("residual_block")

    def forward(self, inputs):
        """Perform the forward pass of the ResNet."""
        # Apply input dropout
        x = (
            self.layer_dict["input_dropout"](inputs)
            if self.input_dropout > 0
            else inputs
        )

        # Apply residual block
        x = self.layer_dict["residual_block"](x)

        # Apply the output layer(s) and return
        return self.layer_dict["output"](x)
