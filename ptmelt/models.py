import warnings
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from ptmelt.blocks import DefaultOutput, DenseBlock, MixtureDensityOutput, ResidualBlock
from ptmelt.losses import MixtureDensityLoss, VAELoss


class MELTModel(nn.Module):
    """
    PT-MELT Base model.

    Args:
        num_features (int): The number of input features.
        num_outputs (int): The number of output units.
        width (int, optional): The width of the hidden layers. Defaults to 32.
        depth (int, optional): The number of hidden layers. Defaults to 2.
        act_fun (str, optional): The activation function to use. Defaults to 'relu'.
        dropout (float, optional): The dropout rate. Defaults to 0.0.
        input_dropout (float, optional): The input dropout rate. Defaults to 0.0.
        batch_norm (bool, optional): Whether to use batch normalization. Defaults to
                                     False.
        batch_norm_type (str, optional): The type of batch normalization to use.
                                         Defaults to 'ema'.
        use_batch_renorm (bool, optional): Whether to use batch renormalization.
                                           Defaults to False.
        output_activation (str, optional): The activation function for the output layer.
                                           Defaults to None.
        initializer (str, optional): The weight initializer to use. Defaults to
                                     'glorot_uniform'.
        l1_reg (float, optional): The L1 regularization strength. Defaults to 0.0.
        l2_reg (float, optional): The L2 regularization strength. Defaults to 0.0.
        num_mixtures (int, optional): The number of mixture components for MDN. Defaults
                                      to 0.
        node_list (list, optional): The list of nodes per layer to alternately define
                                    layers. Defaults to None.
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
        num_mixtures: Optional[int] = 0,
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
        self.num_mixtures = num_mixtures
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
        if self.num_mixtures > 0:
            self.layer_dict.update(
                {
                    "output": MixtureDensityOutput(
                        input_features=self.layer_width[-1],
                        num_mixtures=self.num_mixtures,
                        num_outputs=self.num_outputs,
                        activation=self.output_activation,
                        initializer=self.initializer,
                    )
                }
            )
            self.sub_layer_names.append("output")

        else:
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

    def l1_regularization(self, lambda_l1: float):
        """
        Compute the L1 regularization term for use in the loss function.

        Args:
            lambda_l1 (float): The L1 regularization strength.
        """
        l1_norm = sum(
            p.abs().sum()
            for name, p in self.named_parameters()
            if p.requires_grad and "weight" in name
        )
        return lambda_l1 * l1_norm

    def l2_regularization(self, lambda_l2: float):
        """
        Compute the L2 regularization term for use in the loss function.

        Args:
            lambda_l2 (float): The L2 regularization strength.
        """
        l2_norm = sum(
            p.pow(2.0).sum()
            for name, p in self.named_parameters()
            if p.requires_grad and "weight" in name
        )
        return 0.5 * lambda_l2 * l2_norm

    def get_loss_fn(
        self, loss: Optional[str] = "mse", reduction: Optional[str] = "mean"
    ):
        """
        Get the loss function for the model. Used in the training loop.

        Args:
            loss (str, optional): The loss function to use. Defaults to 'mse'.
            reduction (str, optional): The reduction method for the loss. Defaults to
                                       'mean'.
        """
        if self.special_loss == "mdn":
            warnings.warn(
                "Mixture Density Networks require the use of the MixtureDensityLoss "
                "class. The loss function will be set to automatically."
            )

            return MixtureDensityLoss(
                num_mixtures=self.num_mixtures, num_outputs=self.num_outputs
            )
        elif self.special_loss == "vae":
            warnings.warn(
                "Variational Autoencoders require the use of the VAELoss class. The "
                "loss function will be set to automatically."
            )

            return VAELoss()

        elif loss == "mse":
            return nn.MSELoss(reduction=reduction)
        else:
            raise ValueError(f"Loss function {loss} not recognized.")

    def fit(self, train_dl, val_dl, optimizer, criterion, num_epochs: int):
        """
        Perform the model training loop.

        Args:
            train_dl (DataLoader): The training data loader.
            val_dl (DataLoader): The validation data loader.
            optimizer (Optimizer): The optimizer to use.
            criterion (Loss): The loss function to use.
            num_epochs (int): The number of epochs to train the model.
        """

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

        if self.num_mixtures > 0:
            self.special_loss = "mdn"

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

    def forward(self, inputs: torch.Tensor):
        """
        Perform the forward pass of the ANN.

        Args:
            inputs (torch.Tensor): The input data.
        """
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
                                          Defaults to 2.
        pre_activation (bool, optional): Whether to use pre-activation. Defaults to
                                         True.
        post_add_activation (bool, optional): Whether to use activation after
                                              addition. Defaults to False.
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

        if self.num_mixtures > 0:
            self.special_loss = "mdn"

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

    def forward(self, inputs: torch.Tensor):
        """
        Perform the forward pass of the ResNet.

        Args:
            inputs (torch.Tensor): The input data.
        """
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


class VariationalAutoEncoder(MELTModel):
    def __init__(
        self,
        encode_node_list: Optional[list] = [32, 32],
        decode_node_list: Optional[list] = [32, 32],
        latent_dim: Optional[int] = 2,
        **kwargs,
    ):
        super(VariationalAutoEncoder, self).__init__(**kwargs)

        self.encode_node_list = encode_node_list
        self.decode_node_list = decode_node_list
        self.latent_dim = latent_dim

        self.special_loss = "vae"

    def initialize_layers(self):
        # super(VariationalAutoEncoder, self).initialize_layers()

        # Create the encoder dense block
        self.layer_dict.update(
            {
                "encoder_dense": DenseBlock(
                    input_features=self.num_features,
                    node_list=self.encode_node_list,
                    activation=self.act_fun,
                    dropout=self.dropout,
                    batch_norm=self.batch_norm,
                    batch_norm_type=self.batch_norm_type,
                    use_batch_renorm=self.use_batch_renorm,
                    initializer=self.initializer,
                )
            }
        )
        self.sub_layer_names.append("encoder_dense")

        # Create the latent layer
        self.layer_dict.update(
            {
                "encoder_output": MixtureDensityOutput(
                    input_features=self.encode_node_list[-1],
                    num_mixtures=self.num_mixtures,
                    num_outputs=self.latent_dim,
                    activation=self.output_activation,
                    initializer=self.initializer,
                )
            }
        )
        self.sub_layer_names.append("encoder_output")

        # Create the decoder
        self.layer_dict.update(
            {
                "decoder": DenseBlock(
                    input_features=self.latent_dim,
                    node_list=self.decode_node_list,
                    activation=self.act_fun,
                    dropout=self.dropout,
                    batch_norm=self.batch_norm,
                    batch_norm_type=self.batch_norm_type,
                    use_batch_renorm=self.use_batch_renorm,
                    initializer=self.initializer,
                )
            }
        )
        self.sub_layer_names.append("decoder")

        # Create the output layer
        self.layer_dict.update(
            {
                "output": DefaultOutput(
                    input_features=self.decode_node_list[-1],
                    output_features=self.num_features,
                    activation=self.output_activation,
                    initializer=self.initializer,
                )
            }
        )
        self.sub_layer_names.append("output")

    def forward(self, inputs: torch.Tensor):
        # First apply the encoder

        # input dropout before the encoder
        x = (
            self.layer_dict["input_dropout"](inputs)
            if self.input_dropout > 0
            else inputs
        )

        # Apply the encoder dense block
        x = self.layer_dict["encoder_dense"](x)

        # Apply the encoder output layer to get the latent representation
        x = self.layer_dict["encoder_output"](x)
        # print(f"Encoder output shape: {x.shape}")

        # separate the outputs from the MDN layer into components and construct the latent representation
        m_coeffs = x[:, : self.num_mixtures]
        mean_preds = x[
            :,
            self.num_mixtures : self.num_mixtures + self.num_mixtures * self.latent_dim,
        ]
        log_var_preds = x[:, self.num_mixtures + self.num_mixtures * self.latent_dim :]

        # Normalize the mixture coefficients so they sum to 1 (though they should already)
        m_coeffs = torch.nn.functional.softmax(m_coeffs, dim=1)

        # Sample a component from the mixture
        component = torch.multinomial(m_coeffs, num_samples=1)
        # print(f"Shape of component: {component.shape}")
        # Expand the size of component to be (batch_size, latent_dim)
        component = component.expand(-1, self.latent_dim)
        # print(f"Shape of component after view: {component.shape}")

        # Extract the mean and log-variance for the selected component
        mean = mean_preds.gather(1, component).squeeze(1)
        log_var = log_var_preds.gather(1, component).squeeze(1)

        # Sample from the Gaussian distribution
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        # z = eps.mul(std).add_(mean)
        z = mean + eps * std

        # Apply the decoder
        # print(f"Shape of z: {z.shape}")
        recon_x = self.layer_dict["decoder"](z)

        # Apply the output layer
        out = self.layer_dict["output"](recon_x)

        # Return reconstructed input, mean, and log-variance
        return out, mean, log_var
