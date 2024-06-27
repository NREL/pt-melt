import torch.nn as nn


def get_activation(act_name):
    """Utility method to get activation based on its name."""
    if act_name == "relu":
        return nn.ReLU()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "linear" or act_name is None:
        return nn.Identity()
    elif act_name == "softmax":
        return nn.Softmax(dim=-1)
    else:
        raise ValueError(f"Unsupported activation function {act_name}")


def get_initializer(init_name):
    """Utility method to get initializer based on its name."""
    if init_name == "glorot_uniform":
        return nn.init.xavier_uniform_
    elif init_name == "glorot_normal":
        return nn.init.xavier_normal_
    elif init_name == "he_uniform":
        return nn.init.kaiming_uniform_
    elif init_name == "he_normal":
        return nn.init.kaiming_normal_
    elif init_name == "normal":
        return nn.init.normal_
    elif init_name == "uniform":
        return nn.init.uniform_
    else:
        raise ValueError(f"Unsupported initializer {init_name}")
