import warnings
from typing import Any, Optional

import numpy as np
import torch


def make_predictions(
    model,
    x_data,
    y_normalizer: Optional[Any] = None,
    unnormalize: Optional[bool] = False,
    training: Optional[bool] = False,
):
    """
    Make predictions using the provided model and optionally unscaling the results.

    Args:
        model (torch.nn.Module): A PyTorch model.
        x_data (np.ndarray or torch.Tensor): Input data.
        y_normalizer (scaler, optional): Scikit-learn like scaler object. Defaults to
                                         None.
        unnormalize (bool, optional): Whether to unnormalize the predictions. Defaults
                                      to False.
        training (bool, optional): Whether to use training mode for making predictions.
                                   Defaults to False.
    """
    # Set model to either training or evaluation mode
    model.train() if training else model.eval()

    # Make predictions
    if type(x_data) is np.ndarray:
        x_data = torch.from_numpy(x_data).float()

    if model.num_mixtures > 0:
        pred_array = model(x_data)
        m_coeffs = pred_array[:, : model.num_mixtures]
        mean_preds = pred_array[
            :,
            model.num_mixtures : model.num_mixtures
            + model.num_mixtures * model.num_outputs,
        ]
        log_var_preds = pred_array[
            :, model.num_mixtures + model.num_mixtures * model.num_outputs :
        ]

        # Reshape mean and log_var predictions to separate the mixture components
        mean_preds = mean_preds.view(-1, model.num_mixtures, model.num_outputs)
        log_var_preds = log_var_preds.view(-1, model.num_mixtures, model.num_outputs)

        # Compute the weighted mean and total variance of the predictions
        m_coeffs = m_coeffs.unsqueeze(-1)
        mean_preds_weighted = torch.sum(mean_preds * m_coeffs, axis=1)

        variance_preds_weighted = (
            torch.sum((mean_preds**2 + torch.exp(log_var_preds)) * m_coeffs, axis=1)
            - mean_preds_weighted**2
        )
        std_preds_weighted = torch.sqrt(variance_preds_weighted)

        predictions = mean_preds_weighted.detach().numpy()
        std_pred = std_preds_weighted.detach().numpy()

    else:
        predictions = model(x_data).detach().numpy()
        std_pred = None

    # Unscale the results if required
    if unnormalize and y_normalizer is not None:
        predictions = y_normalizer.inverse_transform(predictions)
        if std_pred is not None:
            std_pred = np.float32(y_normalizer.scale_) * std_pred
    elif unnormalize and y_normalizer is None:
        raise ValueError("y_normalizer must be provided to unnormalize predictions.")

    if model.num_mixtures > 0:
        return predictions, std_pred
    else:
        return predictions


def ensemble_predictions(
    model,
    x_data,
    y_normalizer: Optional[Any] = None,
    unnormalize: Optional[bool] = False,
    n_iter: Optional[int] = 100,
    training: Optional[bool] = False,
):
    """
    Make ensemble predictions using the provided model and optionally unscaling the
    results. The ensemble predictions are computed by making multiple predictions and
    calculating the mean and standard deviation of the predictions.

    Args:
        model (torch.nn.Module): A PyTorch model.
        x_data (np.ndarray or torch.Tensor): Input data.
        y_normalizer (scaler, optional): Scikit-learn like scaler object. Defaults to
                                         None.
        unnormalize (bool, optional): Whether to unnormalize the predictions. Defaults
                                      to False.
        n_iter (int, optional): Number of iterations for making predictions. Defaults
                                to 100.
        training (bool, optional): Whether to use training mode for making predictions.
                                   Defaults to False.
    """
    # Set model to either training or evaluation mode
    model.train() if training else model.eval()

    # Make predictions
    if type(x_data) is np.ndarray:
        x_data = torch.from_numpy(x_data).float()

    predictions = []
    for _ in range(n_iter):
        pred = model(x_data).detach().numpy()
        if unnormalize and y_normalizer is not None:
            pred = y_normalizer.inverse_transform(pred)
        elif unnormalize and y_normalizer is None:
            raise ValueError(
                "y_normalizer must be provided to unnormalize predictions."
            )
        elif not unnormalize and y_normalizer is not None:
            warnings.warn(
                "y_normalizer provided but unnormalize set to False. "
                "Predictions will be in normalized space."
            )

        predictions.append(pred)

    predictions = np.array(predictions)
    pred_mean = np.mean(predictions, axis=0)
    pred_std = np.std(predictions, axis=0)

    return pred_mean, pred_std
