import torch
import torch.nn as nn


def safe_exp(x):
    """Prevents overflow by clipping input range to reasonable values."""
    x = torch.clamp(x, min=-20, max=20)
    return torch.exp(x)


class MixtureDensityLoss(nn.Module):
    """
    Custom loss function for a Gaussian mixture model.

    Args:
        num_mixtures (int): Number of mixture components.
        num_outputs (int): Number of output dimensions.
    """

    def __init__(self, num_mixtures, num_outputs):
        super(MixtureDensityLoss, self).__init__()
        self.num_mixtures = num_mixtures
        self.num_outputs = num_outputs

    def forward(self, y_pred, y_true):
        # NOTE: the order of the parameters is reversed compared to Keras and TensorFlow
        # Extract the mixture coefficients, means, and log-variances
        end_mixture = self.num_mixtures
        end_mean = end_mixture + self.num_mixtures * self.num_outputs
        end_log_var = end_mean + self.num_mixtures * self.num_outputs

        m_coeffs = y_pred[:, :end_mixture]
        mean_preds = y_pred[:, end_mixture:end_mean]
        log_var_preds = y_pred[:, end_mean:end_log_var]

        # Reshape to ensure same shape as y_true replicated across mixtures
        mean_preds = mean_preds.view(-1, self.num_mixtures, self.num_outputs)
        log_var_preds = log_var_preds.view(-1, self.num_mixtures, self.num_outputs)

        # Calculate the Gaussian probability density function for each component
        const_term = -0.5 * self.num_outputs * torch.log(torch.tensor(2 * torch.pi))
        inv_sigma_log = -0.5 * log_var_preds
        exp_term = (
            -0.5
            * torch.square(y_true.unsqueeze(1) - mean_preds)
            / safe_exp(log_var_preds)
        )

        # form the log probabilities
        log_probs = const_term + inv_sigma_log + exp_term

        # Calculate the log likelihood
        weighted_log_probs = log_probs + torch.log(m_coeffs.unsqueeze(-1))
        log_sum_exp = torch.logsumexp(weighted_log_probs, dim=1)

        # Compute the log likelihood loss
        log_likelihood = torch.mean(log_sum_exp)

        # Return the negative log likelihood
        return -log_likelihood


class VAELoss(nn.Module):
    def __init__(self, **kwargs):
        super(VAELoss, self).__init__(**kwargs)

    # def forward(self, x, x_recon, mu, log_var):
    def forward(self, y_pred, y_true):
        x_recon = y_pred[0]
        mu = y_pred[1]
        log_var = y_pred[2]
        x = y_true
        # Compute the reconstruction loss
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction="mean")

        # Compute the KL divergence
        kl_div = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        return recon_loss + kl_div
