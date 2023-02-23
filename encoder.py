import torch
import torch.nn as nn
import numpy as np

from einops import rearrange

class PositionalEncoder(nn.Module):
    """
    Sine-cosine positional encoder for input points.
    """
    def __init__(self, d_input, n_freqs, log_space=False):
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.d_output = d_input * (1 + 2 * self.n_freqs)
        self.log_space = log_space
        self.embed_fns = [lambda x: x]

        # Define frequencies in linear scale or log scale
        if self.log_space:
            freq_bands = torch.logspace(0., 1., self.n_freqs, base=2.)
        else:
            freq_bands = torch.linspace(1, 2**(self.n_freqs - 1), self.n_freqs)

        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))

    def forward(self, x):
        return torch.cat([fn(x) for fn in self.embed_fns], dim=-1)

### IPE helper functions

def expected_sin(x, x_var):
    """Estimates mean and variance of sin(z), z ~ N(x, var)."""
    y = torch.exp(-0.5 * x_var) * torch.sin(x)  # [B, N, 2*3*L]
    y_var = 0.5 * (1 - torch.exp(-2 * x_var) * torch.cos(2 * x)) - y ** 2
    y_var = torch.maximum(torch.zeros_like(y_var), y_var)
    return y, y_var


class IntegratedPositionalEncoder(nn.Module):
    def __init__(self, min_deg, max_deg, diagonal=True):
        super().__init__()
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.diagonal = diagonal
        self.degrees = range(self.min_deg, self.max_deg)

    def forward(self, means_covs):
        if self.diagonal:
            means, covs_diag = means_covs
            scales = torch.tensor([2 ** i for i in self.degrees], device=means.device)  # [L]
            # [B, N, 1, 3] * [L, 1] = [B, N, L, 3]->[B, N, 3L]
            y = rearrange(torch.unsqueeze(means, dim=-2) * torch.unsqueeze(scales, dim=-1),
                          'batch sample scale_dim mean_dim -> batch sample (scale_dim mean_dim)')
            # [B, N, 1, 3] * [L, 1] = [B, N, L, 3]->[B, N, 3L]
            y_var = rearrange(torch.unsqueeze(covs_diag, dim=-2) * torch.unsqueeze(scales, dim=-1) ** 2,
                              'batch sample scale_dim cov_dim -> batch sample (scale_dim cov_dim)')
        else:
            means, x_cov = means_covs
            num_dims = means.shape[-1]
            # [3, L]
            basis = torch.cat([2 ** i * torch.eye(num_dims, device=means.device) for i in self.degrees], 1)
            y = torch.matmul(means, basis)  # [B, N, 3] * [3, 3L] = [B, N, 3L]
            y_var = torch.sum((torch.matmul(x_cov, basis)) * basis, -2)
        return expected_sin(torch.cat([y, y + 0.5 * torch.tensor(np.pi)], dim=-1), torch.cat([y_var] * 2, dim=-1))[0]
