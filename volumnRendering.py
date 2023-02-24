from typing import Tuple

import torch
import torch.nn.functional as F

from utils import cumprod_exclusive

# different from nerf
def raw2outputs(
    raw : torch.Tensor,
    z_vals : torch.Tensor,
    rays_d : torch.Tensor,
    raw_noise_std : float = 0.0,
    white_bkgd : bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert the raw NeRF output into RGB and other maps.
    """

    # Difference between consecutive elements of `z_vals`. [n_rays, n_samples]
    z_mids = 0.5 * (z_vals[..., :-1] + z_vals[..., 1:])
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    # dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], dim=-1)

    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # Add noise to model's predictions for density. Can be used to
    # regularize network during training (prevents floater artifacts).
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

    # Predict density of each sample along each ray. Higher values imply
    # higher likelihood of being absorbed at this point. [n_rays, n_samples]
    alpha = 1.0 - torch.exp(-F.relu(raw[..., 3] + noise) * dists)

    # Compute weight for RGB of each sample along each ray. [n_rays, n_samples]
    # The higher the alpha, the lower subsequent weights are driven.
    weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)

    # Compute weighted RGB map.
    rgb = torch.sigmoid(raw[..., :3])  # [n_rays, n_samples, 3]
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # [n_rays, 3]

    # Estimated depth map is predicted distance.
    depth_map = torch.sum(weights * z_mids, dim=-1)

    # Disparity map is inverse depth.
    disp_map = 1. / torch.max(
        1e-10 * torch.ones_like(depth_map),
        depth_map / torch.sum(weights, -1)
    )

    # Sum of weights along each ray. In [0, 1] up to numerical error.
    acc_map = torch.sum(weights, dim=-1)

    # To composite onto a white background, use the accumulated alpha map.
    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])
    return rgb_map, depth_map, acc_map, weights
