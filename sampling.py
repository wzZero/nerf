from typing import Optional, Tuple

import torch


def stratified_sample(
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        near: float,
        far: float,
        radii: float,
        n_samples: int,
        perturb: Optional[bool] = True,
        inverse_depth: Optional[bool] = False,
):
    t_vals = torch.linspace(0.0, 1.0, n_samples + 1, device=rays_o.device)

    if not inverse_depth:
        z_vals = near * (1.0 - t_vals) + far * t_vals
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)

    if perturb:
        mids = 0.5 * (z_vals[1:] + z_vals[:-1])
        upper = torch.cat([mids, z_vals[-1:]])
        lower = torch.cat([z_vals[:1], mids])
        t_rand = torch.rand([n_samples], device=z_vals.device)
        z_vals = lower + (upper - lower) * t_rand
    z_vals = z_vals.expand(list(rays_o.shape[:-1]) + [n_samples])
    means, covs = cast_rays(z_vals, rays_o, rays_d, radii)
    return (means, covs), z_vals



def hierarchical_sample(
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        z_vals: torch.Tensor,
        weights: torch.Tensor,
        radii: float,
        n_samples: int,
        perturb: Optional[bool] = False
):

    weights_pad = torch.cat([weights[..., :1], weights, weights[..., -1:]], dim=-1)
    weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
    weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

    new_z_samples = sample_pdf(z_vals, weights_blur, n_samples+1, perturb=perturb)
    new_z_samples = new_z_samples.detach()

    z_vals_combined, _ = torch.sort(torch.cat([z_vals, new_z_samples], dim=-1), dim=-1)
    means, covs = cast_rays(z_vals, rays_o, rays_d, radii)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_combined[..., :, None]
    return (means, covs), new_z_samples


def sample_pdf(
        bins: torch.Tensor,
        weights: torch.Tensor,
        n_samples: int,
        perturb: Optional[bool] = False
) -> torch.Tensor:
    pdf = (weights + 1e-5) / torch.sum(weights + 1e-5, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
    if perturb:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=cdf.device)
    else:
        u = torch.linspace(0.0, 1.0, n_samples, device=cdf.device)
    # torch.Size([2500, 64])
    u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    # torch.Size([2500, 64])
    inds = torch.searchsorted(cdf, u, right=True)
    # torch.Size([2500, 64])
    below = torch.clamp_min(inds - 1, 0)
    # torch.Size([2500, 64])
    above = torch.clamp_max(inds, cdf.shape[-1] - 1)
    inds_g = torch.stack([below, above], dim=-1)
    # torch.Size([2500, 64, 2])

    matched_shape = list(inds_g.shape[:-1]) + [cdf.shape[-1]]
    # [2500, 64, 63]
    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), -1, inds_g)
    # torch.Size([2500, 64, 2])
    bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), -1, inds_g)
    # torch.Size([2500, 64, 2])
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    # torch.Size([2500, 64])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    # torch.Size([2500, 64])
    t = (u - cdf_g[..., 0]) / denom
    # torch.Size([2500, 64])
    samples = bins_g[..., 0] + (bins_g[..., 1] - bins_g[..., 0]) * t
    # torch.Size([2500, 64])
    return samples


########### mipnerf sampling helper function #############
def lift_gaussian(rays_d, t_mean, t_var, r_var):
    """Lift a Gaussian defined along a ray to 3D coordinates."""
    mean = torch.unsqueeze(rays_d, dim=-2) * torch.unsqueeze(t_mean, dim=-1)  # [B, 1, 3]*[B, N, 1] = [B, N, 3]
    d_norm_denominator = torch.sum(rays_d ** 2, dim=-1, keepdim=True) + 1e-10
    # min_denominator = torch.full_like(d_norm_denominator, 1e-10)
    # d_norm_denominator = torch.maximum(min_denominator, d_norm_denominator)

    d_outer_diag = rays_d ** 2  # eq (16)
    null_outer_diag = 1 - d_outer_diag / d_norm_denominator
    t_cov_diag = torch.unsqueeze(t_var, dim=-1) * torch.unsqueeze(d_outer_diag,
                                                                  dim=-2)  # [B, N, 1] * [B, 1, 3] = [B, N, 3]
    xy_cov_diag = torch.unsqueeze(r_var, dim=-1) * torch.unsqueeze(null_outer_diag, dim=-2)
    cov_diag = t_cov_diag + xy_cov_diag
    return mean, cov_diag


def conical_frustum_to_gaussian(rays_d, t0, t1, base_radius):
    """Approximate a conical frustum as a Gaussian distribution (mean+cov).
    Assumes the ray is originating from the origin, and base_radius is the
    radius at dist=1. Doesn't assume `directions` is normalized.
    Args:
        directions: torch.tensor float32 3-vector, the axis of the cone
        t0: float, the starting distance of the frustum.
        t1: float, the ending distance of the frustum.
        base_radius: float, the scale of the radius as a function of distance.
        diagonal: boolean, whether or the Gaussian will be diagonal or full-covariance.
        stable: boolean, whether or not to use the stable computation described in
        the paper (setting this to False will cause catastrophic failure).
    Returns:
        a Gaussian (mean and covariance).
    """

    mu = (t0 + t1) / 2
    hw = (t1 - t0) / 2
    t_mean = mu + (2 * mu * hw ** 2) / (3 * mu ** 2 + hw ** 2)
    t_var = (hw ** 2) / 3 - (4 / 15) * ((hw ** 4 * (12 * mu ** 2 - hw ** 2)) /
                                        (3 * mu ** 2 + hw ** 2) ** 2)
    r_var = base_radius ** 2 * ((mu ** 2) / 4 + (5 / 12) * hw ** 2 - 4 / 15 *
                                (hw ** 4) / (3 * mu ** 2 + hw ** 2))

    return lift_gaussian(rays_d, t_mean, t_var, r_var)


def cast_rays(z_vals, rays_o, rays_d, radii):
    """Cast rays (cone-shaped) and featurize sections of it.
    Args:
        z_vals: float array [B, n_sample+1], the "fencepost" distances along the ray.
        rays_o: float array [B, 3], the ray origin coordinates.
        rays_d [B, 3]: float array, the ray direction vectors.
        radii[B, 1]: float array, the radii (base radii for cones) of the rays.
    Returns:
        a tuple of arrays of means and covariances.
    """
    t0 = z_vals[..., :-1]  # [B, n_samples]
    t1 = z_vals[..., 1:]

    means, covs = conical_frustum_to_gaussian(rays_d, t0, t1, radii)
    means = means + torch.unsqueeze(rays_o, dim=-2)
    return means, covs
