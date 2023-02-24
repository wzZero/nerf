from collections import namedtuple
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

from dataset import Ray_keys, Rays

def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    """
    https://github.com/krrish94/nerf-pytorch

    Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.

    Args:
        * tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
        is to be computed.
    Returns:
        * cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
    """
    # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
    cumprod = torch.cumprod(tensor, -1)
    # "Roll" the elements along dimension 'dim' by 1 element.
    cumprod = torch.roll(cumprod, 1, -1)
    # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
    cumprod[..., 0] = 1.0

    return cumprod


def plot_samples(z_vals: torch.Tensor,
                 z_hierarch: Optional[torch.Tensor] = None,
                 ax: Optional[np.ndarray] = None):
    """
    Plot stratified and (optional) hierarchical samples.
    """
    y_vals = 1 + np.zeros_like(z_vals)
    if ax is None:
        ax = plt.subplot()
    ax.plot(z_vals, y_vals, 'b-o')
    if z_hierarch is not None:
        y_hierarch = np.zeros_like(z_hierarch)
        ax.plot(z_hierarch, y_hierarch, 'r-o')
    ax.set_ylim([-1, 2])
    ax.set_title('Stratified  Samples (blue) and Hierarchical Samples (red)')
    ax.axes.yaxis.set_visible(False)
    ax.grid(True)
    return ax

def rearrange_render_image(rays, chunk_size=4096):
    # change Rays to list: [origins, directions, viewdirs, radii, lossmult, near, far]
    single_image_rays = [getattr(rays, key) for key in Ray_keys]
    val_mask = single_image_rays[-3]

    # flatten each Rays attribute and put on device
    single_image_rays = [rays_attr.reshape(-1, rays_attr.shape[-1]) for rays_attr in single_image_rays]
    # get the amount of full rays of an image
    length = single_image_rays[0].shape[0]
    # divide each Rays attr into N groups according to chunk_size,
    # the length of the last group <= chunk_size
    single_image_rays = [[rays_attr[i:i + chunk_size] for i in range(0, length, chunk_size)] for
                         rays_attr in single_image_rays]
    # get N, the N for each Rays attr is the same
    length = len(single_image_rays[0])
    # generate N Rays instances
    single_image_rays = [Rays(*[rays_attr[i] for rays_attr in single_image_rays]) for i in range(length)]
    return single_image_rays, val_mask