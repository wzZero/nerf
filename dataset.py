import numpy as np
import torch
import collections
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Tuple

from settings import Settings

Rays = collections.namedtuple(
    'Rays',
    ('origins', 'directions', 'radii'))

#TODO: DATASET
class MipNeRFDataset(Dataset):
    def __init__(self, filepath, n_training, device):
        data = np.load(filepath)
        self.images = torch.from_numpy(data['images']).to(device)
        self.height, self.width = self.images.shape[1:3]
        self.poses = torch.from_numpy(data['poses']).to(device)
        self.focal = torch.from_numpy(data['focal']).to(device)
        self.n_training = n_training
        self.rays = None
        self.get_rays()
        # original shape [n_training, 2, height, width, 3]
        self.target_rgb = self.images.reshape(-1, 3)

    def __len__(self):
        return self.n_training * self.height * self.width

    def __getitem__(self, idx):
        return self.target_rgb[idx]

    def get_rays(self,
                 height, width, focal_length,
                 c2w: torch.Tensor
                 )
        """
        Returns:
            * rays_o: [height, width, 3] ray origins
            * rays_d: [height, width, 3] ray directions
        """
        # Apply pinhole camera model to gather directions at each pixel
        i, j = torch.meshgrid(
            torch.arange(self.width, dtype=torch.float32).to(c2w),
            torch.arange(self.height, dtype=torch.float32).to(c2w),
            indexing='ij'
        )
        i = i.transpose(-1, -2)
        j = j.transpose(-1, -2)
        directions = torch.stack(
            [
                (i - width * 0.5) / self.focal,
                -(j - height * 0.5) / self.focal,
                -torch.ones_like(i)
            ],
            dim=-1
        )

        # Apply camera pose to directions
        rays_d = torch.sum(directions[..., None, :] * c2w[:3, :3], dim=-1)
        rays_o = c2w[:3, -1].expand(rays_d.shape)

        dx = [
            np.sqrt(np.sum((v[:-1, :, :] - v[1:, :, :]) ** 2, -1)) for v in directions
        ]
        dx = [np.concatenate([v, v[-2:-1, :]], 0) for v in dx]
        radii = [v[..., None] * 2 / np.sqrt(12) for v in dx]

        self.rays = Rays(
            origins=rays_o,
            directions=rays_d,
            radii=radii,
        )



def load_dataset(batch_size, device):
    settings = Settings()
    dataset = MipNeRFDataset("dataset/tiny_nerf_data.npz", settings.n_training, device)
    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset, shuffle=False, sampler=sampler, batch_size=batch_size)
    return loader, sampler, dataset