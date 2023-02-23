import json
from os import path

import numpy as np
import torch
import collections
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import cv2
from PIL import Image

from settings import Settings

Rays = collections.namedtuple(
    'Rays',
    ('origins', 'directions', 'radii'))
Ray_keys = Rays._fields

def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*map(fn, tup))

class BaseDataset(Dataset):
    """BaseDataset Base Class."""

    def __init__(self, data_dir, split, white_bkgd=False, factor=0):
        super(BaseDataset, self).__init__()
        self.near = 2
        self.far = 6
        self.split = split
        self.data_dir = data_dir
        self.white_bkgd = white_bkgd
        self.images = None
        self.rays = None
        self.it = -1
        self.n_examples = 1
        self.factor = factor

    def _flatten(self, x):
        # Always flatten out the height x width dimensions
        x = [y.reshape([-1, y.shape[-1]]) for y in x]
        x = np.concatenate(x, axis=0)
        return x

    def _train_init(self):
        """Initialize training."""

        self._load_images()
        self._generate_rays()

        self.images = self._flatten(self.images)
        self.rays = namedtuple_map(self._flatten, self.rays)

    def _val_init(self):
        self._load_images()
        self._generate_rays()

    def _generate_rays(self):
        """Generating rays for all images."""
        raise ValueError('Implement in different dataset.')

    def _load_images(self):
        raise ValueError('Implement in different dataset.')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.split == 'val':
            index = (self.it + 1) % self.n_examples
            self.it += 1
        rays = Rays(*[getattr(self.rays, key)[index] for key in Ray_keys])
        return rays, self.images[index]

class BlenderDataset(BaseDataset):
    def __init__(self, filepath, split='train', white_bkgd=False, factor=0):
        super(BlenderDataset, self).__init__(filepath, split, white_bkgd, factor)
        if split == 'train':
            self._train_init()
        else:
            self._val_init()

    def _load_images(self):
        """Load images from disk."""
        with open(path.join(self.data_dir, 'transforms_{}.json'.format(self.split)), 'r') as fp:
            meta = json.load(fp)
        images = []
        cams = []
        for i in range(len(meta['frames'])):
            frame = meta['frames'][i]
            fname = path.join(self.data_dir, frame['file_path'] + '.png')
            with open(fname, 'rb') as imgin:
                image = np.array(Image.open(imgin), dtype=np.float32) / 255.
                if self.factor == 2:
                    [halfres_h, halfres_w] = [hw // 2 for hw in image.shape[:2]]
                    image = cv2.resize(image, (halfres_w, halfres_h), interpolation=cv2.INTER_AREA)
                elif self.factor > 0:
                    raise ValueError('Blender dataset only supports factor=0 or 2, {} '
                                     'set.'.format(self.factor))
            cams.append(np.array(frame['transform_matrix'], dtype=np.float32))
            if self.white_bkgd:
                image = image[..., :3] * image[..., -1:] + (1. - image[..., -1:])
            images.append(image[..., :3])

        self.images = images
        del images
        self.h, self.w = self.images[0].shape[:-1]
        self.camtoworlds = cams
        del cams
        camera_angle_x = float(meta['camera_angle_x'])
        self.focal = .5 * self.w / np.tan(.5 * camera_angle_x)
        self.n_examples = len(self.images)


    def _generate_rays(self):
        # Apply pinhole camera model to gather directions at each pixel
        x, y = torch.meshgrid(
            torch.arange(self.width, dtype=torch.float32),
            torch.arange(self.height, dtype=torch.float32),
            indexing='xy'
        )
        camera_dirs = torch.stack(
            [
                (x - self.width * 0.5) / self.focal,
                -(y - self.height * 0.5) / self.focal,
                -torch.ones_like(x)
            ],
            dim=-1
        )

        # Apply camera pose to directions
        rays_d = [(camera_dirs @ c2w[:3, :3].T).copy() for c2w in self.poses]
        rays_o = [
            np.broadcast_to(c2w[:3, -1], v.shape).copy()
            for v, c2w in zip(rays_d, self.poses)
        ]

        # Distance from each unit-norm direction vector to its x-axis neighbor.
        dx = [
            np.sqrt(np.sum((v[:-1, :, :] - v[1:, :, :]) ** 2, -1)) for v in rays_d
        ]
        dx = [np.concatenate([v, v[-2:-1, :]], 0) for v in dx]
        # Cut the distance in half, and then round it out so that it's
        # halfway between inscribed by / circumscribed about the pixel.
        radii = [v[..., None] * 2 / np.sqrt(12) for v in dx]
        self.rays = Rays(
            origins=rays_o,
            directions=rays_d,
            radii=radii,
            )



def load_dataset(file_path):
    ss = Settings()
    train_set = BlenderDataset(file_path, split='train')
    val_set = BlenderDataset(file_path, split='val')
    train_sampler = DistributedSampler(train_set)
    train_loader = DataLoader(train_set, shuffle=False, sampler=train_sampler, batch_size=ss.batch_size)
    return train_loader, train_sampler, train_set, val_set
