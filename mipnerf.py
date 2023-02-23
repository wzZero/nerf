from typing import Optional, Tuple, Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from settings import Settings
from encoder import PositionalEncoder, IntegratedPositionalEncoder
from sampling import stratified_sample, hierarchical_sample
from volumnRendering import raw2outputs


class NeRF(nn.Module):
    def __init__(self,
                 d_input: int,
                 n_layers: int,
                 d_filter: int,
                 skip: Tuple[int],
                 d_viewdirs: Optional[int] = None,
                 ):
        super().__init__()
        self.d_input = d_input
        self.skip = skip
        self.d_viewdirs = d_viewdirs
        self.activation = F.relu

        self.layers = nn.ModuleList(
            [nn.Linear(self.d_input, d_filter)]
        )
        for i in range(n_layers - 1):
            if i in self.skip:
                self.layers.append(nn.Linear(d_filter + self.d_input, d_filter))
            else:
                self.layers.append(nn.Linear(d_filter, d_filter))

        if self.d_viewdirs is not None:
            self.alpha_out = nn.Linear(d_filter, 1)
            self.rgb_filters = nn.Linear(d_filter, d_filter)
            self.branch = nn.Linear(d_filter + self.d_viewdirs, d_filter // 2)
            self.output = nn.Linear(d_filter // 2, 3)
        else:
            self.output = nn.Linear(d_filter, 4)

    def forward(self,
                x: torch.Tensor,
                viewdirs: Optional[torch.Tensor] = None
                ):
        if self.d_viewdirs is None and viewdirs is not None:
            raise ValueError("Cannot input x_direction if d_viewdirs was not given.")

        x_input = x
        for i, layer in enumerate(self.layers):
            x = self.activation(layer(x))
            if i in self.skip:
                x = torch.cat([x, x_input], dim=-1)

        if self.d_viewdirs is not None:
            alpha = self.alpha_out(x)
            x = self.rgb_filters(x)
            x = torch.cat([x, viewdirs], dim=-1)
            x = self.activation(self.branch(x))
            x = self.output(x)
            x = torch.cat([x, alpha], dim=-1)
        else:
            x = self.output(x)

        return x

#TODO: mipnerf forward process
class MipNeRFWrapper(nn.Module):
    def __init__(self, settings: Settings):
        super().__init__()

        # settings
        self.ss = settings

        # encoders
        encoder = IntegratedPositionalEncoder(self.ss.d_input, self.ss.n_freqs, log_space=self.ss.log_space)
        self.encoding_fn = lambda x: encoder(x)

        # view direction encoders
        if settings.use_viewdirs:
            encoder_viewdirs = PositionalEncoder(self.ss.d_input, self.ss.n_freqs_views, self.ss.log_space)
            self.viewdirs_encoding_fn = lambda x: encoder_viewdirs(x)
            d_viewdirs = encoder_viewdirs.d_output
        else:
            self.viewdirs_encoding_fn = None
            d_viewdirs = None

        # models
        self.coarse_model = NeRF(encoder.d_output, self.ss.n_layers, self.ss.d_filter, self.ss.skip, d_viewdirs)
        if self.ss.use_fine_model:
            self.fine_model = NeRF(encoder.d_output, self.ss.n_layers, self.ss.d_filter, self.ss.skip, d_viewdirs)
        else:
            self.fine_model = None

    def forward(self,rays):
        """
        :param rays: batch * (ray_o,ray_d,view_dirs)
        :return:
        """
        ret = []
        t_samples, weights = None, None

        # for i_level in range(2):
        #     if i_level == 0:
        #
        # Sample query points along each ray.
        query_points, z_vals = stratified_sample(rays_o, rays_d, self.ss.near, self.ss.far,
                                                 **self.ss.kwargs_sample_stratified)
        # Prepare batches.
        batches = self.prepare_chunks(query_points, self.encoding_fn, chunksize=self.ss.chunksize)
        if self.viewdirs_encoding_fn is not None:
            batches_viewdirs = self.prepare_viewdirs_chunks(query_points, rays_d, self.viewdirs_encoding_fn,
                                                            chunksize=self.ss.chunksize)
        else:
            batches_viewdirs = [None] * len(batches)

        # Coaese model pass.
        predictions = []
        for batch, batch_viewdirs in zip(batches, batches_viewdirs):
            predictions.append(self.coarse_model(batch, viewdirs=batch_viewdirs))
        raw = torch.cat(predictions, dim=0)
        raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

        # Perform differentiable volume rendering to re-synthesize the RGB image.
        rgb_map, depth_map, acc_map, weights = raw2outputs(raw, z_vals, rays_d)
        outputs = {'z_vals_stratified': z_vals}

        # Fine model pass.
        if self.ss.n_samples_hierarchical > 0:
            # Save previous outputs to return.
            rgb_map_0, depth_map_0, acc_map_0 = rgb_map, depth_map, acc_map

            # Apply hierarchical sampling for fine query points.
            query_points, z_vals_combined, z_hierarch = hierarchical_sample(
                rays_o, rays_d, z_vals, weights, self.ss.n_samples_hierarchical,
                **self.ss.kwargs_sample_hierarchical
            )

            # Prepare inputs as before.
            batches = self.prepare_chunks(query_points, self.encoding_fn, chunksize=self.ss.chunksize)
            if self.viewdirs_encoding_fn is not None:
                batches_viewdirs = self.prepare_viewdirs_chunks(query_points, rays_d, self.viewdirs_encoding_fn,
                                                                chunksize=self.ss.chunksize)
            else:
                batches_viewdirs = [None] * len(batches)

            # Forward pass new samples through fine model.
            if self.fine_model is None:
                self.fine_model = self.coarse_model

            predictions = []
            for batch, batch_viewdirs in zip(batches, batches_viewdirs):
                predictions.append(self.fine_model(batch, viewdirs=batch_viewdirs))
            raw = torch.cat(predictions, dim=0)
            raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

            # Perform differentiable volume rendering to re-synthesize the RGB image.
            rgb_map, depth_map, acc_map, weights = raw2outputs(raw, z_vals_combined, rays_d)

            # Store outputs.
            outputs['z_vals_hierarchical'] = z_hierarch
            outputs['rgb_map_0'] = rgb_map_0
            outputs['depth_map_0'] = depth_map_0
            outputs['acc_map_0'] = acc_map_0

        # Store outputs.
        outputs['rgb_map'] = rgb_map
        outputs['depth_map'] = depth_map
        outputs['acc_map'] = acc_map
        outputs['weights'] = weights
        return outputs

    def get_chunks(self,
                   inputs: torch.Tensor,
                   chunksize: int = 2 ** 15
                   ) -> List[torch.Tensor]:
        return [
            inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)
        ]

    def prepare_chunks(self,
                       points: torch.Tensor,
                       encoding_function: Callable[[torch.Tensor], torch.Tensor],
                       chunksize: int = 2 ** 15
                       ) -> List[torch.Tensor]:
        points = points.reshape((-1, 3))
        points = encoding_function(points)
        points = self.get_chunks(points, chunksize=chunksize)
        return points

    def prepare_viewdirs_chunks(self,
                                points: torch.Tensor,
                                rays_d: torch.Tensor,
                                encoding_function: Callable[[torch.Tensor], torch.Tensor],
                                chunksize: int = 2 ** 15
                                ) -> List[torch.Tensor]:
        # Prepare the viewdirs
        viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        viewdirs = viewdirs[:, None, ...].expand(points.shape).reshape((-1, 3))
        viewdirs = encoding_function(viewdirs)
        viewdirs = self.get_chunks(viewdirs, chunksize=chunksize)
        return viewdirs
