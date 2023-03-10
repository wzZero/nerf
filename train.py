import os
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
from typing import Optional
from matplotlib import pyplot as plt

from utils import rearrange_render_image
from torch.utils.tensorboard import SummaryWriter

from mipnerf import MipNeRFWrapper
from settings import Settings
from dataset import load_dataset


# TODO render image
def render(model, settings, dataloader):
    rays, image = next(dataloader)
    N, h, w, _ = image.shape
    single_image_rays, val_mask = rearrange_render_image(
        rays, settings.val_chunk_size)
    coarse_rgb = []
    fine_rgb = []
    # distances = []
    with torch.no_grad():
        for batch_rays in single_image_rays:
            outputs = model(batch_rays)
            coarse_rgb.append(outputs['rgb_map_0'])
            fine_rgb.append(outputs['rgb_map'])
            # distances.append(outputs['depth_map'])

    coarse_rgb = torch.cat(coarse_rgb, dim=0)
    fine_rgb = torch.cat(fine_rgb, dim=0)

    coarse_rgb = coarse_rgb.reshape(
        N, h, w, coarse_rgb.shape[-1])  # N H W C
    fine_rgb = fine_rgb.reshape(
        N, h, w, fine_rgb.shape[-1])
    return coarse_rgb, fine_rgb, image

    # def plot_samples(
    #     z_vals: torch.Tensor,
    #     z_hierarch: Optional[torch.Tensor] = None,
    #     ax: Optional[np.ndarray] = None
    # ):
    #     """
    #     Plot stratified and (optional) hierarchical samples.
    #     """
    #     y_vals = 1 + np.zeros_like(z_vals)
    #     if ax is None:
    #         ax = plt.subplot()
    #     ax.plot(z_vals, y_vals, 'b-o')
    #     if z_hierarch is not None:
    #         y_hierarch = np.zeros_like(z_hierarch)
    #         ax.plot(z_hierarch, y_hierarch, 'r-o')
    #     ax.set_ylim([-1, 2])
    #     ax.set_title('Stratified  Samples (blue) and Hierarchical Samples (red)')
    #     ax.axes.yaxis.set_visible(False)
    #     ax.grid(True)
    #     return ax
    #
    # fig, ax = plt.subplots(1, 4, figsize=(24, 4), gridspec_kw={'width_ratios': [1, 1, 1, 3]})
    # ax[0].imshow(outputs["rgb_map"].reshape([dataset.h, dataset.w, 3]).detach().cpu().numpy())
    # ax[0].set_title(f'Iteration: {iter + 1}')
    # ax[1].imshow(dataset.images[settings.val_id].detach().cpu().numpy())
    # ax[1].set_title(f'Target')
    # ax[2].plot(range(0, iter + 1), train_psnrs, 'r')
    # ax[2].plot(iternums, val_psnrs, 'b')
    # ax[2].set_title('PSNR (train=red, val=blue')
    # z_vals_strat = outputs['z_vals_stratified'].view((-1, settings.n_samples))
    # z_sample_strat = z_vals_strat[z_vals_strat.shape[0] // 2].detach().cpu().numpy()
    # if 'z_vals_hierarchical' in outputs:
    #     z_vals_hierarch = outputs['z_vals_hierarchical'].view(
    #         (-1, settings.n_samples_hierarchical))
    #     z_sample_hierarch = z_vals_hierarch[z_vals_hierarch.shape[0] // 2].detach().cpu().numpy()
    # else:
    #     z_sample_hierarch = None
    # plot_samples(z_sample_strat, z_sample_hierarch, ax=ax[3])
    # ax[3].margins(0)
    # plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=os.getenv("LOCAL_RANK", -1), type=int)
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    device = torch.device("cuda", args.local_rank)

    settings = Settings()
    writer = SummaryWriter(settings.log_dir)
    model = MipNeRFWrapper(settings)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.lr)
    train_loader, train_sampler, train_set, val_set, val_loader = load_dataset(settings, device)
    lr_sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2500, 5000, 7500, 10000])

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=[args.local_rank])
    else:
        model = model.to(device)

    train_psnrs = []
    val_psnrs = []
    iternums = []

    # with tqdm(total=settings.n_iters, position=args.local_rank * 2) as pbar:
    best_loss = 99999
    for i in range(settings.n_iters):
        model.train()
        train_sampler.set_epoch(i)
        i_loss = 0
        n_batch = 0

        for _,(rays, rgb) in enumerate(tqdm(train_loader)):
            rgb = rgb.to(device)
            outputs = model(rays)
            # Check for any numerical issues.
            for k, v in outputs.items():
                if torch.isnan(v).any():
                    print(f"! [Numerical Alert] {k} contains NaN.")
                if torch.isinf(v).any():
                    print(f"! [Numerical Alert] {k} contains Inf.")

            loss = F.mse_loss(outputs["rgb_map"], rgb)
            i_loss += loss.item()
            n_batch += 1
            # pbar.set_description("GPU {} Loss: {:.4f}".format(args.local_rank, loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_sched.step()

            # psnr = -10.0 * torch.log10(loss)
            # train_psnrs.append(psnr.item())

        writer.add_scalar('loss', i_loss/n_batch, i + 1) # i_loss / n_batch

        # Check PSNR for issues and stop if any are found.
        # if i == settings.warmup_iters - 1:
        #     if val_psnr < settings.warmup_min_fitness:
        #         print(
        #             f'Val PSNR {val_psnr} below warmup_min_fitness {settings.warmup_min_fitness}. Stopping...'
        #         )
        #         return False, train_psnrs, val_psnrs
        # elif i < settings.warmup_iters:
        #     if warmup_stopper is not None and warmup_stopper(i, psnr):
        #         print(
        #             f'Train PSNR flatlined at {psnr} for {warmup_stopper.patience} iters. Stopping...'
        #         )
        #         return False, train_psnrs, val_psnrs

        if (i + 1) % settings.display_rate == 0:
            model.eval()
            coarse, fine, val = render(model, settings, val_loader)
            writer.add_image('coarse', coarse, i+1)
            writer.add_image('fine', fine, i+1)
            writer.add_image('val', val, i+1)

        if i_loss/n_batch < best_loss:
            best_loss = i_loss/n_batch
            if not os.path.exists('./ckpt'):
                os.makedirs('./ckpt')
            torch.save(model.state_dict(), './ckpt/ckpt_mipnerf_loss_%.6f.t7' % best_loss)
            torch.distributed.barrier()
        # pbar.update(1)


main()
