# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Latent SDE fit to a single time series with uncertainty quantification."""
import fire
import logging
import os
import random
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import magnify.data.drw_utils as drw_utils
from torch.utils.data import DataLoader
import magnify.script_utils as script_utils
from magnify.latent_sde.models import LatentSDE
import torchsde
import wandb


# w/ underscore -> numpy; w/o underscore -> torch.
Data = namedtuple('Data', ['ts_', 'ts_ext_', 'ts_vis_', 'ts', 'ts_ext', 'ts_vis', 'ys', 'ys_'])


class LinearScheduler(object):
    def __init__(self, iters, maxval=1.0):
        self._iters = max(1, iters)
        self._val = maxval / self._iters
        self._maxval = maxval

    def step(self):
        self._val = min(self._maxval, self._val + self._maxval / self._iters)

    @property
    def val(self):
        return self._val


class ExpScheduler(object):
    def __init__(self, iters, maxval=1e5):
        self._iters = max(1, iters)
        self._val = 1.0
        self._maxval = maxval

    def step(self):
        increased = np.log10(self._val) + np.log10(self._maxval) / self._iters
        increased = 10**increased
        self._val = min(self._maxval, increased)

    @property
    def val(self):
        return self._val


def manual_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(
        n_train=10000,
        n_val=1000,
        batch_size=100,
        latent_size=16,
        context_size=64,
        hidden_size=64,
        lr_init=1e-2,
        t0=0.,
        t1=4.,
        lr_gamma=0.999,
        num_iters=200,
        kl_anneal_iters=5000,
        pause_every=1,
        noise_std=0.1,
        adjoint=True,
        method="euler",
        show_prior=True,
        dpi=50,
        bandpasses=list('ugriz'),
        trim_single_band=False,
        param_weight=0,
        include_prior_drift=True,
        n_pointings=100,
        train_dir='train_ugriz',
        val_dir='val_ugriz',
        obs_dir='obs'
):
    wandb.init(project="agn-variability", entity="jiwoncpark")
    os.makedirs(train_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def t_transform(x):
        return (x - x.min())/3650.0*(t1-t0)  # + t0

    def y_transform(y):
        return y - 20.0

    train_dataset, val_dataset = drw_utils.get_drw_datasets(train_seed=123,
                                                            val_seed=456,
                                                            n_pointings=n_pointings,
                                                            bandpasses=bandpasses,
                                                            t_transform=t_transform,
                                                            y_transform=y_transform,
                                                            n_train=n_train,
                                                            n_val=n_val,
                                                            train_dir=train_dir,
                                                            val_dir=val_dir,
                                                            obs_dir=obs_dir)
    wandb.config = dict(n_train=len(train_dataset),
                        n_val=len(val_dataset),
                        batch_size=batch_size,
                        latent_size=latent_size,
                        context_size=context_size,
                        hidden_size=hidden_size,
                        lr_init=lr_init,
                        noise_std=noise_std,
                        adjoint=adjoint,
                        n_bandpasses=len(bandpasses),
                        lr_gamma=lr_gamma,
                        )

    output_dim = len(bandpasses)
    n_params = len(train_dataset.slice_params)

    train_loader = DataLoader(train_dataset, shuffle=True,
                              batch_size=batch_size)
    val_loader = DataLoader(val_dataset, shuffle=False,
                            batch_size=len(val_dataset))
    # Expected:
    # xs ~ [T, B, Y_out_dim]
    # ts ~ [T]
    # Current:
    # xs ~ [B, T, Y_out_dim]
    # ts ~ [T]
    latent_sde = LatentSDE(
        data_size=output_dim,  # output (Y) data dim
        latent_size=latent_size,
        context_size=context_size,
        hidden_size=hidden_size,
        n_params=n_params,
        include_prior_drift=include_prior_drift,
    ).to(device)
    wandb.watch(latent_sde)
    n_params = sum(p.numel() for p in latent_sde.parameters() if p.requires_grad)
    print(f"Number of params: {n_params}")
    optimizer = optim.Adam(params=latent_sde.parameters(), lr=lr_init)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=6, gamma=0.75)
    kl_scheduler = LinearScheduler(iters=kl_anneal_iters)
    n_updates = num_iters*len(train_dataset)//batch_size
    param_w_scheduler = LinearScheduler(iters=int(n_updates),
                                        maxval=param_weight)
    param_loss_fn = nn.MSELoss(reduction='mean')
    # Fix the same Brownian motion for visualization.
    vis_batch_size = len(val_dataset)
    ts_vis = t_transform(torch.arange(0, 3650+1, 1))  # tensor
    bm_vis = torchsde.BrownianInterval(
        t0=t0, t1=t1, size=(vis_batch_size, latent_size,), device=device,
        levy_area_approximation="space-time")

    alphas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    percentiles = [0.999, 0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    # vis_idx = np.random.permutation(vis_batch_size)
    # sample_colors = ('#8c96c6', '#8c6bb1', '#810f7c')
    fill_color = '#9ebcda'
    mean_color = '#4d004b'
    # num_samples = len(sample_colors)
    ylims = (-1.2, 1.2)
    if show_prior:
        with torch.no_grad():
            for batch in val_loader:
                # Last example in batch
                ys_val = batch['y'][[-1], :, :]  # [1, T, Y_out_dim]
                ys_val = ys_val.transpose(0, 1)  # [T, 1, Y_out_dim]
                ts = batch['x'][-1, :]  # [T]
                if trim_single_band:
                    trimmed_mask = batch['trimmed_mask'][-1, :, 0]  # [T,]
                    ys_val = ys_val[trimmed_mask, :, :]
                    ts = ts[trimmed_mask]
                break
            zs = latent_sde.sample(ts=ts_vis,
                                   batch_size=vis_batch_size,
                                   bm=bm_vis)  # [T_vis, batch_size, out_dim]
            ts_vis_, zs_ = ts_vis.cpu().numpy(), zs.cpu().numpy()
            zs_ = np.sort(zs_, axis=1)  # sort along batch axis

            img_dir = os.path.join(train_dir, 'prior.png')
            plt.subplot(frameon=False)

            # `zorder` determines who's on top; the larger the more at the top.
            # Plot data
            for band_i in range(output_dim):
                for alpha, percentile in zip(alphas, percentiles):
                    idx = int((1 - percentile) / 2. * vis_batch_size)
                    zs_bot_ = zs_[:, idx, band_i]  # band_i=0 : g-band
                    zs_top_ = zs_[:, -idx, band_i]
                    plt.fill_between(ts_vis_,
                                     zs_bot_,
                                     zs_top_,
                                     alpha=alpha, color=fill_color)
                # print(ts.shape, ys_val.shape)
                plt.scatter(ts.cpu().numpy().squeeze(),
                            ys_val.cpu().numpy()[:, 0, band_i].squeeze(),
                            marker='x', zorder=3, color='k', s=35)  # last in batch
            plt.ylim(ylims)
            plt.xlabel('$t$')
            plt.ylabel('$Y_t$')
            plt.tight_layout()
            plt.savefig(img_dir, dpi=dpi)
            plt.close()
            logging.info(f'Saved prior figure at: {img_dir}')

    last_val_loss = float('inf')  # init
    n_batches = len(train_loader)
    for global_step in tqdm(range(num_iters)):
        latent_sde.train()
        for i, batch in enumerate(train_loader):
            ys_batch = batch['y'].transpose(0, 1)  # [T, B, out_dim]
            ts = batch['x'][-1, :]  # [T]
            param_labels = batch['params'].float().to(device)  # [B, n_params]
            if trim_single_band:
                trimmed_mask = batch['trimmed_mask'][-1, :, 0]  # [T,]
                ys_batch = ys_batch[trimmed_mask, :, :]  # [trimmed_T, B, out_dim]
                ts = ts[trimmed_mask]
            latent_sde.zero_grad()
            log_pxs, log_ratio, param_pred = latent_sde(ys_batch.to(device),
                                                        ts.to(device),
                                                        noise_std, adjoint,
                                                        method)
            recon_loss = -log_pxs + log_ratio * kl_scheduler.val
            param_loss = param_loss_fn(param_pred, param_labels)
            loss = recon_loss + param_weight*param_loss
            log_step = global_step*n_batches+i
            wandb.log(dict(loss=loss.detach().cpu().item(),
                           neg_los_pxs=(-log_pxs).detach().cpu().item(),
                           kl_term=(log_ratio*kl_scheduler.val).detach().cpu().item(),
                           param_loss=(param_loss).detach().cpu().item(),
                           learning_rate=scheduler.optimizer.param_groups[0]['lr']
                           ),
                      log_step
                      )
            loss.backward()
            optimizer.step()
            kl_scheduler.step()
            param_w_scheduler.step()

        latent_sde.eval()
        if global_step % pause_every == 0:
            lr_now = optimizer.param_groups[0]['lr']
            logging.warning(
                f'global_step: {global_step:06d}, lr: {lr_now:.5f}, '
                f'log_pxs: {log_pxs:.4f}, log_ratio: {log_ratio:.4f} '
                f'loss: {loss:.4f}, kl_coeff: {kl_scheduler.val:.4f}'
            )
            # Visualization during training
            with torch.no_grad():
                for i, val_batch in enumerate(val_loader):
                    # Note there's only one batch
                    ys_batch = val_batch['y'].transpose(0, 1)  # [T, B, Y_out_dim]
                    ts = val_batch['x'][-1, :]  # [T]
                    param_labels = val_batch['params'].float().to(device)  # [B, n_params]
                    if trim_single_band:
                        trimmed_mask = val_batch['trimmed_mask'][-1, :, 0]  # [T,]
                        ys_batch = ys_batch[trimmed_mask, :, :]
                        ts = ts[trimmed_mask]
                    log_pxs, log_ratio, param_pred = latent_sde(ys_batch.to(device),
                                                                ts.to(device),
                                                                noise_std, adjoint,
                                                                method)
                    recon_loss = -log_pxs + log_ratio * kl_scheduler.val
                    param_loss = param_loss_fn(param_pred, param_labels)
                    val_loss = recon_loss + param_weight*param_loss
                    unw_val_loss = (recon_loss + param_weight*param_loss).detach().cpu().item()
                    scheduler.step()
                    wandb.log(dict(val_loss=val_loss.detach().cpu().item(),
                                   val_neg_los_pxs=(-log_pxs).detach().cpu().item(),
                                   val_kl_term=(log_ratio*kl_scheduler.val).detach().cpu().item(),
                                   val_param_loss=(param_loss).detach().cpu().item(),
                                   ),
                              log_step
                              )
                # Last example in batch
                trimmed_mask = val_batch['trimmed_mask'][-1, :, 0]  # [T,]
                ys_val = val_batch['y'][[-1], :, :]  # [1, T, Y_out_dim]
                ys_val = ys_val.transpose(0, 1)  # [T, 1, Y_out_dim]
                ts = val_batch['x'][-1, :]  # [T]
                if trim_single_band:
                    ys_val = ys_val[trimmed_mask, :, :]
                    ts = ts[trimmed_mask]  # [T]
                sample = latent_sde.sample_posterior(vis_batch_size,
                                                     ys_val.to(device),
                                                     ts.to(device),
                                                     ts_vis.to(device),
                                                     bm=None,
                                                     method='euler')
                # zs ~ [T_vis=3651, B]
                sample_ = sample.cpu().numpy()  # [T_vis, 1, n_bandpasses]
                # ts_vis_ ~ [T_vis]
                # sample_fp ~ [T_vis]
                fig, ax = plt.subplots()
                for band_i in range(output_dim):
                    # Plot sample for last light curve in batch
                    ax.plot(ts_vis.cpu().numpy(), sample_[:, 0, band_i],  # last in batch
                            color=mean_color)
                    # Plot data for last light curve in batch
                    # ys_val ~ [T_vis, 1, n_bandpasses]
                    ax.scatter(ts.cpu().numpy(),
                               ys_val.cpu().numpy()[:, 0, band_i].squeeze(),  # last in batch
                               marker='x', zorder=3, color='k', s=35)
                ax.set_ylim(ylims)
                ax.set_xlabel('$t$')
                ax.set_ylabel('$Y_t$')
                fig.tight_layout()
                wandb.log({"recovery": wandb.Image(fig)}, log_step)

        if (global_step+1)%10 == 0:
            script_utils.save_state(latent_sde, optimizer, scheduler,
                                    kl_scheduler, global_step, train_dir,
                                    param_w_scheduler, global_step)
        elif unw_val_loss < last_val_loss:
            script_utils.save_state(latent_sde, optimizer, scheduler,
                                    kl_scheduler, global_step, train_dir,
                                    param_w_scheduler, global_step)
            last_val_loss = unw_val_loss
        else:
            pass


if __name__ == '__main__':
    fire.Fire(main)
