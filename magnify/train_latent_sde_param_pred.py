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
import argparse
import logging
import math
import os
import random
from collections import namedtuple
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from torch import distributions, nn, optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import magnify.data.drw_utils as drw_utils
from torch.utils.data import DataLoader
import torchsde

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


class EMAMetric(object):
    def __init__(self, gamma: Optional[float] = .99):
        super(EMAMetric, self).__init__()
        self._val = 0.
        self._gamma = gamma

    def step(self, x: Union[torch.Tensor, np.ndarray]):
        x = x.detach().cpu().numpy() if torch.is_tensor(x) else x
        self._val = self._gamma * self._val + (1 - self._gamma) * x
        return self._val

    @property
    def val(self):
        return self._val


def str2bool(v):
    """Used for boolean arguments in argparse; avoiding `store_true` and `store_false`."""
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')


def manual_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _stable_division(a, b, epsilon=1e-7):
    b = torch.where(b.abs().detach() > epsilon, b, torch.full_like(b, fill_value=epsilon) * b.sign())
    return a / b


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, inp):
        out, _ = self.gru(inp)
        out = self.lin(out)
        return out


class ResMLP(nn.Module):
    """Residual MLP that predicts the target quantities

    Attributes
    ----------
    dim_in : int
        Dimension of stochastic latent vector
    dim_out : int
        Number of target quantities to predict, or number of parameters
        defining the posterior PDF over the target quantities
    """
    def __init__(self, dim_in, dim_out, dim_hidden=16):
        super(ResMLP, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.pre_skip = nn.Sequential(nn.Linear(self.dim_in, self.dim_hidden),
                                      nn.ReLU(),
                                      nn.Linear(self.dim_hidden, self.dim_hidden),
                                      nn.ReLU(),
                                      nn.Linear(self.dim_hidden, self.dim_in),
                                      nn.LayerNorm(self.dim_in))
        self.post_skip = nn.Sequential(nn.Linear(self.dim_in, self.dim_hidden),
                                       nn.ReLU(),
                                       nn.Linear(self.dim_hidden, self.dim_out))

    def forward(self, z0):
        # z0 ~ [B, dim_in]
        out = self.pre_skip(z0)  # [B, dim_in]
        out = out + z0  # [B, dim_in], skip connection
        out = self.post_skip(out)  # projector
        return out


class LatentSDE(nn.Module):
    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(self, data_size, latent_size, context_size, hidden_size,
                 n_params):
        super(LatentSDE, self).__init__()
        # Encoder.
        self.encoder = Encoder(input_size=data_size, hidden_size=hidden_size,
                               output_size=context_size)
        self.qz0_net = nn.Linear(context_size, latent_size + latent_size)

        # Decoder.
        self.f_net = nn.Sequential(
            nn.Linear(latent_size + context_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )
        self.h_net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )
        # This needs to be an element-wise function for the SDE to satisfy diagonal noise.
        self.g_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, hidden_size),
                    nn.Softplus(),
                    nn.Linear(hidden_size, 1),
                    nn.Sigmoid()
                )
                for _ in range(latent_size)
            ]
        )
        self.projector = nn.Linear(latent_size, data_size)

        self.param_mlp = ResMLP(latent_size, n_params, hidden_size)

        self.pz0_mean = nn.Parameter(torch.zeros(1, latent_size))
        self.pz0_logstd = nn.Parameter(torch.zeros(1, latent_size))

        self._ctx = None

    def contextualize(self, ctx):
        """Set the context vector, which is an encoding of the observed
        sequence

        Parameters
        ----------
        ctx : tuple
            A tuple of tensors of sizes (T,), (T, batch_size, d)
        """
        self._ctx = ctx

    def f(self, t, y):
        """Network that decodes the latent and context
        (posterior drift function)
        Time-inhomogeneous (see paper sec 9.12)

        """
        ts, ctx = self._ctx
        # ts ~ [T]
        # ctx ~ [T, B, context_size]
        ts = ts.to(t.device)
        # searchsorted output: if t were inserted into ts, what would the
        # indices have to be to preserve ordering, assuming ts is sorted
        # training time: t is tensor with no size (scalar)
        # inference time: t ~ [num**2, 1] from the meshgrid
        i = min(torch.searchsorted(ts, t.min(), right=True), len(ts) - 1)
        # training time: y ~ [B, latent_dim]
        # inference time: y ~ [num**2, 1] from the meshgrid
        return self.f_net(torch.cat((y, ctx[i]), dim=1))

    def h(self, t, y):
        """Network that decodes the latent
        (prior drift function)

        """
        return self.h_net(y)

    def g(self, t, y):
        """Network that decodes each time step of the latent
        (diagonal diffusion)

        """
        y = torch.split(y, split_size_or_sections=1, dim=1)
        out = [g_net_i(y_i) for (g_net_i, y_i) in zip(self.g_nets, y)]
        return torch.cat(out, dim=1)

    def forward(self, xs, ts, noise_std, adjoint=False, method="euler"):
        # Contextualization is only needed for posterior inference.
        ctx = self.encoder(torch.flip(xs, dims=(0,)))  # [T, B, context_size] = [100, 1024, 64]
        ctx = torch.flip(ctx, dims=(0,))  # revert to original time sequence
        self.contextualize((ts, ctx))

        qz0_mean, qz0_logstd = self.qz0_net(ctx[0]).chunk(chunks=2, dim=1)
        z0 = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)
        # z0 ~ [B, latent_dim] = [1024, 4]

        if adjoint:
            # Must use the argument `adjoint_params`, since `ctx` is not part of the input to `f`, `g`, and `h`.
            adjoint_params = (
                    (ctx,) +
                    tuple(self.f_net.parameters()) + tuple(self.g_nets.parameters()) + tuple(self.h_net.parameters())
            )
            zs, log_ratio = torchsde.sdeint_adjoint(
                self, z0, ts, adjoint_params=adjoint_params, dt=1e-2, logqp=True, method=method)
            # zs ~ [T, B, latent_dim] = [100, 1024, 4]
            # log_ratio ~ [T-1, B] = [99, 1024]
        else:
            zs, log_ratio = torchsde.sdeint(self, z0, ts, dt=1e-2, logqp=True, method=method)

        _xs = self.projector(zs)
        # _xs ~ [T, B, Y_out_dim] = [100, 1024, 3]
        xs_dist = Normal(loc=_xs, scale=noise_std)
        # Sum across times and dimensions, mean across examples in batch
        log_pxs = xs_dist.log_prob(xs).sum(dim=(0, 2)).mean(dim=0)  # scalar

        qz0 = torch.distributions.Normal(loc=qz0_mean, scale=qz0_logstd.exp())
        pz0 = torch.distributions.Normal(loc=self.pz0_mean, scale=self.pz0_logstd.exp())
        # Sum over times, mean over batches
        logqp0 = torch.distributions.kl_divergence(qz0, pz0).sum(dim=0).mean(dim=0)
        logqp_path = log_ratio.sum(dim=0).mean(dim=0)

        # Parameter predictions
        param_pred = self.param_mlp(z0)
        return log_pxs, logqp0 + logqp_path, param_pred

    @torch.no_grad()
    def sample_posterior(self, batch_size, xs, ts, ts_eval,
                         bm=None, method='euler'):
        # Contextualization is only needed for posterior inference.
        ctx = self.encoder(torch.flip(xs, dims=(0,)))  # [T, B, context_size] = [100, 1024, 64]
        ctx = torch.flip(ctx, dims=(0,))  # revert to original time sequence
        self.contextualize((ts, ctx))

        qz0_mean, qz0_logstd = self.qz0_net(ctx[0]).chunk(chunks=2, dim=1)
        z0 = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)
        # z0 ~ [B, latent_dim] = [1024, 4]
        zs, log_ratio = torchsde.sdeint(self, z0, ts_eval, dt=1e-2,
                                        logqp=True, method=method, bm=bm)
        _xs = self.projector(zs)
        # _xs ~ [T, B, Y_out_dim] = [100, 1024, 3]
        return _xs

    @torch.no_grad()
    def sample(self, batch_size, ts, bm=None):
        eps = torch.randn(size=(batch_size, *self.pz0_mean.shape[1:]),
                          device=self.pz0_mean.device)
        z0 = self.pz0_mean + self.pz0_logstd.exp() * eps
        zs = torchsde.sdeint(self, z0, ts, names={'drift': 'h'}, dt=1e-3, bm=bm)
        # Most of the times in ML, we don't sample the observation noise
        # for visualization purposes.
        _xs = self.projector(zs)
        return _xs


def save_state(model, optim, lr_scheduler, kl_scheduler, epoch,
               train_dir):
    """Save the state dict of the current training to disk
    Parameters
    ----------
    train_loss : float
        current training loss
    val_loss : float
        current validation loss
    """
    state = dict(
             model=model.state_dict(),
             optimizer=optim.state_dict(),
             lr_scheduler=lr_scheduler.state_dict(),
             kl_scheduler=kl_scheduler.__dict__,
             epoch=epoch,
             )
    model_path = os.path.join(train_dir, 'model.mdl')
    torch.save(state, model_path)


def main(
        n_train=128*100,
        batch_size=128,
        latent_size=16,
        context_size=128,
        hidden_size=128,
        lr_init=1e-2,
        t0=0.,
        t1=4.,
        lr_gamma=0.999,
        num_iters=150,
        kl_anneal_iters=1000,
        pause_every=1,
        noise_std=0.01,
        adjoint=True,
        train_dir='./dump/gr_no_mask_param/',
        method="euler",
        show_prior=True,
        dpi=50,
        bandpasses=['g', 'r'],
        trim_single_band=False,
        param_weight=1e5,
):
    os.makedirs(train_dir, exist_ok=True)
    logger = SummaryWriter(train_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def t_transform(x):
        return x/3650.0*t1 + t0

    def y_transform(y):
        return y - 25.0

    train_data_dir = '/home/jwp/stage/sl/magnify/latent_ode_data/train_drw_gr'
    val_data_dir = '/home/jwp/stage/sl/magnify/latent_ode_data/val_drw_gr'
    train_dataset, val_dataset = drw_utils.get_drw_datasets(train_seed=123,
                                                            val_seed=456,
                                                            n_pointings=1,
                                                            bandpasses=bandpasses,
                                                            t_transform=t_transform,
                                                            y_transform=y_transform,
                                                            train_dir=train_data_dir,
                                                            val_dir=val_data_dir)

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
    ).to(device)
    n_params = sum(p.numel() for p in latent_sde.parameters() if p.requires_grad)
    print(f"Number of params: {n_params}")
    optimizer = optim.Adam(params=latent_sde.parameters(), lr=lr_init)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                           factor=0.5,
                                                           patience=100)
    kl_scheduler = LinearScheduler(iters=kl_anneal_iters)
    param_loss_fn = nn.MSELoss(reduction='mean')
    # Fix the same Brownian motion for visualization.
    vis_batch_size = len(val_dataset)
    ts_vis = t_transform(torch.arange(0, 3650+1, 1))  # tensor
    bm_vis = torchsde.BrownianInterval(
        t0=t0, t1=t1, size=(vis_batch_size, latent_size,), device=device,
        levy_area_approximation="space-time")

    alphas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    percentiles = [0.999, 0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    vis_idx = np.random.permutation(vis_batch_size)
    sample_colors = ('#8c96c6', '#8c6bb1', '#810f7c')
    fill_color = '#9ebcda'
    mean_color = '#4d004b'
    num_samples = len(sample_colors)
    ylims = (-3.1, 3.1)
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
                                   bm=bm_vis).squeeze()
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

                plt.scatter(ts.cpu().numpy().squeeze(),
                            ys_val.cpu().numpy().squeeze()[:, band_i],
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
            ys_batch = batch['y'].transpose(0, 1)  # [T, B, Y_out_dim]
            ts = batch['x'][-1, :]  # [T]
            param_labels = batch['params'].float().to(device)  # [B, n_params]
            if trim_single_band:
                trimmed_mask = batch['trimmed_mask'][-1, :, 0]  # [T,]
                ys_batch = ys_batch[trimmed_mask, :, :]
                ts = ts[trimmed_mask]
            latent_sde.zero_grad()
            log_pxs, log_ratio, param_pred = latent_sde(ys_batch.to(device),
                                                        ts.to(device),
                                                        noise_std, adjoint,
                                                        method)
            recon_loss = -log_pxs + log_ratio * kl_scheduler.val
            param_loss = param_loss_fn(param_pred, param_labels)
            loss = recon_loss + param_weight*param_loss
            logger.add_scalar('loss',
                              loss.detach().cpu().item(),
                              global_step*n_batches+i)
            logger.add_scalar('neg_log_pxs',
                              (-log_pxs).detach().cpu().item(),
                              global_step*n_batches+i)
            logger.add_scalar('kl_term',
                              (log_ratio*kl_scheduler.val).detach().cpu().item(),
                              global_step*n_batches+i)
            logger.add_scalar('param_loss',
                              (param_loss).detach().cpu().item(),
                              global_step*n_batches+i)
            logger.add_scalar('learning_rate',
                              scheduler.optimizer.param_groups[0]['lr'],
                              global_step*n_batches+i)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            kl_scheduler.step()

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
                    logger.add_scalar('val_loss',
                                      val_loss.detach().cpu().item(),
                                      global_step)
                    logger.add_scalar('val_neg_log_pxs',
                                      (-log_pxs).detach().cpu().item(),
                                      global_step)
                    logger.add_scalar('val_kl_term',
                                      (log_ratio*kl_scheduler.val).detach().cpu().item(),
                                      global_step)
                    logger.add_scalar('val_param_loss',
                                      (param_loss).detach().cpu().item(),
                                      global_step)
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
                                                     method='euler').squeeze()
                # zs ~ [T_vis=3651, B]
                sample_ = sample.cpu().numpy()
                # ts_vis_ ~ [T_vis]
                # sample_ ~ [T_vis]
                fig, ax = plt.subplots()
                for band_i in range(output_dim):
                    # Plot sample for last light curve in batch
                    ax.plot(ts_vis.cpu().numpy(), sample_[:, band_i],  # last in batch
                            color=mean_color)
                    # Plot data for last light curve in batch
                    ax.scatter(ts.cpu().numpy(),
                               ys_val.cpu().numpy().squeeze()[:, band_i],  # last in batch
                               marker='x', zorder=3, color='k', s=35)
                ax.set_ylim(ylims)
                ax.set_xlabel('$t$')
                ax.set_ylabel('$Y_t$')
                fig.tight_layout()
                logger.add_figure('recovery', fig, global_step=global_step)

        if val_loss < last_val_loss:
            save_state(latent_sde, optimizer, scheduler, kl_scheduler,
                       global_step,
                       train_dir)
            last_val_loss = val_loss


if __name__ == '__main__':
    fire.Fire(main)
