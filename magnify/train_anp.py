import os
from functools import partial
import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from tensorboardX import SummaryWriter
import torch
from magnificat.drw_dataset import DRWDataset
from magnificat.cadence import LSSTCadence
from torch.utils.data import DataLoader
from magnify.attentive_neural_process.network import NeuralProcess
from magnify.attentive_neural_process.context_target_sampler import collate_fn_opsim


class Sampler:
    def __init__(self, seed, bandpasses):
        random.seed(seed)
        np.random.seed(seed)
        self.bandpasses = bandpasses

    def sample(self):
        sample_dict = dict()
        for bp in self.bandpasses:
            SF_inf = np.maximum(np.random.randn()*0.05 + 0.2, 0.001)
            # SF_inf = 10**(np.random.randn(N)*(0.25) + -0.8)
            # SF_inf = np.ones(N)*0.15
            # tau = 10.0**np.maximum(np.random.randn(N)*0.5 + 2.0, 0.1)
            tau = np.maximum(np.random.randn()*50.0 + 200.0, 10.0)
            # mag = np.maximum(np.random.randn(N) + 19.0, 17.5)
            mag = 0.0
            # z = np.maximum(np.random.randn(N) + 2.0, 0.5)
            sample_dict[f'tau_{bp}'] = tau
            sample_dict[f'SF_inf_{bp}'] = SF_inf
            sample_dict[f'mag_{bp}'] = mag
        sample_dict['redshift'] = 2.0
        sample_dict['M_i'] = -25.0
        sample_dict['BH_mass'] = 10.0
        return sample_dict


def train(run_dir):
    torch.cuda.empty_cache()
    os.makedirs(run_dir, exist_ok=True)
    train_seed = 123
    train_dataset = DRWDataset(Sampler(train_seed, ['i']), 'train_drw',
                               num_samples=10000,
                               seed=train_seed,
                               shift_x=-3650*0.5,
                               rescale_x=1.0/(3650*0.5)*4.0,
                               delta_x=1.0,
                               max_x=3650.0,
                               err_y=0.001)
    train_dataset.slice_params = [train_dataset.param_names.index(n) for n in ['tau_i', 'SF_inf_i']]
    print(train_dataset.slice_params)
    print(train_dataset.mean_params, train_dataset.std_params)
    # Generate pointings
    cadence_obj = LSSTCadence('obs', train_seed)
    n_pointings = 100
    ra, dec = cadence_obj.get_pointings(n_pointings)
    cadence_obj.get_obs_info(ra, dec)
    cadence_obj.set_bandpasses(['i'])
    # Define collate fn that samples context points based on pointings
    collate_fn = partial(collate_fn_opsim,
                         rng=np.random.default_rng(train_seed),
                         cadence_obj=cadence_obj,
                         n_pointings=n_pointings,
                         every_other=10,
                         exclude_ddf=True)
    train_loader = DataLoader(train_dataset, batch_size=50, collate_fn=collate_fn,
                              shuffle=True)
    # Validation data
    val_seed = 456
    val_dataset = DRWDataset(Sampler(val_seed, ['i']), 'train_drw',
                             num_samples=20,
                             seed=train_seed,
                             shift_x=-3650*0.5,
                             rescale_x=1.0/(3650*0.5)*4.0,
                             delta_x=1.0,
                             max_x=3650.0,
                             err_y=0.001)
    val_dataset.slice_params = train_dataset.slice_params
    val_dataset.mean_params = train_dataset.mean_params
    val_dataset.std_params = train_dataset.std_params
    collate_fn = partial(collate_fn_opsim,
                         rng=np.random.default_rng(val_seed),
                         cadence_obj=cadence_obj,
                         n_pointings=n_pointings,
                         every_other=10)
    val_loader = DataLoader(val_dataset, batch_size=20, collate_fn=collate_fn,
                            shuffle=True)
    epochs = 250
    model = NeuralProcess(hidden_dim=64, latent_dim=32, weight_y_loss=0.1,
                          n_target=len(train_dataset.slice_params),
                          batchnorm=True).cuda()
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5,
                                                           patience=20, verbose=True)
    min_val_loss = np.inf
    writer = SummaryWriter(run_dir)
    for epoch in tqdm(range(epochs)):
        train_single_epoch(model, train_loader, optim, epoch, writer)
        val_loss = eval(model, val_loader, optim, epoch, writer)
        scheduler.step(val_loss)
        # Save model if validation loss decreased
        if val_loss < min_val_loss:
            torch.save({'model': model.state_dict(),
                        'optimizer': optim.state_dict(),
                        'scheduler': scheduler.state_dict()},
                       os.path.join(run_dir, 'checkpoint.pth.tar'))
            min_val_loss = val_loss


def train_single_epoch(model, train_loader, optim, epoch, writer):
    total_loss, kl_loss, mse_loss, meta_loss = 0.0, 0.0, 0.0, 0.0
    model.train()
    for i, data in enumerate(train_loader):
        optim.zero_grad()
        context_x, context_y, target_x, target_y, meta = data
        context_x = context_x.cuda()
        context_y = context_y.cuda()
        target_x = target_x.cuda()
        target_y = target_y.cuda()
        meta = meta.cuda()
        # pass through the latent model
        y_pred, losses, extra = model(context_x, context_y, target_x, target_y, meta)
        loss = losses['loss']
        # Training step
        loss.backward()
        optim.step()
        # Logging
        total_loss += (loss - total_loss)/(i+1)
        kl_loss += (losses['loss_kl'] - kl_loss)/(i+1)
        mse_loss += (losses['loss_mse'] - mse_loss)/(i+1)
        meta_loss += (losses['loss_meta'] - meta_loss)/(i+1)
    writer.add_scalars('training_loss',
                       {'loss': total_loss, 'kl': kl_loss,
                        'mse': losses['loss_mse'],
                        'meta': losses['loss_meta']},
                       epoch)


def eval(model, val_loader, optim, epoch, writer):
    total_loss, kl_loss, mse_loss, meta_loss = 0.0, 0.0, 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            context_x, context_y, target_x, target_y, meta = data
            context_x = context_x.cuda()
            context_y = context_y.cuda()
            target_x = target_x.cuda()
            target_y = target_y.cuda()
            meta = meta.cuda()
            # pass through the latent model
            pred_y, losses, extra = model(context_x, context_y, target_x, target_y, meta)
            loss = losses['loss']
            # Logging
            total_loss += (loss - total_loss)/(i+1)
            kl_loss += (losses['loss_kl'] - kl_loss)/(i+1)
            mse_loss += (losses['loss_mse'] - mse_loss)/(i+1)
            meta_loss += (losses['loss_meta'] - meta_loss)/(i+1)
        for p in range(meta.shape[1]):
            fig = get_params_fig(extra['mean_meta'].cpu().numpy()[:, p],
                                 extra['log_sigma_meta'].cpu().numpy()[:, p],
                                 meta.cpu().numpy()[:, p])
            writer.add_figure(f'param {p} recovery', fig, global_step=epoch)
        # Visualize fit on first light curve
        pred_y, _, extra = model(context_x[0:1], context_y[0:1], target_x[0:1], None)
        pred_y = pred_y.squeeze().cpu().numpy().squeeze()
        std_y = extra['y_dist'].scale.squeeze().cpu().numpy().squeeze()
        target_x = target_x.cpu().numpy()[0:1].squeeze()
        target_y = target_y.cpu().numpy()[0:1].squeeze()
        context_x = context_x.cpu().numpy()[0:1].squeeze()
        context_y = context_y.cpu().numpy()[0:1].squeeze()
        fig = get_light_curve_fig(pred_y, std_y, context_x, context_y, target_x, target_y)
        writer.add_figure('fit', fig, global_step=epoch)
    writer.add_scalars('val_loss',
                       {'loss': total_loss, 'kl': kl_loss,
                        'mse': losses['loss_mse'],
                        'meta': losses['loss_meta']},
                       epoch)
    return total_loss


def test(checkpoint_path):
    state_dict = torch.load(checkpoint_path)
    model = NeuralProcess(hidden_dim=8, latent_dim=8).cuda()
    model.load_state_dict(state_dict=state_dict['model'])
    test_seed = 789
    test_dataset = DRWDataset(Sampler(test_seed), 'test_drw',
                              num_samples=8,
                              seed=test_seed,
                              shift_x=-3650*0.5,
                              rescale_x=1.0/(3650*0.5)*4.0,
                              delta_x=1.0,
                              max_x=3650.0,
                              err_y=0.01)
    cadence_obj = LSSTCadence('obs', test_seed)
    rng = np.random.default_rng(test_seed)
    # Define collate fn that samples context points based on pointings
    collate_fn = partial(collate_fn_opsim,
                         rng=rng,
                         cadence_obj=cadence_obj,
                         n_pointings=10,
                         every_other=10)
    dloader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn,
                         shuffle=False)

    model.eval()
    with torch.no_grad():
        for i, d in enumerate(dloader):
            context_x, context_y, target_x, target_y, meta = d
            context_x = context_x.cuda()
            context_y = context_y.cuda()
            target_x = target_x.cuda()
            meta = meta.cuda()
            pred_y, _, extra = model(context_x, context_y, target_x, None)
            pred_y = pred_y.squeeze().cpu().numpy()
            std_y = extra['y_dist'].scale.squeeze().cpu().numpy()
            target_x = target_x.squeeze().cpu().numpy()
            target_y = target_y.squeeze().cpu().numpy()
            context_x = context_x.squeeze().cpu().numpy()
            context_y = context_y.squeeze().cpu().numpy()
            fig = get_light_curve_fig(pred_y, std_y, context_x, context_y, target_x, target_y)
            fig.savefig('light_curve.png')
            break


def get_params_fig(pred_mean, pred_log_sigma, truth):
    fig, ax = plt.subplots()
    truth_grid = np.linspace(truth.min(), truth.max(), 20)
    ax.errorbar(truth, pred_mean, np.exp(pred_log_sigma), fmt='o',
                color='tab:blue')
    ax.plot(truth_grid, truth_grid, linestyle='--', color='tab:gray')
    return fig


def get_light_curve_fig(pred_y, std_y, context_x, context_y, target_x, target_y):
    fig, ax = plt.subplots()
    target_sorted_i = np.argsort(target_x)
    target_x = target_x[target_sorted_i]
    pred_y = pred_y[target_sorted_i]
    std_y = std_y[target_sorted_i]
    ax.scatter(target_x, pred_y, marker='.', color='tab:blue')
    ax.fill_between(target_x,
                    pred_y - std_y,
                    pred_y + std_y,
                    alpha=0.2,
                    facecolor="tab:blue",
                    interpolate=True,
                    label="uncertainty",
                    )
    target_y = target_y[target_sorted_i]
    ax.scatter(target_x, target_y, marker='.', color='k', label='target')
    ax.scatter(context_x, context_y, marker='*', color='tab:orange', label='context')
    ax.legend()
    return fig


if __name__ == '__main__':
    run_dir = os.path.join('results', 'E4')
    train(run_dir)
    # test(os.path.join(run_dir, 'checkpoint.pth.tar'))