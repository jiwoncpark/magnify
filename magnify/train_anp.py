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
    def __init__(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def sample(self, N):
        # SF_inf = np.maximum(np.random.randn(N)*0.2 + 1.0, 0.2)
        # SF_inf = 10**(np.random.randn(N)*(0.25) + -0.8)
        SF_inf = np.ones(N)*0.15
        # tau = 10.0**np.maximum(np.random.randn(N)*0.5 + 2.0, 0.1)
        tau = np.maximum(np.random.randn(N)*10.0 + 200.0, 100.0)
        # mag = np.maximum(np.random.randn(N) + 19.0, 17.5)
        mag = np.zeros(N)
        # z = np.maximum(np.random.randn(N) + 2.0, 0.5)
        z = np.ones(N)*2.0
        return np.stack([SF_inf, tau, mag, z], axis=-1)


def train(run_dir):
    os.makedirs(run_dir, exist_ok=True)
    train_seed = 123
    train_dataset = DRWDataset(Sampler(train_seed), 'train_drw',
                               num_samples=100,
                               seed=train_seed,
                               shift_x=-3650*0.5,
                               rescale_x=1.0/(3650*0.5)*4.0,
                               delta_x=1.0,
                               max_x=3650.0,
                               err_y=0.01)
    # Generate pointings
    cadence_obj = LSSTCadence('obs', train_seed)
    n_pointings = 100
    ra, dec = cadence_obj.get_pointings(n_pointings)
    cadence_obj.get_obs_info(ra, dec)
    # Define collate fn that samples context points based on pointings
    collate_fn = partial(collate_fn_opsim,
                         rng=np.random.default_rng(train_seed),
                         cadence_obj=cadence_obj,
                         n_pointings=n_pointings,
                         every_other=10)
    train_loader = DataLoader(train_dataset, batch_size=2, collate_fn=collate_fn,
                              shuffle=True)
    # Validation data
    val_seed = 456
    val_dataset = DRWDataset(Sampler(val_seed), 'val_drw',
                             num_samples=8,
                             seed=val_seed,
                             shift_x=-3650*0.5,
                             rescale_x=1.0/(3650*0.5)*4.0,
                             delta_x=1.0,
                             max_x=3650.0,
                             err_y=0.01)
    collate_fn = partial(collate_fn_opsim,
                         rng=np.random.default_rng(val_seed),
                         cadence_obj=cadence_obj,
                         n_pointings=n_pointings,
                         every_other=10)
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn,
                            shuffle=True)
    epochs = 1000
    model = NeuralProcess(hidden_dim=16, latent_dim=16).cuda()
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5, patience=10, verbose=True)
    writer = SummaryWriter(run_dir)
    for epoch in tqdm(range(epochs)):
        train_single_epoch(model, train_loader, optim, epoch, writer)
        val_loss = eval(model, val_loader, optim, epoch, writer)
        scheduler.step(val_loss)
        # Save model by each epoch
        torch.save({'model': model.state_dict(),
                    'optimizer': optim.state_dict(),
                    'scheduler': scheduler.state_dict()},
                   os.path.join(run_dir, 'checkpoint.pth.tar'))


def train_single_epoch(model, train_loader, optim, epoch, writer):
    total_loss, kl_loss, mse_loss = 0.0, 0.0, 0.0
    model.train()
    for i, data in enumerate(train_loader):
        context_x, context_y, target_x, target_y = data
        context_x = context_x.cuda()
        context_y = context_y.cuda()
        target_x = target_x.cuda()
        target_y = target_y.cuda()
        # pass through the latent model
        y_pred, losses, extra = model(context_x, context_y, target_x, target_y)
        loss = losses['loss']
        # Training step
        optim.zero_grad()
        loss.backward()
        optim.step()
        # Logging
        total_loss += (loss - total_loss)/(i+1)
        kl_loss += (losses['loss_kl'] - kl_loss)/(i+1)
        mse_loss += (losses['loss_mse'] - mse_loss)/(i+1)
    writer.add_scalars('training_loss',
                       {'loss': total_loss, 'kl': kl_loss,
                        'mse': losses['loss_mse']},
                       epoch)


def eval(model, val_loader, optim, epoch, writer):
    total_loss, kl_loss, mse_loss = 0.0, 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            context_x, context_y, target_x, target_y = data
            context_x = context_x.cuda()
            context_y = context_y.cuda()
            target_x = target_x.cuda()
            target_y = target_y.cuda()
            # pass through the latent model
            pred_y, losses, extra = model(context_x, context_y, target_x, target_y)
            loss = losses['loss']
            # Logging
            total_loss += (loss - total_loss)/(i+1)
            kl_loss += (losses['loss_kl'] - kl_loss)/(i+1)
            mse_loss += (losses['loss_mse'] - mse_loss)/(i+1)
        # Visualize fit on last light curve
        pred_y, _, extra = model(context_x, context_y, target_x, None)
    pred_y = pred_y.squeeze().cpu().numpy()
    std_y = extra['y_dist'].scale.squeeze().cpu().numpy()
    target_x = target_x.squeeze().cpu().numpy()
    target_y = target_y.squeeze().cpu().numpy()
    context_x = context_x.squeeze().cpu().numpy()
    context_y = context_y.squeeze().cpu().numpy()
    fig = get_fig(pred_y, std_y, context_x, context_y, target_x, target_y)
    writer.add_figure('fit', fig, global_step=epoch)
    writer.add_scalars('val_loss',
                       {'loss': total_loss, 'kl': kl_loss,
                        'mse': losses['loss_mse']},
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
            context_x, context_y, target_x, target_y = d
            context_x = context_x.cuda()
            context_y = context_y.cuda()
            target_x = target_x.cuda()
            pred_y, _, extra = model(context_x, context_y, target_x, None)
            pred_y = pred_y.squeeze().cpu().numpy()
            std_y = extra['y_dist'].scale.squeeze().cpu().numpy()
            target_x = target_x.squeeze().cpu().numpy()
            target_y = target_y.squeeze().cpu().numpy()
            context_x = context_x.squeeze().cpu().numpy()
            context_y = context_y.squeeze().cpu().numpy()
            fig = get_fig(pred_y, std_y, context_x, context_y, target_x, target_y)
            fig.savefig('light_curve.png')
            break


def get_fig(pred_y, std_y, context_x, context_y, target_x, target_y):
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
    run_dir = os.path.join('results', 'E1')
    train(run_dir)
    # test(os.path.join(run_dir, 'checkpoint.pth.tar'))