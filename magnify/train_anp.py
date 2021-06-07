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
from magnificat.samplers.dc2_sampler import DC2Sampler
from magnificat.cadence import LSSTCadence
from torch.utils.data import DataLoader
from magnify.attentive_neural_process.network import NeuralProcess
from magnify.attentive_neural_process.context_target_sampler import collate_fn_opsim


def train(run_dir, train_params, bandpasses, log_params,
          n_train=10000, n_val=50, n_pointings=1000,
          checkpoint_path=None):
    torch.cuda.empty_cache()
    os.makedirs(run_dir, exist_ok=True)
    train_seed = 123
    n_agn = 11441
    cat_idx = np.arange(n_agn)
    np.random.default_rng(train_seed).shuffle(cat_idx)
    train_cat_idx = cat_idx[: n_train]
    val_cat_idx = cat_idx[-n_val:]
    train_dataset = DRWDataset(DC2Sampler(train_seed, bandpasses, train_cat_idx), 'train_drw',
                               num_samples=n_train,
                               seed=train_seed,
                               shift_x=-3650*0.5,
                               rescale_x=1.0/(3650*0.5)*4.0,
                               delta_x=1.0,
                               max_x=3650.0,
                               err_y=0.01,
                               bandpasses=bandpasses)
    train_dataset.slice_params = [train_dataset.param_names.index(n) for n in train_params]
    train_dataset.log_params = log_params
    train_dataset.get_normalizing_metadata(set_metadata=True)
    print(train_dataset.slice_params)
    print(train_dataset.mean_params, train_dataset.std_params)
    # Generate pointings
    cadence_obj = LSSTCadence('obs', train_seed)
    ra, dec = cadence_obj.get_pointings(n_pointings)
    cadence_obj.get_obs_info(ra, dec)
    cadence_obj.set_bandpasses(bandpasses)
    # Define collate fn that samples context points based on pointings
    collate_fn = partial(collate_fn_opsim,
                         rng=np.random.default_rng(train_seed),
                         cadence_obj=cadence_obj,
                         n_pointings=n_pointings,
                         every_other=10,
                         exclude_ddf=True,)
    train_loader = DataLoader(train_dataset, batch_size=20, collate_fn=collate_fn,
                              shuffle=True)
    # Validation data
    val_seed = 456
    val_dataset = DRWDataset(DC2Sampler(val_seed, bandpasses, val_cat_idx), 'val_drw',
                             num_samples=n_val,
                             seed=val_seed,
                             shift_x=-3650*0.5,
                             rescale_x=1.0/(3650*0.5)*4.0,
                             delta_x=1.0,
                             max_x=3650.0,
                             err_y=0.01,
                             bandpasses=bandpasses)
    val_dataset.slice_params = train_dataset.slice_params
    val_dataset.log_params = log_params
    val_dataset.mean_params = train_dataset.mean_params
    val_dataset.std_params = train_dataset.std_params
    collate_fn = partial(collate_fn_opsim,
                         rng=np.random.default_rng(val_seed),
                         cadence_obj=cadence_obj,
                         n_pointings=n_pointings,
                         every_other=10)
    val_loader = DataLoader(val_dataset, batch_size=n_val, collate_fn=collate_fn,
                            shuffle=False)
    epochs = 400
    model = NeuralProcess(x_dim=len(bandpasses),
                          y_dim=len(bandpasses),
                          hidden_dim=32, latent_dim=32, weight_y_loss=1.0,
                          n_target=len(train_dataset.slice_params),
                          ).cuda()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params: ", total_params)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5,
                                                           patience=15, verbose=True)
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path)
        model.load_state_dict(state['model'])
        optim.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scheduler.patience = 10
        def get_lr(gamma, optimizer):
            return [group['lr'] * gamma
                    for group in optimizer.param_groups]
        for param_group, lr in zip(optim.param_groups, get_lr(0.2, optim)):
            param_group['lr'] = lr

        # print(scheduler.__dict__)
    model.train()
    min_val_loss = np.inf
    writer = SummaryWriter(run_dir)
    for epoch in tqdm(range(epochs)):
        train_single_epoch(model, train_loader, val_loader, optim, epoch, writer)
        val_loss = eval(model, val_loader, epoch, writer, log=False)
        scheduler.step(val_loss)
        # Save model if validation loss decreased
        if val_loss < min_val_loss:
            torch.save({'model': model.state_dict(),
                        'optimizer': optim.state_dict(),
                        'scheduler': scheduler.state_dict()},
                       os.path.join(run_dir, 'checkpoint.pth.tar'))
            min_val_loss = val_loss


def train_single_epoch(model, train_loader, val_loader, optim, epoch, writer):
    total_loss, mse_loss, meta_loss = 0.0, 0.0, 0.0
    for i, data in enumerate(train_loader):
        model.train()
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
        mse_loss += (losses['loss_mse'] - mse_loss)/(i+1)
        meta_loss += (losses['loss_meta'] - meta_loss)/(i+1)
        if i % 100 == 0:
            eval(model, val_loader, epoch*len(train_loader)+i, writer)
    writer.add_scalars('training_loss',
                       {'loss': total_loss,
                        'meta': losses['loss_meta']},
                       epoch)


def eval(model, val_loader, epoch, writer, log=True):
    total_loss, mse_loss, meta_loss = 0.0, 0.0, 0.0
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
            mse_loss += (losses['loss_mse'] - mse_loss)/(i+1)
            meta_loss += (losses['loss_meta'] - meta_loss)/(i+1)
        if False:
            for p in range(meta.shape[1]):
                fig = get_params_fig(extra['mean_meta'].cpu().numpy()[:, p],
                                     extra['log_sigma_meta'].cpu().numpy()[:, p],
                                     meta.cpu().numpy()[:, p])
                writer.add_figure(f'param {p} recovery', fig, global_step=epoch)
        # Get histogram of errors
        if log:
            model.param_loss.set_trained_pred(extra['pred_meta'])
            sample = model.param_loss.sample(mean=torch.zeros([1, meta.shape[1]]).cuda(),
                                             std=torch.ones([1, meta.shape[1]]).cuda(),
                                             n_samples=100).mean(1)  # [n_batch, Y_dim]
            error = np.mean(sample - meta.cpu().numpy(), axis=-1)
            writer.add_histogram('Mean error', error, epoch)
        # Visualize fit on first light curve
        if log:
            bp_i = 0
            pred_y, _, extra = model(context_x[0:1], context_y[0:1], target_x[0:1], None)
            pred_y = pred_y.cpu().numpy()[0, :, bp_i]
            std_y = extra['y_dist'].scale.cpu().numpy()[0, :, bp_i]
            target_x = target_x.cpu().numpy()[0, :, bp_i]
            target_y = target_y.cpu().numpy()[0, :, bp_i]
            context_x = context_x.cpu().numpy()[0, :, bp_i]
            context_y = context_y.cpu().numpy()[0, :, bp_i]
            fig = get_light_curve_fig(pred_y, std_y, context_x, context_y, target_x, target_y)
            writer.add_figure('fit', fig, global_step=epoch)
        if log:
            writer.add_scalars('val_loss',
                               {'loss': total_loss,
                                'meta': losses['loss_meta']},
                           epoch)
    return total_loss


def get_params_fig(pred_mean, pred_log_sigma, truth):
    fig, ax = plt.subplots()
    truth_grid = np.linspace(truth.min(), truth.max(), 20)
    ax.errorbar(truth, pred_mean, np.exp(pred_log_sigma), fmt='o',
                color='tab:blue')
    ax.plot(truth_grid, truth_grid, linestyle='--', color='tab:gray')
    return fig


def get_light_curve_fig(pred_y, std_y, context_x, context_y, target_x, target_y):
    fig, ax = plt.subplots(figsize=(10, 5))
    target_x = target_x
    pred_y = pred_y
    std_y = std_y
    ax.scatter(target_x, pred_y, marker='.', color='tab:blue')
    ax.fill_between(target_x,
                    pred_y - std_y,
                    pred_y + std_y,
                    alpha=0.2,
                    facecolor="tab:blue",
                    interpolate=True,
                    label="uncertainty",
                    )
    ax.scatter(target_x, target_y, marker='.', color='k', label='target')
    ax.scatter(context_x, context_y, marker='*', color='tab:orange', label='context')
    #ax.legend()
    return fig


if __name__ == '__main__':
    run_dir = os.path.join('results', 'E1')
    bandpasses = list('ugrizy')
    train_params = [f'tau_{bp}' for bp in bandpasses]
    train_params += [f'SF_inf_{bp}' for bp in bandpasses]
    train_params += ['BH_mass', 'M_i', 'redshift']
    train_params += [f'mag_{bp}' for bp in bandpasses]
    log_params = [True for bp in bandpasses]
    log_params += [True for bp in bandpasses]
    log_params += [False, False, False]
    log_params += [False for bp in bandpasses]

    train(run_dir, train_params, bandpasses, n_train=11000, n_val=50,
          n_pointings=1000, log_params=log_params,
          checkpoint_path=os.path.join(run_dir, 'checkpoint.pth.tar')
          )
    # test(os.path.join(run_dir, 'checkpoint.pth.tar'))