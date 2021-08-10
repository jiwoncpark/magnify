import os
import sys
sys.path.insert(0, '.')
from functools import partial
import numpy as np
import torch
from magnificat.drw_dataset import DRWDataset
from magnificat.samplers.dc2_sampler import DC2Sampler
from torch.utils.data import DataLoader
import lib.utils as utils

def get_data_min_max(records, device):
    """Get minimum and maximum for each feature (bandpass)
    across the whole dataset

    Parameters
    ----------
    records : iterable
        Each element is the dictionary of x, y, trimmed_mask, and params

    Returns
    -------
    tuple
        min and max values in y across the dataset
    """

    data_min, data_max = None, None
    inf = torch.Tensor([float("Inf")])[0].to(device)

    n_samples = len(records)
    print(f"Computing global min and max of {n_samples} training examples...")
    for data_i in range(n_samples):
        data = records[data_i]
        mask = data['trimmed_mask']
        vals = data['y']

        n_features = vals.size(-1)

        batch_min = []
        batch_max = []
        for i in range(n_features):
            non_missing_vals = vals[:, i][mask[:, i] == 1]
            if len(non_missing_vals) == 0:
                batch_min.append(inf)
                batch_max.append(-inf)
            else:
                batch_min.append(torch.min(non_missing_vals))
                batch_max.append(torch.max(non_missing_vals))

        batch_min = torch.stack(batch_min)
        batch_max = torch.stack(batch_max)

        if (data_min is None) and (data_max is None):
            data_min = batch_min
            data_max = batch_max
        else:
            data_min = torch.min(data_min, batch_min)
            data_max = torch.max(data_max, batch_max)

    return data_min.to(device), data_max.to(device)


def variable_time_collate_fn(batch, args, device=torch.device("cpu"),
                             data_type="train",
                             data_min=None, data_max=None):
    """
    Expects a batch of time series data in the form of (
    record_id, tt, vals, mask, labels) where
        - record_id is a patient id
        - tt is a 1-dimensional tensor containing T time values of
        observations.
        - vals is a (T, D) tensor containing observed values for D variables.
        - mask is a (T, D) tensor containing 1 where values were
        observed and 0 otherwise.
        - labels is a list of labels for the current patient,
        if labels are available. Otherwise None.
    Returns:
        combined_tt: The union of all time observations.
        combined_vals: (M, T, D) tensor containing the observed values.
        combined_mask: (M, T, D) tensor containing 1 where values
        were observed and 0 otherwise.
    """
    D = batch[0]['y'].shape[-1]  # first example (0), get y values (2),
    # and dim 1 of shape is Y_dim
    common_times = batch[0]['x']
    batch_size = len(batch)
    # Whether each time was observed in any filter in the batch
    observed = torch.zeros_like(common_times).to(bool)  # init unobserved
    for b in range(batch_size):
        # Was it observed in any of the filters, for this example?
        obs_example = batch[b]['trimmed_mask'].any(dim=1)  # [len(common_times),]
        observed = torch.logical_or(observed, obs_example)

    combined_tt = common_times[observed].to(device)
    inv_indices = torch.arange(len(combined_tt)-1, -1, -1)

    combined_vals = torch.zeros([batch_size, len(combined_tt), D])
    combined_mask = torch.zeros([batch_size, len(combined_tt), D])

    combined_labels = None
    N_labels = len(batch[0]['params'])

    combined_labels = torch.zeros(len(batch), N_labels) + torch.tensor(float('nan'))  # [B, N_labels]

    for b, data in enumerate(batch):
        # Slice observed times only
        vals = data['y'][observed, :]
        mask = data['trimmed_mask'][observed, :]
        if 'params' in data:
            labels = data['params']

        # Populate batch y, mask in inverse time
        combined_vals[b, inv_indices] = vals
        combined_mask[b, inv_indices] = mask.to(torch.float32)

        if labels is not None:
            combined_labels[b, :] = labels

    # Put on device
    combined_tt = combined_tt.to(device)
    combined_vals = combined_vals.to(device)
    combined_mask = combined_mask.to(device)
    combined_labels = combined_labels.to(device)
    combined_vals, _, _ = utils.normalize_masked_data(combined_vals,
                                                      combined_mask,
                                                      att_min=data_min,
                                                      att_max=data_max)

    if torch.max(combined_tt) != 0.:
        combined_tt = combined_tt / 3650.0  # out of 10 years, in days

    data_dict = {"data": combined_vals,
                 "time_steps": combined_tt,
                 "mask": combined_mask,
                 "labels": combined_labels}

    data_dict = utils.split_and_subsample_batch(data_dict,
                                                args,
                                                data_type=data_type)
    return data_dict


def collate_fn_opsim(batch, rng, n_pointings, frac_context=0.9,
                     exclude_ddf=True):
    x, y, meta = zip(*batch)  # batch ~ list of (x, y, param) tuples
    x = torch.stack(x, dim=0)  # [batch_size, n_points, n_filters]
    y = torch.stack(y, dim=0)  # [batch_size, n_points, n_filters]
    meta = torch.stack(meta, dim=0).float()  # [batch_size, n_params]
    # Log-parameterize some params
    # n_full_x = x.shape[1]
    obs_i = rng.choice(n_pointings)
    if exclude_ddf:
        while len(cadence_obj.get_mjd_single_pointing(obs_i, rounded=True)) > 1500:
            obs_i = rng.choice(n_pointings)
    target_i = cadence_obj.get_mjd_i_band_pointing(obs_i, rounded=True).astype(np.int32)  # [n_points,]
    sub_i = rng.choice(np.arange(len(target_i)),
                       size=int(len(target_i)*frac_context),
                       replace=False)  # [n_points*frac_context,]
    context_i = target_i[sub_i]
    context_i.sort()
    # every_other_10 = np.arange(0, n_full_x, every_other)
    # target_i = np.union1d(context_i, every_other_10)
    # target_i.sort()
    mask = torch.from_numpy(cadence_obj.get_mask_single_pointing(obs_i)).bool()  # [n_points, n_filters]
    return (x[:, context_i, :], y[:, context_i, :],
            x[:, target_i, :], y[:, target_i, :],
            meta,
            mask,
            torch.from_numpy(sub_i).long())


def get_drw_datasets(train_seed, val_seed):
    bandpasses = list('ugrizy')
    train_params = [f'tau_{bp}' for bp in bandpasses]
    train_params += [f'SF_inf_{bp}' for bp in bandpasses]
    train_params += ['BH_mass', 'M_i', 'redshift']
    train_params += [f'mag_{bp}' for bp in bandpasses]
    log_params = [True for bp in bandpasses]
    log_params += [True for bp in bandpasses]
    log_params += [False, False, False]
    log_params += [False for bp in bandpasses]
    n_pointings = 1000

    train_cat_idx = np.load('/home/jwp/stage/sl/latent_ode/train_idx.npy')  # 11227
    val_cat_idx = np.load('/home/jwp/stage/sl/latent_ode/val_idx.npy')  # 114
    n_train = len(train_cat_idx)
    n_val = len(val_cat_idx)
    train_dataset = DRWDataset(DC2Sampler(train_seed, bandpasses, train_cat_idx),
                               out_dir='/home/jwp/stage/sl/latent_ode/train_drw',
                               num_samples=n_train,
                               seed=train_seed,
                               n_pointings_init=n_pointings,
                               is_training=True,
                               shift_x=0.0,  # -3650*0.5,
                               rescale_x=1.0,  # 1.0/(3650*0.5)*4.0,
                               err_y=0.01)
    train_dataset.slice_params = [train_dataset.param_names.index(n) for n in train_params]
    train_dataset.log_params = log_params
    train_dataset.get_normalizing_metadata(set_metadata=True)

    # Validation data
    val_dataset = DRWDataset(DC2Sampler(val_seed, bandpasses, val_cat_idx),
                             out_dir='/home/jwp/stage/sl/latent_ode/val_drw',
                             num_samples=n_val,
                             seed=val_seed,
                             n_pointings_init=n_pointings,
                             is_training=False,
                             shift_x=0.0,  # -3650*0.5,
                             rescale_x=1.0,  # 1.0/(3650*0.5)*4.0,,
                             err_y=0.01)
    val_dataset.slice_params = train_dataset.slice_params
    val_dataset.log_params = log_params
    val_dataset.mean_params = train_dataset.mean_params
    val_dataset.std_params = train_dataset.std_params

    return train_dataset, val_dataset
