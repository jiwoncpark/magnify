import torch
import numpy as np


class GetRandomObservations:
    """
    Return random subset of indices corresponding to observations

    """

    def __init__(self, out_dir, n_pointings: int, seed: int = 123):
        from numpy.random import default_rng
        from magnificat.cadence import LSSTCadence
        self.out_dir = out_dir
        self.n_pointings = n_pointings
        self.seed = seed
        self.rng = default_rng(self.seed)
        self.cadence_obj = LSSTCadence(out_dir, self.seed)
        ra, dec = self.cadence_obj.get_pointings(self.n_pointings)
        self.cadence_obj.get_obs_info(ra, dec)

    def __call__(self, batch_size, n_possible_points):
        # Randomly sample observation
        obs_i = self.rng.choice(self.n_pointings)
        obs_idx = self.cadence_obj.get_mjd_single_pointing(obs_i, rounded=True).astype(np.int32)
        # All the available indices, with batching
        idx_all = (
            np.arange(n_possible_points)
            .reshape(1, n_possible_points)
            .repeat(batch_size, axis=0)
        )
        idx = torch.from_numpy(idx_all[:, obs_idx])
        return idx


def collate_fn_opsim(batch, rng, cadence_obj, n_pointings, frac_context=0.9,
                     exclude_ddf=True):
    x, y, meta = zip(*batch)  # batch ~ list of (x, y, param) tuples
    x = torch.stack(x, axis=0)  # [batch_size, n_points, n_filters]
    y = torch.stack(y, axis=0)  # [batch_size, n_points, n_filters]
    meta = torch.stack(meta, axis=0).float()  # [batch_size, n_params]
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


def collate_fn_opsim_v1(batch, rng, cadence_obj, n_pointings, frac_context=0.9,
                        exclude_ddf=True):
    x, y, meta = zip(*batch)  # batch ~ list of (x, y, param) tuples
    x = torch.stack(x, axis=0)  # [batch_size, n_points, n_filters]
    y = torch.stack(y, axis=0)  # [batch_size, n_points, n_filters]
    meta = torch.stack(meta, axis=0).float()  # [batch_size, n_params]
    # Log-parameterize some params
    n_full_x = x.shape[1]
    obs_i = rng.choice(n_pointings)
    if exclude_ddf:
        while len(cadence_obj.get_mjd_single_pointing(obs_i, rounded=True)) > 1500:
            obs_i = rng.choice(n_pointings)
    context_i = cadence_obj.get_mjd_single_pointing(obs_i, rounded=True).astype(np.int32)  # [n_points,]

    every_other_10 = np.arange(0, n_full_x, 20)
    target_i = np.union1d(context_i, every_other_10)
    target_i.sort()
    #mask = torch.from_numpy(cadence_obj.get_mask_single_pointing(obs_i)).bool()  # [n_points, n_filters]
    return (x[:, context_i, :], y[:, context_i, :],
            x[:, target_i, :], y[:, target_i, :],
            meta,
            None, None)


def collate_fn_sdss(batch, device, pred_frac=0.1):
    x, y, meta = zip(*batch)  # batch ~ list of (x, y, param) tuples
    x = [x_i.unsqueeze(0).to(device) for x_i in x]
    y = [y_i.unsqueeze(0).to(device) for y_i in y]
    meta = [m.unsqueeze(0).to(device) for m in meta]
    n_obs = [x_i.shape[1] for x_i in x]
    obs_frac = 1.0 - pred_frac
    # Sorted random idx of observed times
    # Note that sort() returns a named tuple of values, indices
    context_i = [torch.randint(low=0,
                               high=n_obs_i,
                               size=[int(n_obs_i*obs_frac)],
                               dtype=torch.int64).sort().values for n_obs_i in n_obs]
    x_context = [x_i[:, context_i[i], :] for i, x_i in enumerate(x)]  # length batch_size
    y_context = [y_i[:, context_i[i], :] for i, y_i in enumerate(y)]  # length batch_size
    return (x_context, y_context,
            x, y,
            meta)


def collate_fn_baseline(batch, rng, cadence_obj, n_pointings, every_other=10,
                        exclude_ddf=True, pointings_band=3):
    x, y, meta = zip(*batch)  # batch ~ list of (x, y, param) tuples
    x = torch.stack(x, axis=0)  # [batch_size, n_points, n_filters]
    y = torch.stack(y, axis=0)  # [batch_size, n_points, n_filters]
    meta = torch.stack(meta, axis=0).float()  # [batch_size, n_params]
    # Log-parameterize some params
    obs_i = rng.choice(n_pointings)
    if exclude_ddf:
        while len(cadence_obj.get_mjd_single_pointing(obs_i, rounded=True)) > 1500:
            obs_i = rng.choice(n_pointings)
    context_i = cadence_obj.get_mjd_single_pointing(obs_i, rounded=True).astype(np.int32)  # [n_points,]
    # Compute summary stats
    flux_mean = torch.mean(y[:, context_i, :], dim=1)  # [batch_size, n_filters]
    flux_std = torch.std(y[:, context_i, :], dim=1)  # [batch_size, n_filters]
    return (flux_mean, flux_std, meta)


def collate_fn_multi_filter(batch, rng, cadence_obj, n_pointings, every_other=10,
                            exclude_ddf=True):
    x, y, meta = zip(*batch)  # batch ~ list of (x, y, param) tuples
    x = torch.stack(x, axis=0)  # [batch_size, n_points, n_filters]
    y = torch.stack(y, axis=0)  # [batch_size, n_points, n_filters]
    meta = torch.stack(meta, axis=0)  # [batch_size, n_params]
    # Log-parameterize some params
    obs_i = rng.choice(n_pointings)
    if exclude_ddf:
        while len(cadence_obj.get_mjd_single_pointing(obs_i, rounded=True)) > 1500:
            obs_i = rng.choice(n_pointings)
    mjd = torch.from_numpy(cadence_obj.get_mjd_single_pointing(obs_i, rounded=True).astype(np.int32))
    mask_c = torch.from_numpy(cadence_obj.get_mask_single_pointing(obs_i))  # [n_points, n_filters]
    mask_t = torch.from_numpy(rng.choice([True, False], size=mask_c.shape, p=[0.1, 0.9]))
    return (x, y, mjd, mask_c, mask_t,
            meta)

if __name__ == '__main__':
    from magnificat.sdss_dr7_dataset import SDSSDR7Dataset
    from torch.utils.data import DataLoader
    from functools import partial
    agn_params = ['M_i', 'BH_mass', 'redshift']
    bp_params = ['log_rf_tau', 'log_sf_inf']
    bandpasses = list('ugriz')
    val_frac = 0.1
    seed = 123
    dataset = SDSSDR7Dataset(out_dir='sdss_dr7',
                             agn_params=agn_params,
                             bp_params=bp_params,
                             bandpasses=bandpasses,
                             num_samples=10,
                             metadata_kwargs=dict(keep_agn_mode='max_obs'),
                             light_curve_kwargs=dict(),)
    # Define collate fn that samples context points based on pointings
    collate_fn = partial(collate_fn_sdss,
                         pred_frac=0.1,
                         device=torch.device('cpu')
                         )
    loader = DataLoader(dataset,
                        batch_size=7,
                        collate_fn=collate_fn,
                        )
    dataset.get_normalizing_metadata(loader)
    print(dataset.mean_params)
    x, y, meta = dataset[0]
    print(x.shape, y.shape, meta.shape)
    print(loader.batch_size)
    for d in loader:
        print("length of d: ", len(d))
        print("batch size: ", len(d[0]))
        print("params of first example: ", d[-1][0])
