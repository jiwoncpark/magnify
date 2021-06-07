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


def collate_fn_opsim(batch, rng, cadence_obj, n_pointings, every_other=10,
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
    every_other_10 = np.arange(0, n_full_x, every_other)
    target_i = np.union1d(context_i, every_other_10)
    target_i.sort()
    return (x[:, context_i, :], y[:, context_i, :],
            x[:, target_i, :], y[:, target_i, :],
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