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


def collate_fn_opsim(batch, rng, cadence_obj, n_pointings, every_other=10):
    x, y, meta = zip(*batch)  # batch ~ list of (x, y, param) tuples
    x = torch.stack(x, axis=0)  # [batch_size, n_points, 1]
    y = torch.stack(y, axis=0)  # [batch_size, n_points, 1]
    meta = torch.stack(meta, axis=0)  # [batch_size, n_params]
    n_full_x = x.shape[1]
    obs_i = rng.choice(n_pointings)
    every_other_10 = np.arange(0, n_full_x, every_other)
    context_i = cadence_obj.get_mjd_single_pointing(obs_i,
                                                    rounded=True).astype(np.int32)
    target_i = np.union1d(context_i, every_other_10)
    return (x[:, context_i, :], y[:, context_i, :],
            x[:, target_i, :], y[:, target_i, :],
            meta)
