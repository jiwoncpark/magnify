"""Utility functions used in inference

"""
import numpy as np
import torch


def get_samples(model,
                n_samples, ts_query, ts_data, ys_data,
                device=None):
    """Get posterior samples of the light curve for an AGN

    Parameters
    ----------
    model : torch.nn
        Instance of the Latent SDE model
    n_samples : int
        Number of samples to get
    ts_vis : torch.tensor
        Query times
    ts_data : torch.tensor
        Data times, of shape `[T_data,]`
    ys_data : torch.tensor
        Data magnitudes, of shape `[T_data, 1, out_dim]` where
        1 is the trivial batch size for a single AGN

    Returns
    -------
    np.ndarray
        Posterior samples of the light curve of shape
        `[T_query, out_dim, n_samples]`

    """
    if device is None:
        device = torch.device('cpu')
    out_dim = model.out_dim
    samples = np.empty([len(ts_query), out_dim, n_samples])
    for s in range(n_samples):
        sample = model.sample_posterior(1,  # dummy, doesn't matter
                                        ys_data.to(device),
                                        ts_data.to(device),
                                        ts_query.to(device),
                                        bm=None,
                                        method='euler').squeeze()
        sample_ = sample.cpu().numpy()  # [T_query, out_dim]
        samples[:, :, s] = sample_  # Populate
    return samples

