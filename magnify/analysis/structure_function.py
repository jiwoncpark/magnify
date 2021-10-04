"""Various methods for computing the structure function of a time series

"""

import numpy as np
from tqdm import tqdm
from astroML.time_series import ACF_scargle, ACF_EK
from scipy import fftpack


def sf_from_acf(acf):
    """Convert autocorrelation function (ACF) into structure function (SF)

    Parameters
    ----------
    acf : np.ndarray
    """
    return (1.0 - acf)**0.5


def get_acf_numpy(y):
    """Get the ACF using numpy. Works best for finely and regularly
    sampled time series

    Parameters
    ----------
    y : np.ndarray
        Observed time series

    Returns
    -------
    np.ndarray
        ACF computed from 1-element offset in y

    """
    def autocorr(x, t=1):
        return np.corrcoef(np.array([x[:-t], x[t:]]))
    acf = np.empty(*y.shape)  # init
    acf[0] = np.corrcoef(np.array([y, y]))[0, 1]
    for i, lag_i in tqdm(enumerate(np.arange(1, len(y))), total=len(y)-1):
        acf[lag_i] = autocorr(y, lag_i)[0, 1]
    return acf


def get_acf_scargle(t, y,
                    dy=1e-6, n_omega=2.0**12, omega_max=np.pi/5.0):
    """Compute the Auto-correlation function via Scargle's method

    Parameters
    ----------
    t : array_like
        times of observation.  Assumed to be in increasing order.
    y : array_like
        values of each observation.  Should be same shape as t
    dy : float or array_like
        errors in each observation.
    n_omega : int (optional)
        number of angular frequencies at which to evaluate the periodogram
        default is 2^10
    omega_max : float (optional)
        maximum value of omega at which to evaluate the periodogram
        default is 100

    Note
    ----
    Source code from astroML.time_series.ACF

    Returns
    -------
    ACF, t : ndarrays
        The auto-correlation function and associated times
    """
    ACF, t = ACF_scargle(t, y, dy, n_omega, omega_max)
    return ACF, t


def get_acf_ek(t, y, dy, bins=20):
    """Auto-correlation function via the Edelson-Krolik method

    Parameters
    ----------
    t : array_like
        times of observation.  Assumed to be in increasing order.
    y : array_like
        values of each observation.  Should be same shape as t
    dy : float or array_like
        errors in each observation.
    bins : int or array_like (optional)
        if integer, the number of bins to use in the analysis.
        if array, the (nbins + 1) bin edges.
        Default is bins=20.

    Note
    ----
    Source code from astroML.time_series.ACF

    Returns
    -------
    ACF : ndarray
        The auto-correlation function and associated times
    err : ndarray
        the error in the ACF
    bins : ndarray
        bin edges used in computation
    """
    ACF, ACF_err, bins = ACF_EK(t, y, dy, bins=20)
    return ACF, ACF_err, bins
