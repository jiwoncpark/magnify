

def hparams_power(hparams):
    """Some value we want to go up in powers of 2

    So any hyper param that ends in power will be used this way.
    """
    hparams_old = hparams.copy()
    for k in hparams_old.keys():
        if k.endswith("_power"):
            k_new = k.replace("_power", "")
            hparams[k_new] = int(2 ** hparams[k])
    return hparams


def log_prob_sigma(value, loc, log_scale):
    """A slightly more stable (not confirmed yet) log prob taking in log_var instead of scale.
    modified from https://github.com/pytorch/pytorch/blob/2431eac7c011afe42d4c22b8b3f46dedae65e7c0/torch/distributions/normal.py#L65
    """
    var = torch.exp(log_scale * 2)
    return (
        -((value - loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))
    )


def kl_loss_var(prior_mu, log_var_prior, post_mu, log_var_post):
    """
    Analytical KLD for two gaussians, taking in log_variance instead of scale ( given variance=scale**2) for more stable gradients

    For version using scale see https://github.com/pytorch/pytorch/blob/master/torch/distributions/kl.py#L398
    """

    var_ratio_log = log_var_post - log_var_prior
    kl_div = (
        (var_ratio_log.exp() + (post_mu - prior_mu) ** 2) / log_var_prior.exp()
        - 1.0
        - var_ratio_log
    )
    kl_div = 0.5 * kl_div
    logger.warning('seems to be an error in kl_loss_var, dont use it')
    return kl_div