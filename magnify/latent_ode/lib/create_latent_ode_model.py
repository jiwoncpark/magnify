###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import torch.nn as nn
import magnify.latent_ode.lib.utils as utils
from magnify.latent_ode.lib.latent_ode import LatentODE
from magnify.latent_ode.lib.encoder_decoder import *
from magnify.latent_ode.lib.diffeq_solver import DiffeqSolver
from magnify.latent_ode.lib.ode_func import ODEFunc, ODEFunc_w_Poisson

#####################################################################################################


def create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device,
                           classif_per_tp=False, n_labels=1):
    """

    Parameters
    ----------
    classif_per_tp: whether to do classification per time point rather than
                    time series as a whole

    Relevant args (default values)

    latents: Size of the latent state (10)
    poisson: Model poisson-process likelihood for the density of events
             in addition to reconstruction. (True)
    z0_encoder: Type of encoder for Latent ODE model: odernn or rnn (odernn)
    classif: Include binary classification loss -- used for Physionet dataset
             for hospiral mortality
    linear_classif: If using a classifier, use a linear classifier instead of
                    1-layer NN

    """

    dim = args.latents
    if args.poisson:
        lambda_net = utils.create_net(dim, input_dim,
                                      n_layers = 1, n_units = args.units, nonlinear = nn.Tanh)

        # ODE function produces the gradient for latent state and for poisson rate
        ode_func_net = utils.create_net(dim * 2, args.latents * 2,
                                        n_layers = args.gen_layers, n_units = args.units, nonlinear = nn.Tanh)

        gen_ode_func = ODEFunc_w_Poisson(
                                         input_dim = input_dim,
                                         latent_dim = args.latents * 2,
                                         ode_func_net = ode_func_net,
                                         lambda_net = lambda_net,
                                         device = device).to(device)
    else:
        dim = args.latents
        ode_func_net = utils.create_net(dim, args.latents,
                                        n_layers = args.gen_layers, n_units = args.units, nonlinear = nn.Tanh)

        gen_ode_func = ODEFunc(
                               input_dim = input_dim,
                               latent_dim = args.latents,
                               ode_func_net = ode_func_net,
                               device = device).to(device)

    z0_diffeq_solver = None
    n_rec_dims = args.rec_dims
    enc_input_dim = int(input_dim) * 2  # we concatenate the mask
    gen_data_dim = input_dim

    z0_dim = args.latents  #
    if args.poisson:
        z0_dim += args.latents  # predict the initial poisson rate

    if args.z0_encoder == "odernn":
        ode_func_net = utils.create_net(n_rec_dims, n_rec_dims,
            n_layers = args.rec_layers, n_units = args.units, nonlinear = nn.Tanh)

        rec_ode_func = ODEFunc(
            input_dim = enc_input_dim,
            latent_dim = n_rec_dims,
            ode_func_net = ode_func_net,
            device = device).to(device)

        z0_diffeq_solver = DiffeqSolver(enc_input_dim, rec_ode_func, "euler", args.latents,
            odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)

        encoder_z0 = Encoder_z0_ODE_RNN(n_rec_dims, enc_input_dim, z0_diffeq_solver,
            z0_dim = z0_dim, n_gru_units = args.gru_units, device = device).to(device)

    elif args.z0_encoder == "rnn":
        encoder_z0 = Encoder_z0_RNN(z0_dim, enc_input_dim,
                                 lstm_output_size = n_rec_dims, device = device).to(device)
    else:
        raise Exception("Unknown encoder for Latent ODE model: " + args.z0_encoder)

    decoder = Decoder(args.latents, gen_data_dim).to(device)

    diffeq_solver = DiffeqSolver(gen_data_dim, gen_ode_func, 'dopri5',
                                args.latents,
                                odeint_rtol=1e-3, odeint_atol=1e-4,
                                device=device)

    model = LatentODE(
                    input_dim = gen_data_dim,
                    latent_dim = args.latents,
                    encoder_z0 = encoder_z0,
                    decoder = decoder,
                    diffeq_solver = diffeq_solver,
                    z0_prior = z0_prior,
                    device = device,
                    obsrv_std = obsrv_std,
                    use_poisson_proc = args.poisson,
                    use_classifier=args.classif,
                    use_regressor=args.regress,
                    linear_classifier = args.linear_classif,
                    classif_per_tp = classif_per_tp,
                    n_labels = n_labels
                    ).to(device)

    return model
