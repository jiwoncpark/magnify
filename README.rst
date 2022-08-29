=======================================================================================================================
mAGNify - Prediction of black hole properties from learned representations of LSST-like multivariate time series of AGN
=======================================================================================================================

Joint reconstruction and parameter regression experiments

Installation
============

::

$pip install -e .

Models available
================

1. Attentive neural process (Kim et al 2018)

::

$python magnify/train_anp.py

2. Latent ODE (Rubanova et al 2019)

* Toy dataset of 1d periodic functions with varying frequency

::

$python magnify/train_latent_ode.py --niters 600 -n 1000 -s 50 -l 10 --dataset periodic --latent-ode --noise-weight 0.01 --regress


* Mock AGN light curves, simulated using the damped random walk model

::

$python magnify/train_latent_ode.py --batch-size 60 --niters 50 -n 10000 -l 20 --dataset drw --latent-ode --regress


3. Latent SDE (Li et al 2020)

* Single mock AGN light curve, simulated using the damped random walk model

::

$pip install torchsde
$python magnify/train_latent_sde_param_pred.py

