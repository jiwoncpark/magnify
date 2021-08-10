# Latent ODEs for Irregularly-Sampled Time Series Applied to Mock AGN Light Curves


The latent ODE paper:
> Yulia Rubanova, Ricky Chen, David Duvenaud. "Latent ODEs for Irregularly-Sampled Time Series" (2019)
[[arxiv]](https://arxiv.org/abs/1907.03907)

<p align="center">
<img align="middle" src="./assets/viz.gif" width="800" />
</p>

## Prerequisites

Install `torchdiffeq` from https://github.com/rtqichen/torchdiffeq.

## Joint reconstruction and parameter regression experiments

* Toy dataset of 1d periodic functions with varying frequency
```bash
python run_models.py --niters 600 -n 1000 -s 50 -l 10 --dataset periodic --latent-ode --noise-weight 0.01 --regress
```

* Mock AGN light curves, simulated using the damped random walk model
```bash
python run_models.py --batch-size 10 --niters 600 -n 1000 -l 10 --dataset drw --latent-ode --regress
```

### Running different models

* ODE-RNN
```
python3 run_models.py --niters 500 -n 1000 -l 10 --dataset periodic  --ode-rnn
```

* Latent ODE with ODE-RNN encoder
```
python3 run_models.py --niters 500 -n 1000 -l 10 --dataset periodic  --latent-ode
```

* Latent ODE with ODE-RNN encoder and poisson likelihood
```
python3 run_models.py --niters 500 -n 1000 -l 10 --dataset periodic  --latent-ode --poisson
```

* Latent ODE with RNN encoder (Chen et al, 2018)
```
python3 run_models.py --niters 500 -n 1000 -l 10 --dataset periodic  --latent-ode --z0-encoder rnn
```

* RNN-VAE
```
python3 run_models.py --niters 500 -n 1000 -l 10 --dataset periodic  --rnn-vae
```

*  Classic RNN
```
python3 run_models.py --niters 500 -n 1000 -l 10 --dataset periodic  --classic-rnn
```

* GRU-D

GRU-D consists of two parts: input imputation (--input-decay) and exponential decay of the hidden state (--rnn-cell expdecay)

```
python3 run_models.py --niters 500 -n 100  -b 30 -l 10 --dataset periodic  --classic-rnn --input-decay --rnn-cell expdecay
```


### Making the visualization
```
python3 run_models.py --niters 100 -n 5000 -b 100 -l 3 --dataset periodic --latent-ode --noise-weight 0.5 --lr 0.01 --viz --rec-layers 2 --gen-layers 2 -u 100 -c 30
```

Also check out the demo notebooks!
