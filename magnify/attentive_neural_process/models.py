"""Taken with minor modifications from

https://github.com/3springs/attentive-neural-processes/blob/af431a267bad309b2d5698f25551986e2c4e7815/neural_processes/models/neural_process/model.py

"""

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from magnify.attentive_neural_process.modules.layers import BatchMLP, LSTMBlock
from magnify.attentive_neural_process.modules.attention import Attention


class LatentEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=32,
        latent_dim=32,
        self_attention_type="dot",
        n_encoder_layers=3,
        min_std=0.01,
        batchnorm=False,
        dropout=0,
        attention_dropout=0,
        use_lvar=False,
        use_self_attn=False,
        attention_layers=2,
        use_lstm=False
    ):
        super().__init__()
        # self._input_layer = nn.Linear(input_dim, hidden_dim)
        if use_lstm:
            self._encoder = LSTMBlock(input_dim, hidden_dim, batchnorm=batchnorm,
                                      dropout=dropout, num_layers=n_encoder_layers)
        else:
            self._encoder = BatchMLP(input_dim, hidden_dim, batchnorm=batchnorm,
                                     dropout=dropout, num_layers=n_encoder_layers)
        if use_self_attn:
            self._self_attention = Attention(
                hidden_dim,
                self_attention_type,
                attention_layers,
                rep="identity",
                dropout=attention_dropout,
            )
        self._penultimate_layer = nn.Linear(hidden_dim, hidden_dim)
        self._mean = nn.Linear(hidden_dim, latent_dim)
        self._log_var = nn.Linear(hidden_dim, latent_dim)
        self._min_std = min_std
        self._use_lvar = use_lvar
        self._use_lstm = use_lstm
        self._use_self_attn = use_self_attn

    def forward(self, x, y):
        encoder_input = torch.cat([x, y], dim=-1)

        # Pass final axis through MLP
        encoded = self._encoder(encoder_input)

        # Aggregator: take the mean over all points
        if self._use_self_attn:
            attention_output = self._self_attention(encoded, encoded, encoded)
            mean_repr = attention_output.mean(dim=1)
        else:
            mean_repr = encoded.mean(dim=1)

        # Have further MLP layers that map to the parameters of the Gaussian latent
        mean_repr = torch.relu(self._penultimate_layer(mean_repr))

        # Then apply further linear layers to output latent mu and log sigma
        mean = self._mean(mean_repr)
        log_var = self._log_var(mean_repr)

        if self._use_lvar:
            # Clip it in the log domain, so it can only approach self.min_std, this helps avoid mode collapase
            # 2 ways, a better but untested way using the more stable log domain, and the way from the deepmind repo
            log_var = F.logsigmoid(log_var)
            log_var = torch.clamp(log_var, np.log(self._min_std), -np.log(self._min_std))
            sigma = torch.exp(0.5 * log_var)
        else:
            sigma = self._min_std + (1 - self._min_std) * torch.sigmoid(log_var * 0.5)
        dist = torch.distributions.Normal(mean, sigma)
        return dist, log_var


class DeterministicEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        x_dim,
        hidden_dim=32,
        n_d_encoder_layers=3,
        self_attention_type="dot",
        cross_attention_type="dot",
        use_self_attn=False,
        attention_layers=2,
        batchnorm=False,
        dropout=0,
        attention_dropout=0,
        use_lstm=False,
    ):
        super().__init__()
        self._use_self_attn = use_self_attn
        # self._input_layer = nn.Linear(input_dim, hidden_dim)
        if use_lstm:
            self._d_encoder = LSTMBlock(input_dim, hidden_dim, batchnorm=batchnorm, dropout=dropout, num_layers=n_d_encoder_layers)
        else:
            self._d_encoder = BatchMLP(input_dim, hidden_dim, batchnorm=batchnorm, dropout=dropout, num_layers=n_d_encoder_layers)
        if use_self_attn:
            self._self_attention = Attention(
                hidden_dim,
                self_attention_type,
                attention_layers,
                rep="identity",
                dropout=attention_dropout,
            )
        self._cross_attention = Attention(
            hidden_dim,
            cross_attention_type,
            x_dim=x_dim,
            attention_layers=attention_layers,
        )

    def forward(self, context_x, context_y, target_x):
        # Concatenate x and y along the filter axes
        d_encoder_input = torch.cat([context_x, context_y], dim=-1)

        # Pass final axis through MLP
        d_encoded = self._d_encoder(d_encoder_input)

        if self._use_self_attn:
            d_encoded = self._self_attention(d_encoded, d_encoded, d_encoded)

        # Apply attention as mean aggregation
        h = self._cross_attention(context_x, d_encoded, target_x)  # [B, T_target, H]

        return h


class Decoder(nn.Module):
    def __init__(
        self,
        x_dim,
        y_dim,
        hidden_dim=32,
        latent_dim=32,
        n_decoder_layers=3,
        use_deterministic_path=True,
        min_std=0.01,
        use_lvar=False,
        batchnorm=False,
        dropout=0,
        use_lstm=False,
    ):
        super(Decoder, self).__init__()
        self._target_transform = nn.Linear(x_dim, hidden_dim)
        if use_deterministic_path:
            hidden_dim_2 = 2 * hidden_dim + latent_dim
        else:
            hidden_dim_2 = hidden_dim + latent_dim

        if use_lstm:
            self._decoder = LSTMBlock(hidden_dim_2, hidden_dim_2, batchnorm=batchnorm, dropout=dropout, num_layers=n_decoder_layers)
        else:
            self._decoder = BatchMLP(hidden_dim_2, hidden_dim_2, batchnorm=batchnorm, dropout=dropout, num_layers=n_decoder_layers)
        self._mean = nn.Linear(hidden_dim_2, y_dim)
        self._std = nn.Linear(hidden_dim_2, y_dim)
        self._use_deterministic_path = use_deterministic_path
        self._min_std = min_std
        self._use_lvar = use_lvar

    def forward(self, r, z, target_x):
        # concatenate target_x and representation
        x = self._target_transform(target_x)

        if self._use_deterministic_path:
            z = torch.cat([r, z], dim=-1)

        r = torch.cat([z, x], dim=-1)

        r = self._decoder(r)

        # Get the mean and the variance
        mean = self._mean(r)
        log_sigma = self._std(r)

        # Bound or clamp the variance
        if self._use_lvar:
            log_sigma = torch.clamp(log_sigma, math.log(self._min_std), -math.log(self._min_std))
            sigma = torch.exp(log_sigma)
        else:
            sigma = self._min_std + (1 - self._min_std) * F.softplus(log_sigma)

        dist = torch.distributions.Normal(mean, sigma)

        return dist, log_sigma


class ParamDecoder(nn.Module):
    def __init__(
        self,
        x_dim,
        n_target,
        hidden_dim=32,
        latent_dim=32,
        n_decoder_layers=3,
        use_deterministic_path=True,
        min_std=0.01,
        use_lvar=False,
        batchnorm=False,
        dropout=0,
        use_lstm=False,
    ):
        super(ParamDecoder, self).__init__()
        # self._target_transform = nn.Linear(x_dim, hidden_dim)
        self._decoder = nn.Sequential(nn.Linear(hidden_dim + latent_dim, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.LayerNorm(hidden_dim))
        self._mean = nn.Linear(hidden_dim, n_target)
        self._std = nn.Linear(hidden_dim, n_target)
        # self._use_deterministic_path = use_deterministic_path
        self._min_std = min_std
        self._use_lvar = use_lvar

    def forward(self, r, z, target_x):
        # concatenate target_x and representation
        # x = self._target_transform(target_x)
        # print("x: ", x.shape)

        # if self._use_deterministic_path:
        #    z = torch.cat([r, z], dim=-1)
        #    print("r: ", r.shape)
        r = torch.mean(r, dim=1)  # [B, H]
        r = torch.cat([z, r], dim=-1)  # [B, L + H]
        r = self._decoder(r)

        # Get the mean and the variance
        mean = self._mean(r)
        log_sigma = self._std(r)

        return mean, log_sigma
