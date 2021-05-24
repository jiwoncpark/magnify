"""Taken with minor modifications from

https://github.com/3springs/attentive-neural-processes/blob/af431a267bad309b2d5698f25551986e2c4e7815/neural_processes/models/neural_process/model.py

"""

import torch
from torch import nn
import torch.nn.functional as F
from magnify.losses.gaussian_nll import DoubleGaussianNLL
from magnify.attentive_neural_process.modules.layers import BatchNormSequence
from magnify.attentive_neural_process.models import (LatentEncoder,
                                                     DeterministicEncoder,
                                                     Decoder, ParamDecoder)
from magnify.attentive_neural_process.utils import kl_loss_var, log_prob_sigma
from magnify.attentive_neural_process.utils import hparams_power


class NeuralProcess(nn.Module):

    @staticmethod
    def FROM_HPARAMS(hparams):
        hparams = hparams_power(hparams)
        return NeuralProcess(**hparams)

    def __init__(self,
                 x_dim=1,  # features in input
                 y_dim=1,  # number of features in output
                 n_target=1,
                 hidden_dim=32,  # size of hidden space
                 latent_dim=32,  # size of latent space
                 # type of attention: "uniform", "dot", "multihead" "ptmultihead":
                 # see attentive neural processes paper
                 latent_enc_self_attn_type="ptmultihead",
                 det_enc_self_attn_type="ptmultihead",
                 det_enc_cross_attn_type="ptmultihead",
                 n_latent_encoder_layers=2,
                 n_det_encoder_layers=2,  # number of deterministic encoder layers
                 n_decoder_layers=2,
                 use_deterministic_path=True,
                 weight_y_loss=1.0,
                 # To avoid collapse use a minimum standard deviation,
                 # should be much smaller than variation in labels
                 min_std=0.001,
                 dropout=0,
                 use_self_attn=False,
                 attention_dropout=0,
                 batchnorm=False,
                 use_lvar=False,  # Alternative loss calculation, may be more stable
                 attention_layers=2,
                 use_rnn=False,  # use RNN/LSTM?
                 use_lstm_le=False,  # use another LSTM in latent encoder instead of MLP
                 use_lstm_de=False,  # use another LSTM in determinstic encoder instead of MLP
                 use_lstm_d=False,  # use another lstm in decoder instead of MLP
                 context_in_target=False,
                 **kwargs,
                 ):

        super(NeuralProcess, self).__init__()

        self._use_rnn = use_rnn
        self.context_in_target = context_in_target

        # Sometimes input normalisation can be important,
        # an initial batch norm is a nice way to ensure this
        # https://stackoverflow.com/a/46772183/221742
        # self.norm_x = BatchNormSequence(x_dim, affine=False)
        # self.norm_y = BatchNormSequence(y_dim, affine=False)

        if self._use_rnn:
            self._lstm_x = nn.LSTM(
                input_size=x_dim,
                hidden_size=hidden_dim,
                num_layers=attention_layers,
                dropout=dropout,
                batch_first=True
            )
            self._lstm_y = nn.LSTM(
                input_size=y_dim,
                hidden_size=hidden_dim,
                num_layers=attention_layers,
                dropout=dropout,
                batch_first=True
            )
            x_dim = hidden_dim
            y_dim2 = hidden_dim
        else:
            y_dim2 = y_dim

        self._latent_encoder = LatentEncoder(
            x_dim + y_dim2,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            self_attention_type=latent_enc_self_attn_type,
            n_encoder_layers=n_latent_encoder_layers,
            attention_layers=attention_layers,
            dropout=dropout,
            use_self_attn=use_self_attn,
            attention_dropout=attention_dropout,
            batchnorm=batchnorm,
            min_std=min_std,
            use_lvar=use_lvar,
            use_lstm=use_lstm_le,
        )

        self._deterministic_encoder = DeterministicEncoder(
            input_dim=x_dim + y_dim2,
            x_dim=x_dim,
            hidden_dim=hidden_dim,
            self_attention_type=det_enc_self_attn_type,
            cross_attention_type=det_enc_cross_attn_type,
            n_d_encoder_layers=n_det_encoder_layers,
            attention_layers=attention_layers,
            use_self_attn=use_self_attn,
            dropout=dropout,
            batchnorm=batchnorm,
            attention_dropout=attention_dropout,
            use_lstm=use_lstm_de,
        )

        self._decoder = Decoder(
            x_dim,
            y_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
            batchnorm=batchnorm,
            min_std=min_std,
            use_lvar=use_lvar,
            n_decoder_layers=n_decoder_layers,
            use_deterministic_path=use_deterministic_path,
            use_lstm=use_lstm_d,

        )
        self._use_deterministic_path = use_deterministic_path
        self._use_lvar = use_lvar

        self._param_loss = DoubleGaussianNLL(n_target, device=torch.device('cuda:0'))

        self._param_decoder = ParamDecoder(x_dim,
                                           n_target=self._param_loss.out_dim,
                                           hidden_dim=hidden_dim,
                                           latent_dim=latent_dim,
                                           dropout=dropout,
                                           batchnorm=batchnorm,
                                           min_std=min_std,
                                           use_lvar=use_lvar,
                                           n_decoder_layers=n_decoder_layers,
                                           use_deterministic_path=use_deterministic_path,
                                           use_lstm=use_lstm_d
                                           )
        self._weight_y_loss = weight_y_loss

    def forward(self, context_x, context_y, target_x, target_y=None,
                target_meta=None, sample_latent=None,):
        if sample_latent is None:
            sample_latent = self.training

        #device = next(self.parameters()).device

        # if self.hparams.get('bnorm_inputs', True):
        # https://stackoverflow.com/a/46772183/221742
        # target_x = self.norm_x(target_x)
        # context_x = self.norm_x(context_x)
        # context_y = self.norm_y(context_y)

        if self._use_rnn:
            # see https://arxiv.org/abs/1910.09323 where x is substituted with h = RNN(x)
            # x need to be provided as [B, T, H]
            target_x, _ = self._lstm_x(target_x)
            context_x, _ = self._lstm_x(context_x)
            context_y, _ = self._lstm_y(context_y)

        dist_prior, log_var_prior = self._latent_encoder(context_x, context_y)

        if (target_y is not None):
            #target_y2 = self.norm_y(target_y)
            if self._use_rnn:
                target_y2, _ = self._lstm_y(target_y)
            dist_post, log_var_post = self._latent_encoder(target_x, target_y)
            if self.training:
                z = dist_post.rsample() if sample_latent else dist_post.loc
            else:
                z = dist_prior.rsample() if sample_latent else dist_prior.loc
        else:
            z = dist_prior.rsample() if sample_latent else dist_prior.loc

        num_targets = target_x.size(1)
        z_repeated = z.unsqueeze(1).repeat(1, num_targets, 1)  # [B, T_target, H]

        if self._use_deterministic_path:
            r = self._deterministic_encoder(context_x, context_y,
                                            target_x)  # [B, T_target, H]
        else:
            r = None

        dist, log_sigma = self._decoder(r, z_repeated, target_x)
        pred_meta = self._param_decoder(r, z, target_x)

        if target_y is not None:
            if self._use_lvar:
                log_p = log_prob_sigma(target_y, dist.loc, log_sigma).mean(-1)  # [B, T_target, Y].mean(-1)
                if self.context_in_target:
                    log_p[:, :context_x.size(1)] /= 100
                loss_kl = kl_loss_var(dist_prior.loc, log_var_prior,
                                      dist_post.loc, log_var_post).mean(-1)  # [B, R].mean(-1)
            else:  # default method
                log_p = dist.log_prob(target_y).mean(-1).mean(-1)  # [B, T_target, Y].mean(-1).mean(-1)
                if self.context_in_target:
                    # There's the temptation for it to fit only on context,
                    # where it knows the answer, and learn very low uncertainty.
                    log_p[:, :context_x.size(1)] /= 100
                loss_kl = torch.distributions.kl_divergence(
                    dist_post, dist_prior).mean(-1)  # [B, R].mean(-1)
            # Components of time series regression loss
            mse_loss = F.mse_loss(dist.loc, target_y, reduction='none').mean()
            loss_p = -log_p
            # For meta parameter regression
            loss_meta = self._param_loss(pred_meta, target_meta)
            # Final loss
            loss = ((loss_kl + loss_p)*self._weight_y_loss + loss_meta).mean()
            # Weight loss nearer to prediction time?
            #weight = (torch.arange(loss_p.shape[1]) + 1).float().to(device)[None, :]
            #loss_p_weighted = loss_p / torch.sqrt(weight)  # We want to  weight nearer stuff more
            #loss_p_weighted = loss_p_weighted.mean()
            loss_kl = loss_kl.mean()
            loss_p = loss_p.mean()
            loss_meta = loss_meta.mean()

        else:
            loss_p = None
            mse_loss = None
            loss_kl = None
            loss = None
            #loss_p_weighted = None
            loss_meta = None

        y_pred = dist.rsample() if self.training else dist.loc
        return (y_pred,
                dict(loss=loss, loss_p=loss_p, loss_kl=loss_kl, loss_mse=mse_loss,
                     #loss_p_weighted=loss_p_weighted,
                     loss_meta=loss_meta
                     ),
                dict(log_sigma=log_sigma, y_dist=dist)
                     #log_sigma_meta=log_sigma_meta,
                     #mean_meta=mean_meta
                )
