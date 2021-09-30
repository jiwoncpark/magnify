import torch
from torch import nn
from torch.distributions import Normal
import torchsde


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, inp):
        out, _ = self.gru(inp)
        out = self.lin(out)
        return out


class ResMLP(nn.Module):
    """Residual MLP that predicts the target quantities

    Attributes
    ----------
    dim_in : int
        Dimension of stochastic latent vector
    dim_out : int
        Number of target quantities to predict, or number of parameters
        defining the posterior PDF over the target quantities
    """
    def __init__(self, dim_in, dim_out, dim_hidden=16):
        super(ResMLP, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.pre_skip = nn.Sequential(nn.Linear(self.dim_in, self.dim_hidden),
                                      nn.Softplus(),
                                      nn.Linear(self.dim_hidden, self.dim_hidden),
                                      nn.Softplus(),
                                      nn.Linear(self.dim_hidden, self.dim_in),
                                      nn.LayerNorm(self.dim_in))
        self.post_skip = nn.Sequential(nn.Linear(self.dim_in, self.dim_hidden),
                                       nn.Softplus(),
                                       nn.Linear(self.dim_hidden, self.dim_out))

    def forward(self, z0):
        # z0 ~ [B, dim_in]
        out = self.pre_skip(z0)  # [B, dim_in]
        out = out + z0  # [B, dim_in], skip connection
        out = self.post_skip(out)  # projector
        return out


class LatentSDE(nn.Module):
    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(self, data_size, latent_size, context_size, hidden_size,
                 n_params, include_prior_drift=False):
        super(LatentSDE, self).__init__()
        self.include_prior_drift = include_prior_drift
        # Encoder.
        self.encoder = Encoder(input_size=data_size, hidden_size=hidden_size,
                               output_size=context_size)
        self.qz0_net = nn.Linear(context_size, latent_size + latent_size)

        # Decoder.
        self.f_net = nn.Sequential(
            nn.Linear(latent_size + context_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )
        self.h_net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )
        # This needs to be an element-wise function for the SDE to satisfy diagonal noise.
        self.g_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, hidden_size),
                    nn.Softplus(),
                    nn.Linear(hidden_size, 1),
                    nn.Sigmoid()
                )
                for _ in range(latent_size)
            ]
        )
        self.projector = nn.Linear(latent_size, data_size)
        if self.include_prior_drift:
            param_mlp_dim_in = latent_size*4 + 3
        else:
            param_mlp_dim_in = latent_size
        self.param_mlp = ResMLP(param_mlp_dim_in, n_params, hidden_size)

        self.pz0_mean = nn.Parameter(torch.zeros(1, latent_size))
        self.pz0_logstd = nn.Parameter(torch.zeros(1, latent_size))

        self._ctx = None

    def contextualize(self, ctx):
        """Set the context vector, which is an encoding of the observed
        sequence

        Parameters
        ----------
        ctx : tuple
            A tuple of tensors of sizes (T,), (T, batch_size, d)
        """
        self._ctx = ctx

    def f(self, t, y):
        """Network that decodes the latent and context
        (posterior drift function)
        Time-inhomogeneous (see paper sec 9.12)

        """
        ts, ctx = self._ctx
        # ts ~ [T]
        # ctx ~ [T, B, context_size]
        ts = ts.to(t.device)
        # searchsorted output: if t were inserted into ts, what would the
        # indices have to be to preserve ordering, assuming ts is sorted
        # training time: t is tensor with no size (scalar)
        # inference time: t ~ [num**2, 1] from the meshgrid
        i = min(torch.searchsorted(ts, t.min(), right=True), len(ts) - 1)
        # training time: y ~ [B, latent_dim]
        # inference time: y ~ [num**2, 1] from the meshgrid
        f_in = torch.cat((y, ctx[i]), dim=1)
        # Training time (for each time step)
        # t ~ []
        # y ~ [B, latent_dim]
        # f_in ~ [B, latent_dim + context_dim]
        f_out = self.f_net(f_in)
        # f_out ~ [B, latent_dim]
        return f_out

    def h(self, t, y):
        """Network that decodes the latent
        (prior drift function)

        """
        return self.h_net(y)

    def g(self, t, y):
        """Network that decodes each time step of the latent
        (diagonal diffusion)

        """
        y = torch.split(y, split_size_or_sections=1, dim=1)
        out = [g_net_i(y_i) for (g_net_i, y_i) in zip(self.g_nets, y)]
        g_out = torch.cat(out, dim=1)  # [B, latent_dim]
        return g_out

    def forward(self, xs, ts, noise_std, adjoint=False, method="euler"):
        # Contextualization is only needed for posterior inference.
        ctx = self.encoder(torch.flip(xs, dims=(0,)))  # [T, B, context_size] = [100, 1024, 64]
        ctx = torch.flip(ctx, dims=(0,))  # revert to original time sequence
        self.contextualize((ts, ctx))

        qz0_mean, qz0_logstd = self.qz0_net(ctx[0]).chunk(chunks=2, dim=1)
        z0 = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)
        # z0 ~ [B, latent_dim] = [1024, 4]

        if adjoint:
            # Must use the argument `adjoint_params`, since `ctx` is not part of the input to `f`, `g`, and `h`.
            adjoint_params = (
                    (ctx,) +
                    tuple(self.f_net.parameters()) + tuple(self.g_nets.parameters()) + tuple(self.h_net.parameters())
            )
            zs, log_ratio = torchsde.sdeint_adjoint(
                self, z0, ts, adjoint_params=adjoint_params, dt=1e-2,
                logqp=True, method=method)
            # zs ~ [T, B, latent_dim] = [100, 1024, 4]
            # log_ratio ~ [T-1, B] = [99, 1024]
        else:
            zs, log_ratio = torchsde.sdeint(self, z0, ts, dt=1e-2, logqp=True,
                                            method=method)
        _xs = self.projector(zs)
        # _xs ~ [T, B, Y_out_dim] = [100, 1024, 3]
        xs_dist = Normal(loc=_xs, scale=noise_std)
        # Sum across times and dimensions, mean across examples in batch
        log_pxs = xs_dist.log_prob(xs).sum(dim=0).mean()  # scalar

        qz0 = torch.distributions.Normal(loc=qz0_mean, scale=qz0_logstd.exp())
        pz0 = torch.distributions.Normal(loc=self.pz0_mean, scale=self.pz0_logstd.exp())
        # Sum over times, mean over batches
        logqp0 = torch.distributions.kl_divergence(qz0, pz0).sum(dim=0).mean(dim=0)
        logqp_path = log_ratio.sum(dim=0).mean(dim=0)

        g = xs[:, :, [-1]].mean(dim=0)  # approx g mag
        r = xs[:, :, [1]].mean(dim=0)  # approx r mag
        gr_std = xs.std(dim=0)  # approx std in g, r
        # Parameter predictions
        if self.include_prior_drift:
            param_pred = self.param_mlp(torch.cat([z0,
                                                  self.h(None, z0),
                                                  self.f(ts, z0),
                                                  self.g(None, z0),
                                                  g-r,
                                                  gr_std],
                                                  dim=-1))
        else:
            param_pred = self.param_mlp(z0)

        return log_pxs, logqp0 + logqp_path, param_pred

    @torch.no_grad()
    def sample_posterior(self, batch_size, xs, ts, ts_eval,
                         bm=None, method='euler'):
        # Contextualization is only needed for posterior inference.
        ctx = self.encoder(torch.flip(xs, dims=(0,)))  # [T, B, context_size] = [100, 1024, 64]
        ctx = torch.flip(ctx, dims=(0,))  # revert to original time sequence
        self.contextualize((ts, ctx))

        qz0_mean, qz0_logstd = self.qz0_net(ctx[0]).chunk(chunks=2, dim=1)
        z0 = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)
        # z0 ~ [B, latent_dim] = [1024, 4]
        zs, log_ratio = torchsde.sdeint(self, z0, ts_eval, dt=1e-2,
                                        logqp=True, method=method, bm=bm)
        _xs = self.projector(zs)
        # _xs ~ [T, B, Y_out_dim] = [100, 1024, 3]
        return _xs

    @torch.no_grad()
    def sample(self, batch_size, ts, bm=None):
        eps = torch.randn(size=(batch_size, *self.pz0_mean.shape[1:]),
                          device=self.pz0_mean.device)
        z0 = self.pz0_mean + self.pz0_logstd.exp() * eps
        zs = torchsde.sdeint(self, z0, ts, names={'drift': 'h'},
                             dt=1e-3, bm=bm)
        # Most of the times in ML, we don't sample the observation noise
        # for visualization purposes.
        _xs = self.projector(zs)
        return _xs
