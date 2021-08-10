import torch
from torch import nn
from magnify.losses.gaussian_nll import FullRankGaussianNLL


class BaselineMLP(nn.Module):
    def __init__(self,
                 X_dim,
                 Y_dim,
                 hidden_dim=32,
                 latent_dim=32,
                 n_decoder_layers=3,
                 dropout=0,
                 device=torch.device('cuda:0')):
        super(BaselineMLP, self).__init__()
        self.hidden_dim = hidden_dim
        self.X_dim = X_dim
        self.Y_dim = Y_dim
        self.n_decoder_layers = n_decoder_layers
        self.dropout = dropout
        self.loss = FullRankGaussianNLL(Y_dim, device=device)
        self.out_dim = self.loss.out_dim
        input_dim = self.X_dim
        layers = nn.ModuleList()
        for i in range(self.n_decoder_layers - 1):
            mlp_layer = nn.Sequential(nn.Linear(input_dim, self.hidden_dim),
                                      nn.ReLU(),
                                      nn.LayerNorm(self.hidden_dim),
                                      nn.Dropout(self.dropout))
            layers.append(mlp_layer)
            # Input dim changes after the first layer
            input_dim = self.hidden_dim
        layers.append(nn.Sequential(nn.Linear(input_dim, self.hidden_dim),
                                    nn.ReLU(),
                                    nn.Dropout(self.dropout),
                                    nn.Linear(input_dim, self.out_dim)
                                    ))
        self.layers = layers

    def forward(self, x, target):
        out = self.layers[0](x)
        for i in range(1, self.n_decoder_layers-1):
            out = out + self.layers[i](out)
        out = self.layers[-1](out)
        loss = self.loss(out, target)
        return out, loss


if __name__ == '__main__':
    model = BaselineMLP(12, 21, hidden_dim=128, n_decoder_layers=6, device=torch.device('cpu'))
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params: ", total_params)
    dummy_x = torch.zeros((10, 12))
    dummy_target = torch.zeros((10, 21))
    out, loss = model(dummy_x, dummy_target)
    print(out.shape)  # should be [batch_size, out_dim]
    print(loss.shape)  # should be [batch_size,]

