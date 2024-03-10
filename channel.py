import torch
import torch.nn as nn
import dgl

class Channel(nn.Module):
    def __init__(self, snr):
        super(Channel, self).__init__()
        self.snr = snr

    def forward(self, g: dgl.DGLGraph):
        # z_hat : (batch_size, num_nodes, feature_dim)
        z_hat = g.ndata['h']
        k = torch.prod(z_hat.shape[1:])
        sig_pwr = torch.sum(torch.abs(z_hat).square(), dim=1, keepdim=True) / k
        noi_pwr = sig_pwr / ( 10 ** (self.snr / 10))
        noise = torch.randn_like(z_hat) * torch.sqrt(noi_pwr)
        g.ndata['h'] = z_hat + noise
        return g


        # return z_hat + noise
