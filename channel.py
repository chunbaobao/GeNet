import torch
import torch.nn as nn


class Channel(nn.Module):
    def __init__(self, snr):
        super(Channel, self).__init__()
        self.snr = snr

    def forward(self, z_hat: torch.Tensor):
        # z_hat : (batch_size, num_nodes, feature_dim)
        k = torch.prod(z_hat.shape[1:])
        sig_pwr = torch.sum(torch.abs(z_hat).square(), dim=1, keepdim=True) / k
        noi_pwr = sig_pwr / ( 10 ** (self.snr / 10))
        noise = torch.randn_like(z_hat) * torch.sqrt(noi_pwr)
        return z_hat + noise


        # return z_hat + noise
