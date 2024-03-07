import torch
import torch.nn as nn


class Channel(nn.Module):
    def __init__(self, snr):
        super(Channel, self).__init__()
        self.snr = snr

    def forward(self, z_hat: torch.Tensor):
        pass

        # return z_hat + noise
