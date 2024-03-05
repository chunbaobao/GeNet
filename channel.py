import torch
import torch.nn as nn

class Channel(nn.Module):
    def __init__(self, channel_type, snr):
        super(Channel, self).__init__()
        self.channel_type = channel_type
        self.snr = snr

    def forward(self, z_hat: torch.Tensor):
        if self.channel_type == 'AWGN':
            if z_hat.dim() == 4:
                # k = np.prod(z_hat.size()[1:])
                k = torch.prod(torch.tensor(z_hat.size()[1:]))
                sig_pwr = torch.sum(torch.abs(z_hat).square(), dim=(1, 2, 3), keepdim=True)/k
            elif z_hat.dim() == 3:
                # k = np.prod(z_hat.size())
                k = torch.prod(torch.tensor(z_hat.size()))
                sig_pwr = torch.sum(torch.abs(z_hat).square())/k
            noi_pwr = sig_pwr / ( 10 ** (self.snr / 10))
            noise = torch.randn_like(z_hat) * torch.sqrt(noi_pwr)
            return z_hat + noise
        elif self.channel_type == 'Rayleigh':
            pass
        else:
            raise Exception('Unknown type of channel')
