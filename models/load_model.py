"""
    Utility file to select GraphNN model as
    selected by the user
"""
from models.gatedgcn import GatedGCNNet
from models.gcn import GCNNet
from models.gat import GATNet
from models.mlp import MLPNet
from models.mlp_readout import MLPReadout
import torch.nn as nn
import channel
import torch

def GatedGCN(net_params):
    return GatedGCNNet(net_params)


def GCN(net_params):
    return GCNNet(net_params)


def GAT(net_params):
    return GATNet(net_params)


def MLP(net_params):
    return MLPNet(net_params)


class GeNet(nn.Module):
    def __init__(self, model_name, net_params, snr=None):
        super(GeNet, self).__init__()
        self.encoder = self.gnn_model(model_name, net_params)
        if snr is not None:
            self.channel = channel.Channel(snr)
        self.decoder = MLPReadout(net_params)

    @staticmethod
    def gnn_model(MODEL_NAME, net_params):
        models = {
            'GatedGCN': GatedGCN,
            'GCN': GCN,
            'GAT': GAT,
            'MLP': MLP
        }
        return models[MODEL_NAME](net_params)

    def set_channel(self, snr):
        self.channel = channel.Channel(snr)

    def forward(self, g, h, e):
        g = self.encoder(g, h, e)
        if hasattr(self, 'channel'):
            g = self.channel(g)
        hg = self.decoder(g)
        return hg

    def loss(self, pred, label, is_constrain = False, k = 0.1):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        if is_constrain:
            loss = loss + self.decoder.constrain_loss() * k
        return loss
