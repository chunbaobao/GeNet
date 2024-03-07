"""
    Utility file to select GraphNN model as
    selected by the user
"""
from gatedgcn import GatedGCNNet
from gcn import GCNNet
from gat import GATNet
from mlp import MLPNet
from mlp_readout import MLPReadout
import torch.nn as nn
import channel



def GatedGCN(net_params):
    return GatedGCNNet(net_params)

def GCN(net_params):
    return GCNNet(net_params)

def GAT(net_params):
    return GATNet(net_params)


def MLP(net_params):
    return MLPNet(net_params)



class GeNet(nn.Module):
    def __init__(self,model_name, net_params,snr = None):
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
        
    def forward(self, x):
        x = self.encoder(x)
        if hasattr(self, 'channel'):
            x = self.channel(x)
        x = self.decoder(x)
        return x
    
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss