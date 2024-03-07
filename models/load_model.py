"""
    Utility file to select GraphNN model as
    selected by the user
"""
from gatedgcn import GatedGCNNet
from gcn import GCNNet
from gat import GATNet
from mlp import MLPNet



def GatedGCN(net_params):
    return GatedGCNNet(net_params)

def GCN(net_params):
    return GCNNet(net_params)

def GAT(net_params):
    return GATNet(net_params)


def MLP(net_params):
    return MLPNet(net_params)




def gnn_model(MODEL_NAME, net_params):
    models = {
        'GatedGCN': GatedGCN,
        'GCN': GCN,
        'GAT': GAT,
        'MLP': MLP
    }
        
    return models[MODEL_NAME](net_params)