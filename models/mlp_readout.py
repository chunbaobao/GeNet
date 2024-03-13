import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
"""
    MLP Layer used after graph vector representation
"""


class MLPReadout(nn.Module):

    def __init__(self, net_params, L=2):  # L=nb_hidden_layers
        super().__init__()
        input_dim = net_params['out_dim']
        output_dim = net_params['n_classes']
        self.readout = net_params['readout']
        list_FC_layers = [nn.Linear(input_dim//2**l, input_dim//2 **
                                    (l+1), bias=True) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim//2**L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

        # for layer in self.FC_layers:
        #     nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

    def forward(self, g):

        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
        y = hg
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y

    def constrain_loss(self):
        loss = 0
        for layer in self.FC_layers:
            loss += torch.sum(torch.abs(torch.sum(layer.weight, dim=-1)))
        return loss
