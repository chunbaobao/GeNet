import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import GATConv

"""
    GAT: Graph Attention Network
    Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""


class GATLayer(nn.Module):
    """
    Parameters
    ----------
    in_dim : 
        Number of input features.
    out_dim : 
        Number of output features.
    num_heads : int
        Number of heads in Multi-Head Attention.
    dropout :
        Required for dropout of attn and feat in GATConv
    batch_norm :
        boolean flag for batch_norm layer.
    residual : 
        If True, use residual connection inside this layer. Default: ``False``.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.

    Using dgl builtin GATConv by default:
    https://github.com/graphdeeplearning/benchmarking-gnns/commit/206e888ecc0f8d941c54e061d5dffcc7ae2142fc
    """

    def __init__(self, in_dim, out_dim, num_heads, dropout, batch_norm, residual=False, activation=F.elu):
        super().__init__()
        self.residual = residual
        self.activation = activation
        self.batch_norm = batch_norm

        if in_dim != (out_dim*num_heads):
            self.residual = False

        if dgl.__version__ < "0.5":
            self.gatconv = GATConv(in_dim, out_dim, num_heads, dropout, dropout)
        else:
            self.gatconv = GATConv(in_dim, out_dim, num_heads, dropout,
                                   dropout, allow_zero_in_degree=True)

        if self.batch_norm:
            self.batchnorm_h = nn.BatchNorm1d(out_dim * num_heads)

    def forward(self, g, h):
        h_in = h  # for residual connection

        h = self.gatconv(g, h).flatten(1)

        if self.batch_norm:
            h = self.batchnorm_h(h)

        if self.activation:
            h = self.activation(h)

        if self.residual:
            h = h_in + h  # residual connection

        return h


"""
    GAT: Graph Attention Network
    Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""


class GATNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        num_heads = net_params['n_heads']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.dropout = dropout

        self.embedding_h = nn.Linear(in_dim, hidden_dim * num_heads)

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList([GATLayer(hidden_dim * num_heads, hidden_dim, num_heads,
                                              dropout, self.batch_norm, self.residual) for _ in range(n_layers-1)])
        self.layers.append(GATLayer(hidden_dim * num_heads, out_dim,
                           1, dropout, self.batch_norm, self.residual))
        # self.MLP_layer = MLPReadout(out_dim, n_classes)

    def forward(self, g, h, e):
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        for conv in self.layers:
            h = conv(g, h)
        g.ndata['h'] = h

        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes

        return hg
        # return self.MLP_layer(hg)
