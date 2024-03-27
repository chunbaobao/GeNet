import torch
import torch.nn as nn
import dgl


class Channel(nn.Module):
    def __init__(self, snr):
        super(Channel, self).__init__()
        self.snr = snr

    def forward(self, graph_or_tensor):

        if isinstance(graph_or_tensor, dgl.DGLGraph):
            # z_hat : (num_nodes, feature_dim)
            z_hat = graph_or_tensor.ndata['h']
            k = torch.prod(torch.tensor(z_hat.shape)) / graph_or_tensor.batch_size
            sig_pwr = torch.sum(torch.abs(z_hat).square(), dim=1, keepdim=True) / k
            noi_pwr = sig_pwr / (10 ** (self.snr / 10))
            noise = torch.randn_like(z_hat) * torch.sqrt(noi_pwr)
            graph_or_tensor.ndata['h'] = z_hat + noise
            return graph_or_tensor
        elif isinstance(graph_or_tensor, torch.Tensor):
            # z_hat : (batch_size, feature)
            z_hat = graph_or_tensor
            k = torch.prod(torch.tensor(z_hat.shape[1:]))
            sig_pwr = torch.sum(torch.abs(z_hat).square(), dim=1, keepdim=True) / k
            noi_pwr = sig_pwr / (10 ** (self.snr / 10))
            noise = torch.randn_like(z_hat) * torch.sqrt(noi_pwr)
            return z_hat + noise
        else:
            raise Exception('Unknown Type: {}'.format(type(graph_or_tensor)))

        # return z_hat + noise


if __name__ == '__main__':
    # test
    import dgl
    import torch
    from dgl.data import MiniGCDataset
    from dgl.dataloading import GraphDataLoader

    dataset = MiniGCDataset(80, 10, 20)
    dataloader = GraphDataLoader(dataset, batch_size=5, shuffle=True)

    for batched_graph, labels in dataloader:
        print(batched_graph)
        print(labels)
        break

    snr = 10
    channel = Channel(snr)
    for batched_graph, labels in dataloader:
        batched_graph.ndata['h'] = torch.randn(batched_graph.num_nodes(), 5)
        print(batched_graph)

        batched_graph = channel(batched_graph)
        print(batched_graph)
        break

    # test
    x = torch.randn(32, 10, 3)
    channel = Channel(snr)
    print(x)
    x = channel(x)
    print(x)
    print('done')
