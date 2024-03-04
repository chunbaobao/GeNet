# Prepare superpixels dataset for image dataset using SLIC algorithm
# https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic
# this script refers to https://github.com/graphdeeplearning/benchmarking-gnns/data/superpixels
# and https://github.com/bknyaz/graph_attention_pool/extract_superpixels.py

import numpy as np
import torch
import pickle
import time
import os
import os
import scipy
import pickle
from skimage.segmentation import slic
from torchvision import datasets
import multiprocessing as mp
import scipy.ndimage
import scipy.spatial
import argparse
import dgl
from scipy.spatial.distance import cdist
from utils import split_dataset


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_file_exists(path, dataset: str):
    if os.path.isfile(os.path.join(path, dataset+'.pkl')):
        print("{} superpixels already extracted".format(dataset))
        return True
    else:
        print("Extracting superpixels for {}".format(dataset))
        return False


def sigma(dists, kth=8):
    # Compute sigma and reshape
    try:
        # Get k-nearest neighbors for each node
        # ! kth+1 nearest neighbors because kth contain the node itself
        knns = np.partition(dists, kth, axis=-1)[:, kth::-1]
        sigma = knns.sum(axis=1).reshape((knns.shape[0], 1))/kth
    except ValueError:     # handling for graphs with num_nodes less than kth
        num_nodes = dists.shape[0]
        # this sigma value is irrelevant since not used for final compute_edge_list
        sigma = np.array([1]*num_nodes).reshape(num_nodes, 1)

    return sigma + 1e-8  # adding epsilon to avoid zero value of sigma


def compute_adjacency_matrix_images(coord, feat, use_feat=True, kth=8):
    coord = coord.reshape(-1, 2)
    # Compute coordinate distance
    c_dist = cdist(coord, coord)  # * euclidean distance: ((x1-x2)^2 + (y1-y2)^2)^0.5

    if use_feat:
        # Compute feature distance
        f_dist = cdist(feat, feat)
        # Compute adjacency
        A = np.exp(- (c_dist/sigma(c_dist))**2 - (f_dist/sigma(f_dist))**2)
    else:
        A = np.exp(- (c_dist/sigma(c_dist))**2)

    # Convert to symmetric matrix
    A = 0.5 * (A + A.T)
    A[np.diag_indices_from(A)] = 0
    return A


def compute_edges_list(A, kth=8+1):
    # Get k-similar neighbor indices for each node

    num_nodes = A.shape[0]
    new_kth = num_nodes - kth

    if num_nodes > 9:
        knns = np.argpartition(A, new_kth-1, axis=-1)[:, new_kth:-1]
        knn_values = np.partition(A, new_kth-1, axis=-1)[:, new_kth:-1]  # NEW
    else:
        # handling for graphs with less than kth nodes
        # in such cases, the resulting graph will be fully connected
        knns = np.tile(np.arange(num_nodes), num_nodes).reshape(num_nodes, num_nodes)
        knn_values = A  # NEW

        # removing self loop
        if num_nodes != 1:
            knn_values = A[knns != np.arange(num_nodes)[:, None]].reshape(num_nodes, -1)  # NEW
            knns = knns[knns != np.arange(num_nodes)[:, None]].reshape(num_nodes, -1)
    return knns, knn_values  # NEW


class DGLFormDataset(torch.utils.data.Dataset):
    """
        DGLFormDataset wrapping graph list and label list as per pytorch Dataset.
        *lists (list): lists of 'graphs' and 'labels' with same len().
    """

    def __init__(self, *lists):
        assert all(len(lists[0]) == len(li) for li in lists)
        self.lists = lists
        self.graph_lists = lists[0]
        self.graph_labels = lists[1]

    def __getitem__(self, index):
        return tuple(li[index] for li in self.lists)

    def __len__(self):
        return len(self.lists[0])


# refer to benchmarking-gnns
def self_loop(g):
    """
        Utility function only, to be used only when necessary as per user self_loop flag
        : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']


        This function is called inside a function in SuperPixDataset class.
    """
    new_g = dgl.graph([])
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata['feat'] = g.ndata['feat']

    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)

    # This new edata is not used since this function gets called only for GCN, GAT
    # However, we need this for the generic requirement of ndata and edata
    new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())
    return new_g


# ! need to
class SuperPixDataset(torch.utils.data.Dataset):  # load from pkl file

    def __init__(self, name):
        """
            Loading Superpixels datasets
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        data_dir = 'data'
        with open(data_dir+name+'.pkl', "rb") as f:
            f = pickle.load(f)
            self.train = f[0]
            self.val = f[1]
        print('train, test, val sizes :', len(self.train), len(self.test), len(self.val))
        print("[I] Data load time: {:.4f}s".format(time.time()-start))

    # form a mini batch from a given list of samples = [(graph, label) pairs]

    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels))
        #tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        #tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        #snorm_n = torch.cat(tab_snorm_n).sqrt()
        #tab_sizes_e = [ graphs[i].number_of_edges() for i in range(len(graphs))]
        #tab_snorm_e = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_e ]
        #snorm_e = torch.cat(tab_snorm_e).sqrt()
        for idx, graph in enumerate(graphs):
            graphs[idx].ndata['feat'] = graph.ndata['feat'].float()
            graphs[idx].edata['feat'] = graph.edata['feat'].float()
        batched_graph = dgl.batch(graphs)

        return batched_graph, labels


def config_parser():
    parser = argparse.ArgumentParser(description='Extract SLIC superpixels from images')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'minst', 'fashionmnist', 'cifar10'])
    parser.add_argument('--data_dir', type=str, default='../dataset', help='path to the dataset')
    parser.add_argument('--out_dir', type=str, default='./data',
                        help='path where to save superpixels')
    parser.add_argument('--seed', type=int, default=100, help='seed for shuffling nodes')
    args = parser.parse_args()

    return args


# refers to extract_superpixels.py
def process_image(params):

    img, n_sp, compactness, shuffle = params

    assert img.dtype == np.uint8, img.dtype
    img = (img / 255.).astype(np.float32)

    channel_axis = 2 if len(img.shape) > 2 else None

    # number of actually extracted superpixels (can be different from requested in SLIC)
    # number of superpixels we ask to extract (larger to extract more superpixels - closer to the desired n_sp)
    superpixels = slic(img, n_segments=n_sp, compactness=compactness,
                           channel_axis=channel_axis, start_label=0)
    sp_indices = np.unique(superpixels)
    n_sp_extracted = len(sp_indices)

    assert n_sp_extracted == np.max(superpixels) + 1, ('superpixel indices', np.unique(superpixels))

    if shuffle:
        ind = np.random.permutation(n_sp_extracted)
    else:
        ind = np.arange(n_sp_extracted)

    sp_order = sp_indices[ind].astype(np.int32)
    if len(img.shape) == 2:
        img = img[:, :, None]  # upward dimension

    n_ch = 1 if img.shape[2] == 1 else 3

    sp_intensity, sp_coord = [], []
    for seg in sp_order:
        mask = (superpixels == seg).squeeze()
        avg_value = np.zeros(n_ch)
        for c in range(n_ch):
            avg_value[c] = np.mean(img[:, :, c][mask])
        cntr = np.array(scipy.ndimage.center_of_mass(mask))  # row, col
        sp_intensity.append(avg_value)  # ! avg color (1 dim for gray, 3 dim for RGB)
        sp_coord.append(cntr)  # ! avg position
    sp_intensity = np.array(sp_intensity, np.float32)
    sp_coord = np.array(sp_coord, np.float32)
    

    return sp_intensity, sp_coord, sp_order, superpixels


class Image2Graph(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_dir,
                 out_dir,
                 dataset_name,
                 valid_split= 0.1,
                 use_mean_px=True,
                 use_coord=True):
        
        
        self.dataset_name = dataset_name
        self.graph_lists = []
        self.valid_split = valid_split
        self.out_dir = out_dir
        print("process %s dataset to superpixels using slic algorithm" % (dataset_name))
        if dataset_name == 'fashionmnist':
            self.img_size = 28
            n_sp = 95
            compactness = .25
            dataset = datasets.FashionMNIST(root=dataset_dir, train=True, download=False)
            images = dataset.data.numpy()
            labels = dataset.targets
            n_images = len(dataset)
            with mp.Pool() as pool:
                self.sp_data = pool.map(
                    process_image, [(images[i], n_sp, compactness, True) for i in range(n_images)])
            self.graph_labels = torch.LongTensor(labels)
        elif dataset_name == 'cifar10':
            n_sp = 150
            compactness = 10
            self.img_size = 32
            dataset = datasets.CIFAR10(root=dataset_dir, train=True, download=False)
            images = dataset.data.numpy()
            labels = dataset.targets
            n_images = len(dataset)
            with mp.Pool() as pool:
                self.sp_data = pool.map(
                    process_image, [(images[i], n_sp, compactness, True) for i in range(n_images)])
            self.graph_labels = torch.LongTensor(self.labels)
        elif dataset_name == 'mnist':
            self.img_size = 28
            n_sp = 95
            compactness = .25
            dataset = datasets.MNIST(root=dataset_dir, train=True, download=False)
            images = dataset.data.numpy()
            labels = dataset.targets
            n_images = len(dataset)
            with mp.Pool() as pool:
                self.sp_data = pool.map(
                    process_image, [(images[i], n_sp, compactness, True) for i in range(n_images)])
            self.graph_labels = torch.LongTensor(labels)
        else:
            raise Exception("Unknown dataset")
        self.use_mean_px = use_mean_px
        self.use_coord = use_coord
        self.n_samples = len(self.labels)

        print("preparing %d graphs for the %s date..." % (self.n_samples, dataset_name))
        self._prepare()

    def _prepare(self):
        self.Adj_matrices, self.node_features, self.edges_lists, self.edge_features = [], [], [], []
        for index, sample in enumerate(self.sp_data):
            mean_px, coord = sample[:2]

            try:
                coord = coord / self.img_size
            except AttributeError:
                VOC_has_variable_image_sizes = True

            if self.use_mean_px:
                # using super-pixel locations + features
                A = compute_adjacency_matrix_images(coord, mean_px)
            else:
                # using only super-pixel locations
                A = compute_adjacency_matrix_images(coord, mean_px, False)
            edges_list, edge_values_list = compute_edges_list(A)  # NEW

            N_nodes = A.shape[0]

            mean_px = mean_px.reshape(N_nodes, -1)
            coord = coord.reshape(N_nodes, 2)
            # ! 3 dimension feat for (coord and position)
            x = np.concatenate((mean_px, coord), axis=1)

            edge_values_list = edge_values_list.reshape(-1)  # NEW # TO DOUBLE-CHECK !

            self.node_features.append(x)
            self.edge_features.append(edge_values_list)  # NEW
            self.Adj_matrices.append(A)
            self.edges_lists.append(edges_list)

        for index in range(len(self.sp_data)):
            g = dgl.graph([])  # ? dgl.DGLGraph() is replaced # NEW
            g.add_nodes(self.node_features[index].shape[0])
            g.ndata['feat'] = torch.Tensor(self.node_features[index]).half()

            for src, dsts in enumerate(self.edges_lists[index]):
                # handling for 1 node where the self loop would be the only edge
                # since, VOC Superpixels has few samples (5 samples) with only 1 node
                if self.node_features[index].shape[0] == 1:
                    g.add_edges(src, dsts)
                else:
                    g.add_edges(src, dsts[dsts != src])

            # adding edge features for Residual Gated ConvNet
            edge_feat_dim = g.ndata['feat'].shape[1]  # dim same as node feature dim
            #g.edata['feat'] = torch.ones(g.number_of_edges(), edge_feat_dim).half()
            g.edata['feat'] = torch.Tensor(self.edge_features[index]).unsqueeze(1).half()  # NEW

            self.graph_lists.append(g)

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]

    def split_dataset(self):
        valid_split = self.valid_split
        train_idx, valid_idx = split_dataset(self.graph_labels, valid_split)
        self.train = DGLFormDataset([self.graph_lists[i] for i in train_idx], [self.graph_labels[i] for i in train_idx])
        self.valid = DGLFormDataset([self.graph_lists[i] for i in valid_idx], [self.graph_labels[i] for i in valid_idx])
        

    def creat_pkl(self):
        self.split_dataset()
        print("saving dataset to %s" % self.dataset_name+".pkl")
        with open(self.out_dir+self.name+'.pkl', 'wb') as f:
            pickle.dump([self.train, self.valid], f)


def main():
    args = config_parser()
    print(args)
    set_seed(args.seed)

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    if args.dataset == 'fashionmnist' or args.dataset == 'all':
        name = 'fashionmnist'
        if not check_file_exists(args.out_dir, name):

            image2graph = Image2Graph(args.data_dir, args.out_dir, name, 0.1)
            image2graph.creat_pkl()
            
    if args.dataset == 'cifar10' or args.dataset == 'all':
        name = 'cifar10'
        if not check_file_exists(args.out_dir, name):
            image2graph = Image2Graph(args.data_dir, args.out_dir, name, 0.1)
            image2graph.creat_pkl()
            
    if args.datase == 'mnist' or args.dataset == 'all':
        name = 'mnist'
        if not check_file_exists(args.out_dir, name):
            image2graph = Image2Graph(args.data_dir, args.out_dir, name, 0.1)
            image2graph.creat_pkl()

if __name__ == '__main__':
    main()