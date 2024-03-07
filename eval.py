import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from utils import view_model_param, gpu_setup ,set_seed
from torch.nn.parallel import DataParallel
from prepare_dataset import SuperPixDataset
import numpy as np
from channel import Channel
from prepare_dataset import process_image



class RotatedTestDateset:
    def __init__(self,
                 dataset_dir,
                 dataset_name,
                 use_mean_px=True):
        self.dataset_name = dataset_name
        self.graph_lists = []
        

        if dataset_name == 'fashionmnist':
            self.img_size = 28
            n_sp = 95
            compactness = .25
            dataset = datasets.FashionMNIST(root=dataset_dir, train=True, download=False)

        elif dataset_name == 'cifar10':
            n_sp = 150
            compactness = 10
            self.img_size = 32
            dataset = datasets.CIFAR10(root=dataset_dir, train=True, download=False)

        elif dataset_name == 'mnist':
            self.img_size = 28
            n_sp = 95
            compactness = .25
            dataset = datasets.MNIST(root=dataset_dir, train=True, download=False)

        else:
            raise Exception("Unknown dataset")   
        print("processing %s dataset to superpixels using slic algorithm..." % (dataset_name))
        
        
        
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
            