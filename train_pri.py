# -*- coding: utf-8 -*-
"""
Created on Tue Mar  17:00:00 2024

@author: chun
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from utils import view_model_param, gpu_setup ,set_seed
from torch.nn.parallel import DataParallel
from prepare_dataset import SuperPixDataset
import numpy as np
from train import train_epoch_sparse as train_epoch, evaluate_network_sparse as evaluate_network

def config_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='./out', type=str, help='path of output')
    parser.add_argument('--dataset_dir',default='./data', type=str, help='path of dataset')

    return parser.parse_args()


def main():
    models = ['gcn', 'gat', 'gatedgnc', 'mlp']
    datasets = ['mnist','cifar10','fashionmnist']
    args = config_parser()
    
    
    
    for model in models:
        for dataset in datasets:
            
        




if __name__ == '__main__': 
    main()
    print('Done!')