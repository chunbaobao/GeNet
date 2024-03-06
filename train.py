# -*- coding: utf-8 -*-
"""
Created on Tue Mar  17:00:00 2024

@author: chun
"""
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



def config_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--channel', default='AWGN', type=str,
                        choices=['AWGN', 'Rayleigh'], help='channel type')
    parser.add_argument('--out', default='./out', type=str, help='path of output')
    parser.add_argument('--dataset_dir',default='./data', type=str, help='path of dataset')
    parser.add_argument('--snr_list', default=['20','15','10','5','0'], nargs='+', help='snr_list')
    parser.add_argument('--model', default='gcn', type=str,
                        choices=['gcn', 'gat', 'gatedgnc', 'graphsage', 'mlp'], help='model select')

    return parser.parse_args()


def main():
    pass




if __name__ == '__main__': 
    main()
    print('Done!')