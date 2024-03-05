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
    parser.add_argument('--out', default='./out', type=str, help='saved_path')
    parser.add_argument('--dataset_dir')
    parser.add_argument('--snr_list', default=['19', '13',
                        '7', '4', '1'], nargs='+', help='snr_list')
    parser.add_argument('--model', default='gcn', type=str,
                        choices=['gcn', 'gat', 'gatedgnc', 'graphsage', 'mlp'], help='model select')

    return parser.parse_args()


def main():
    pass




if __name__ == '__main__': 
    main()
    print('Done!')