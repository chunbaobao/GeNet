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
from utils import view_model_param, gpu_setup
from torch.nn.parallel import DataParallel
from prepare_dataset import SuperPixDataset
import numpy as np

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def main():
    pass




if __name__ == '__main__': 
    main()
    print('Done!')