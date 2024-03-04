import numpy as np
import os
import torch

def gpu_setup(use_gpu, gpu_id):
    if torch.cuda.is_available() and use_gpu:
        device = torch.device(f'cuda:{gpu_id}')
        print('cuda available with GPU:',torch.cuda.get_device_name(gpu_id))
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device

def view_model_param(model_name, model):
    total_param = 0
    print("MODEL DETAILS:\n")
    #print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', model_name, total_param)
    return total_param

def split_dataset(labels, valid_split=0.1):
    idx = np.random.permutation(len(labels))
    valid_idx = []
    train_idx = []
    label_count = [0 for _ in range(1+max(labels))]
    valid_count = [0 for _ in label_count]
    
    for i in idx:
        label_count[labels[i]] += 1
    
    
    for i in idx:
        l = labels[i]
        if valid_count[l] < label_count[l]*valid_split:
            valid_count[l] += 1
            valid_idx.append(i)
        else:
            train_idx.append(i)
    return train_idx, valid_idx