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
from utils import view_model_param, gpu_setup, set_seed, accuracy
from prepare_dataset import SuperPixDataset, TestDataset
import numpy as np
from channel import Channel
import time
import random
from tensorboardX import SummaryWriter
from glob import glob
from models.load_model import GeNet


def train_epoch(model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()

        batch_scores = model.forward(batch_graphs, batch_x, batch_e)
        loss = model.loss(batch_scores, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(batch_scores, batch_labels)
        nb_data += batch_labels.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data

    return epoch_loss, epoch_train_acc, optimizer


def evaluate_network(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_labels = batch_labels.to(device)

            batch_scores = model.forward(batch_graphs, batch_x, batch_e)
            loss = model.loss(batch_scores, batch_labels)
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(batch_scores, batch_labels)
            nb_data += batch_labels.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data

    return epoch_test_loss, epoch_test_acc


def config_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='./out', type=str, help='path of output')
    parser.add_argument('--dataset_dir', default='./data', type=str, help='path of dataset')
    parser.add_argument('--snr_list', default=['20', '15',
                        '10', '5', '0'], nargs='+', help='snr_list')
    parser.add_argument('--model', default='gcn', type=str,
                        choices=['gcn', 'gat', 'gatedgnc', 'mlp'], help='model select')

    return parser.parse_args()


def train_pipeline(model_name, dataset_name, params, dirs):
    # """
    #     PARAMETERS
    # """

    device = gpu_setup(True, 1)
    n_heads = -1
    edge_feat = False
    pseudo_dim_MoNet = -1
    kernel = -1
    gnn_per_block = -1
    embedding_dim = -1
    pool_ratio = -1
    n_mlp_GIN = -1
    gated = False
    self_loop = False
    #self_loop = True
    max_time = 12

    if model_name == 'GatedGCN':
        seed = 41
        epochs = 1000
        batch_size = 16
        init_lr = 5e-5
        lr_reduce_factor = 0.5
        lr_schedule_patience = 25
        min_lr = 1e-6
        weight_decay = 0
        L = 4
        hidden_dim = 70
        out_dim = hidden_dim
        dropout = 0.0
        readout = 'mean'

    if model_name == 'GCN':
        seed = 41
        epochs = 1000
        batch_size = 5
        init_lr = 5e-5
        lr_reduce_factor = 0.5
        lr_schedule_patience = 25
        min_lr = 1e-6
        weight_decay = 0
        L = 4
        hidden_dim = 146
        out_dim = hidden_dim
        dropout = 0.0
        readout = 'mean'

    if model_name == 'GAT':
        seed = 41
        epochs = 1000
        batch_size = 50
        init_lr = 5e-5
        lr_reduce_factor = 0.5
        lr_schedule_patience = 25
        min_lr = 1e-6
        weight_decay = 0
        L = 4
        n_heads = 8
        hidden_dim = 19
        out_dim = n_heads*hidden_dim
        dropout = 0.0
        readout = 'mean'
        print('True hidden dim:', out_dim)

    if model_name == 'MLP':
        seed = 41
        epochs = 1000
        batch_size = 50
        init_lr = 5e-4
        lr_reduce_factor = 0.5
        lr_schedule_patience = 25
        min_lr = 1e-6
        weight_decay = 0
        gated = False  # MEAN
        L = 4
        hidden_dim = 168
        out_dim = hidden_dim
        dropout = 0.0
        readout = 'mean'
        gated = True  # GATED
        L = 4
        hidden_dim = 150
        out_dim = hidden_dim
        dropout = 0.0
        readout = 'mean'

    # generic new_params
    net_params = {}
    net_params['device'] = device
    net_params['gated'] = gated  # for mlpnet baseline
    net_params['in_dim'] = trainset[0][0].ndata['feat'][0].size(0)
    net_params['in_dim_edge'] = trainset[0][0].edata['feat'][0].size(0)
    net_params['residual'] = True
    net_params['hidden_dim'] = hidden_dim
    net_params['out_dim'] = out_dim
    num_classes = len(np.unique(np.array(trainset[:][1])))
    net_params['n_classes'] = num_classes
    net_params['n_heads'] = n_heads
    net_params['L'] = L  # min L should be 2
    net_params['readout'] = "sum"
    net_params['layer_norm'] = True
    net_params['batch_norm'] = True
    net_params['in_feat_dropout'] = 0.0
    net_params['dropout'] = 0.0
    net_params['edge_feat'] = edge_feat
    net_params['self_loop'] = self_loop

    # for MLPNet
    net_params['gated'] = gated

    # specific for MoNet
    net_params['pseudo_dim_MoNet'] = pseudo_dim_MoNet
    net_params['kernel'] = kernel

    # specific for GIN
    net_params['n_mlp_GIN'] = n_mlp_GIN
    net_params['learn_eps_GIN'] = True
    net_params['neighbor_aggr_GIN'] = 'sum'

    # specific for graphsage
    net_params['sage_aggregator'] = 'meanpool'

    # specific for diffpoolnet
    net_params['data_mode'] = 'default'
    net_params['gnn_per_block'] = gnn_per_block
    net_params['embedding_dim'] = embedding_dim
    net_params['pool_ratio'] = pool_ratio
    net_params['linkpred'] = True
    net_params['num_pool'] = 1
    net_params['cat'] = False
    net_params['batch_size'] = batch_size

    # specific for RingGNN
    net_params['radius'] = 2
    num_nodes_train = [trainset[i][0].number_of_nodes() for i in range(len(trainset))]
    num_nodes_test = [testset[i][0].number_of_nodes() for i in range(len(testset))]
    num_nodes = num_nodes_train + num_nodes_test
    net_params['avg_node_num'] = int(np.ceil(np.mean(num_nodes)))

    # specific for 3WLGNN
    net_params['depth_of_mlp'] = 2

    # calculate assignment dimension: pool_ratio * largest graph's maximum
    # number of nodes  in the dataset
    max_num_nodes_train = max(num_nodes_train)
    max_num_nodes_test = max(num_nodes_test)
    max_num_node = max(max_num_nodes_train, max_num_nodes_test)
    net_params['assign_dim'] = int(
        max_num_node * net_params['pool_ratio']) * net_params['batch_size']

    t0 = time.time()
    per_epoch_time = []
    dataset = SuperPixDataset(dataset_name)

    if model_name in ['GCN', 'GAT']:
        if net_params['self_loop']:
            print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
            dataset._add_self_loops()

    trainset, valset = dataset.train, dataset.val
    testset = TestDataset(
        dataset_name, rotated_angle=params['rotated_angle'], n_sp_test=params['n_sp_test'])
    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']

    # Write the network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n"""
                .format(dataset_name, model_name, params, net_params, net_params['total_param']))

    log_dir = os.path.join(root_log_dir, "RUN_" + str(0))
    writer = SummaryWriter(log_dir=log_dir)

    # setting seeds
    set_seed(params['seed'])

    print("Training Graphs: ", len(trainset))
    print("Validation Graphs: ", len(valset))
    print("Test Graphs: ", len(testset))
    print("Number of Classes: ", net_params['n_classes'])

    model = GeNet(model_name, net_params, snr=params['snr'])
    model = model.to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)

    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_accs, epoch_val_accs = [], []

    # batching exception for Diffpool
    drop_last = True if model_name == 'DiffPool' else False

    # import train functions for all other GCNs

    train_loader = DataLoader(
        trainset, batch_size=params['batch_size'], shuffle=True, drop_last=drop_last, collate_fn=dataset.collate)
    val_loader = DataLoader(
        valset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last, collate_fn=dataset.collate)
    test_loader = DataLoader(
        testset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last, collate_fn=dataset.collate)

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        with tqdm(range(params['epochs'])) as t:
            for epoch in t:

                t.set_description('Epoch %d' % epoch)

                start = time.time()

                epoch_train_loss, epoch_train_acc, optimizer = train_epoch(
                    model, optimizer, device, train_loader, epoch)

                epoch_val_loss, epoch_val_acc = evaluate_network(model, device, val_loader, epoch)
                _, epoch_test_acc = evaluate_network(model, device, test_loader, epoch)

                epoch_train_losses.append(epoch_train_loss)
                epoch_val_losses.append(epoch_val_loss)
                epoch_train_accs.append(epoch_train_acc)
                epoch_val_accs.append(epoch_val_acc)

                writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                writer.add_scalar('train/_acc', epoch_train_acc, epoch)
                writer.add_scalar('val/_acc', epoch_val_acc, epoch)
                writer.add_scalar('test/_acc', epoch_test_acc, epoch)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

                t.set_postfix(time=time.time()-start, lr=optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                              train_acc=epoch_train_acc, val_acc=epoch_val_acc,
                              test_acc=epoch_test_acc)

                per_epoch_time.append(time.time()-start)

                # Saving checkpoint
                ckpt_dir = os.path.join(root_ckpt_dir, "RUN_")
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                torch.save(model.state_dict(), '{}.pkl'.format(ckpt_dir + "/epoch_" + str(epoch)))

                files = glob.glob(ckpt_dir + '/*.pkl')
                for file in files:
                    epoch_nb = file.split('_')[-1]
                    epoch_nb = int(epoch_nb.split('.')[0])
                    if epoch_nb < epoch-1:
                        os.remove(file)

                scheduler.step(epoch_val_loss)

                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break

                # Stop training after params['max_time'] hours
                if time.time()-t0 > params['max_time']*3600:
                    print('-' * 89)
                    print("Max_time for training elapsed {:.2f} hours, so stopping".format(
                        params['max_time']))
                    break

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')

    _, test_acc = evaluate_network(model, device, test_loader, epoch)
    _, train_acc = evaluate_network(model, device, train_loader, epoch)
    print("Test Accuracy: {:.4f}".format(test_acc))
    print("Train Accuracy: {:.4f}".format(train_acc))
    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time()-t0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    writer.close()

    """
        Write the results in out_dir/results folder
    """
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nTEST ACCURACY: {:.4f}\nTRAIN ACCURACY: {:.4f}\n\n
    Convergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\n"""
                .format(dataset_name, model_name, params, net_params, model, net_params['total_param'],
                        np.mean(np.array(test_acc))*100, np.mean(np.array(train_acc))*100, epoch, (time.time()-t0)/3600, np.mean(per_epoch_time)))


def main():
    models = ['gcn', 'gat', 'gatedgcn', 'mlp']
    datasets = ['mnist', 'cifar10', 'fashionmnist']


if __name__ == '__main__':
    main()
    print('Done!')
