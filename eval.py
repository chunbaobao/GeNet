import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import collate
from prepare_dataset import SuperPixDataset, TestDataset, DGLFormDataset
from tensorboardX import SummaryWriter
from models.load_model import GeNet
import yaml
from train import evaluate_network


def eval_model(params):
    model_path = params['root']
    config_path = os.path.dirname(model_path).replace('checkpoint', 'config') + 'txt'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(config)
        net_params = config['net_params']
        model_name = config['model_name']
        dataset_name = config['dataset_name']
        params = config['params']

    model = GeNet(model_name, net_params)
    model.load_state_dict(torch.load(model_path))

    testset = TestDataset(dataset_name, params['rotation'])
    test_loader = DataLoader(
        testset, batch_size=params['batch_size'], shuffle=False, collate_fn=collate)
    writer = SummaryWriter(log_dir=os.path.dirname(
        model_path).replace('checkpoint', 'config') + 'eval')
    for snr in params['snr_list']:
        model.set_channel(params['snr'])
        test_loss, test_acc = evaluate_network(model, params['device'], test_loader)
        writer.add_scalar('test_loss', test_loss, snr)
        writer.add_scalar('test_acc', test_acc, snr)


def main():
    params = {}
    params['snr_list'] = range(0, 31, 1)
    params['rotation'] = [0]
    params['path'] = './out/checkpoints/GatedGCN_CIFAR10_GPU1_01h45m26s_on_Mar_07_2024/xxx.pkl'
    eval_model(params)


if __name__ == '__main__':
    main()
