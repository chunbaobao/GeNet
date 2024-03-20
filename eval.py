import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from utils import collate
from prepare_dataset import SuperPixDataset, TestDataset, DGLFormDataset, process_image
from tensorboardX import SummaryWriter
from models.load_model import GeNet, load_baseline
import yaml
from train import evaluate_network
import torchvision.transforms.functional as TF
import time
import gc
import multiprocessing as mp
import numpy as np
from utils import gpu_setup

def evaluate_baseline(model, device, test_loader):
    # baseline model does not need to set eval mode!!!
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy


class PaintedDateSet(Dataset):
    def __init__(self, dataset_name, rotated_angle = 0, is_plot = False):
        data_path = '../dataset'
        if dataset_name == 'mnist':
            self.img_size = 28
            n_sp = 75
            compactness = .25
            dataset = datasets.MNIST(root=data_path, train=False, download=False)

        elif dataset_name == 'cifar10':
            n_sp = 150
            compactness = 10
            self.img_size = 32
            dataset = datasets.CIFAR10(root=data_path, train=False, download=False)
        else:
            raise Exception('Invalid dataset name')
        # N * H * W (* C)
        images = dataset.data.numpy() if isinstance(dataset.data, torch.Tensor) else dataset.data
        labels = dataset.targets
        if rotated_angle != 0:
            if dataset_name == 'mnist':
                # 6 and 9 are unrecognizable when rotated
                valid_labels = [i for i in range(10) if i != 6 and i != 9]
                valid_indices = [i for i, label in enumerate(
                    labels) if label.item() in valid_labels]
                images = images[valid_indices] if not is_plot else images
                labels = labels[valid_indices] if not is_plot else labels
                images = TF.rotate(torch.from_numpy(images), rotated_angle, expand=False)
            else:
                # N * C * H * W
                images = TF.rotate(torch.from_numpy(images).permute(0,3,1,2), rotated_angle, expand=False)
                images = images.permute(0,2,3,1)
            images = images.numpy()
            
        n_images = len(images)
        with mp.Pool() as pool:
           sp_data = pool.map(
                process_image, [(images[i], n_sp, compactness, False, i, dataset_name) for i in range(n_images)])

        # sp_data = []
        # for i in range(n_images):
        #     sp_data.append(process_image((images[i], n_sp, compactness, False, i, dataset_name)))
           
           
        self.painted_imgs = []
        self.labels = labels
        for idx, (sp_intensity, _, sp_order, superpixels) in enumerate(sp_data):
            painted_img = np.zeros_like(images[idx], dtype=np.float32) # H * W (* C)
            for seg in sp_order:
                mask = (superpixels == seg)
                painted_img[mask] = sp_intensity[seg]
            painted_img = painted_img[:,:,None] if painted_img.ndim == 2 else painted_img # H * W * C
            painted_img = torch.from_numpy(painted_img)
            painted_img = painted_img.permute((2, 0, 1)) #　C * H * W
            self.painted_imgs.append(painted_img)

  
        
            
    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
        """
        return self.painted_imgs[idx], self.labels[idx]
            


def eval_model(device):

    print('evaluating GNN model')
    # * model_path need to change
    model_path = 'out/checkpoints/GATEDGCN_CIFAR10_06h53m54s_on_Mar_15_2024_PC/epoch_252.pkl'
    # model_path = 'out/checkpoints/GATEDGCN_MNIST_12h18m11s_on_Mar_15_2024_PC/epoch_198.pkl'

    # load config from train.py
    config_path = os.path.dirname(model_path).replace('checkpoint', 'config') + '.yaml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.UnsafeLoader)
        # print(config)
        net_params = config['net_params']
        model_name = config['model_name']
        dataset_name = config['dataset_name']
        params = config['params']
        
    # load model
    model = GeNet(model_name, net_params)
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    
    # # for snr
    print('evaluating snr...')
    testset = TestDataset(dataset_name).test
    test_loader = DataLoader(
        testset, batch_size=params['batch_size'], shuffle=False, collate_fn=testset.collate)
    if not os.path.exists('./out/eval/snr'):
        os.makedirs('./out/eval/snr')
    writer = SummaryWriter(log_dir='./out/eval/snr/{}_{}_{}'.
                            format(model_name, dataset_name, time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')))
    for snr in range(-50, 31, 1):
        model.set_channel(snr)
        test_loss, test_acc = evaluate_network(model, device, test_loader)
        writer.add_scalar('test_loss/snr', test_loss, snr)
        writer.add_scalar('test_acc/snr', test_acc, snr)
    writer.close()
    
    
    # for rotation
    print('evaluating rotation...')
    if not os.path.exists('./out/eval/rotation'):
        os.makedirs('./out/eval/rotation')
    writer = SummaryWriter(log_dir='./out/eval/rotation/{}_{}_{}'.
                            format(model_name, dataset_name, time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')))
    for rotation in range(0, 360, 1):
        testset = TestDataset(dataset_name, rotation).test
        test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, collate_fn=testset.collate)
        model.set_channel(None)
        test_loss, test_acc = evaluate_network(model, device, test_loader)
        writer.add_scalar('test_loss/rotation', test_loss, rotation)
        writer.add_scalar('test_acc/rotation', test_acc, rotation)
        del testset, test_loader
        gc.collect() 
    writer.close()
    
    # for n_sp
    # print('evaluating n_sp...')
    # if not os.path.exists('./out/eval/n_sp'):
    #     os.makedirs('./out/eval/n_sp')
    # writer = SummaryWriter(log_dir='./out/eval/n_sp/{}_{}_{}'.
    #                         format(model_name, dataset_name, time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')))
    # if dataset_name == 'mnist':
    #     n_sp_range = range(20, 91, 1)
    # else:
    #     n_sp_range = range(30, 131, 1)
        
    # for n_sp in n_sp_range:
    #     testset = TestDataset(dataset_name, n_sp_test = n_sp).test
    #     test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, collate_fn=testset.collate)
    #     model.set_channel(None)
    #     test_loss, test_acc = evaluate_network(model, device, test_loader)
    #     writer.add_scalar('test_loss/n_sp', test_loss, n_sp)
    #     writer.add_scalar('test_acc/n_sp', test_acc, n_sp)
    #     del testset, test_loader
    #     gc.collect() 
    # writer.close()
        
        
        

        

def eval_baseline(device, dataset_name, is_paint = True):
    print('evaluating baseline model on {} dataset, is_paint: {}'.format(dataset_name, is_paint))
    
    # load model
    model_name = 'resnet'
    model = load_baseline(dataset_name)
    model.to(device)
    
    
    # for snr
    print('evaluating snr...')
    if not os.path.exists('./out/eval/snr'):
        os.makedirs('./out/eval/snr')
    writer = SummaryWriter(log_dir='./out/eval/snr/{}_{}_{}_{}'.
                            format(model_name, dataset_name, time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y'), is_paint))
    if is_paint:
        testset = PaintedDateSet(dataset_name)
        
    else:
        if dataset_name == 'mnist':
            testset = datasets.MNIST(root='../dataset', train=False, download=False, transform=transforms.ToTensor())
        else:
            testset = datasets.CIFAR10(root='../dataset', train=False, download=False, transform=transforms.ToTensor())
        
    for snr in range(-50, 31, 1):
        test_loader = DataLoader(testset, batch_size=32, shuffle=False, collate_fn=collate)   
        model.set_channel(snr)
        test_acc = evaluate_baseline(model, device, test_loader)
        writer.add_scalar('test_acc/snr', test_acc, snr)
    writer.close()
    
    
    # for rotation
    print('evaluating rotation...')
    if not os.path.exists('./out/eval/rotation'):
        os.makedirs('./out/eval/rotation')
    writer = SummaryWriter(log_dir='./out/eval/rotation/{}_{}_{}_{}'.
                            format(model_name, dataset_name, time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y'), is_paint))
    for rotation in range(0, 360, 1):
        if is_paint:
            testset = PaintedDateSet(dataset_name, rotation)
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: TF.rotate(x, rotation))])
            if dataset_name == 'mnist':
                testset = datasets.MNIST(root='../dataset', train=False, download=False, transform=transform)
            else:
                testset = datasets.CIFAR10(root='../dataset', train=False, download=False, transform=transform)            

        test_loader = DataLoader(testset, batch_size=32, shuffle=False)   
        model.set_channel(None)
        test_acc = evaluate_baseline(model, device, test_loader)
        writer.add_scalar('test_acc/rotation', test_acc, rotation)
        del testset, test_loader
    writer.close()
    print('-'*89)


def main():
    if torch.cuda.device_count() > 1:
        device = gpu_setup(True, 1)
    elif torch.cuda.is_available():
        device = gpu_setup(True, 0)
    else:
        device = gpu_setup(False, 0)
        
    # for GNN models    
    eval_model(device)
    
    # # for baseline models
    # for dataset_name in ['mnist', 'cifar10']:
    #     for is_paint in [True, False]:
    #         eval_baseline(device, dataset_name, is_paint=is_paint)


if __name__ == '__main__':
    main()
