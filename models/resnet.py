# refers to https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
# https://github.com/S-HuaBomb/mnist-classification-in-action
# https://github.com/ThunderVVV/pytorch_resnet_cifar10
# https://github.com/AmitaiBiton/FashionMNIST_resnet18
'''
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import datasets, transforms, models
import torch
from torch.utils.data import DataLoader
import channel
from utils import view_model_param


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    


class ResNet_CIFAR10(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        if hasattr(self, 'channel') and self.channel is not None:
            out = self.channel(out)
        out = self.linear(out)
        return out
    
    def set_channel(self, snr=None):
        if snr is None:
            self.channel = None
        else:
            self.channel = channel.Channel(snr)
            
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.bn = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)  # feature map doesnot change
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = F.relu(self.bn(self.conv1(x)))
        y = self.bn(self.conv1(y))

        return F.relu(x + y)  


class ResNet_MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.rblock1 = ResidualBlock(channels=16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.rblock2 = ResidualBlock(channels=32)

        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(512, 10)

    def set_channel(self, snr=None):
        if snr is None:
            self.channel = None
        else:
            self.channel = channel.Channel(snr)


    def forward(self, x):
        batch_size = x.size(0)

        x = self.mp(F.relu(self.conv1(x)))  # N,16,12,12
        x = self.rblock1(x)  # N,16,12,12

        x = self.mp(F.relu(self.conv2(x)))  # N,32,4,4
        x = self.rblock2(x)  # N,32,4,4

        if hasattr(self, 'channel') and self.channel is not None:
            x = self.channel(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)

        return x
    
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss

class ResNet_FashionMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(num_classes=10)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)
        
    def set_channel(self, snr=None):
        if snr is None:
            self.channel = None
        else:
            self.channel = channel.Channel(snr)
            
    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        if hasattr(self, 'channel') and self.channel is not None:
            x = self.channel(x)
        x = self.resnet.fc(x)
        return x
    
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss

def resnet_cifar10():
    return ResNet_CIFAR10(BasicBlock, [3, 3, 3])

def resnet_mnist():
    return ResNet_MNIST()

def resnet_fashionmnist():
    return ResNet_FashionMNIST()




if __name__ == '__main__':
    dataset_names = ['mnist','cifar10']
    dataset_names = ['fashionmnist']
    snr = 20
    for dataset_name in dataset_names:
        if dataset_name == 'mnist':
            model = resnet_mnist()
            dataset = datasets.MNIST(root='../dataset', train=False, download=False, transform=transforms.ToTensor())
        elif dataset_name == 'cifar10':
            model = resnet_cifar10()
            dataset = datasets.CIFAR10(root='../dataset', train=False, download=False, transform=transforms.ToTensor())
        elif dataset_name == 'fashionmnist':
            model = resnet_fashionmnist()
            dataset = datasets.FashionMNIST(root='../dataset', train=False, download=False, 
                                            transform=transforms.ToTensor())
        
        model.load_state_dict(torch.load('./models/resnet_{}.pth'.format(dataset_name)))
        model.set_channel(snr)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in dataloader:
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print('Accuracy on {} dataset: {:.2f}%'.format(dataset_name.upper(), accuracy * 100))
