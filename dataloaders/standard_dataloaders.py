import time

import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#return MNIST dataloader
def mnist_dataloader(batch_size=256,train=True, shuffle = True):

    dataloader=DataLoader(
    datasets.MNIST('/app/research/fsl_da/data/mnist',train=train,download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ])),
    batch_size=batch_size,shuffle=shuffle, drop_last = False)

    return dataloader

def svhn_dataloader(batch_size ,train=True, shuffle = True):
    dataloader = DataLoader(
        datasets.SVHN('/app/research/fsl_da/data/SVHN', split=('train' if train else 'test'), download=True,
                       transform=transforms.Compose([
                           transforms.Resize((28,28)),
                           transforms.Grayscale(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5), (0.5))
                       ])),
        batch_size=batch_size, shuffle=shuffle, drop_last = False)

    return dataloader

def usps_dataloader(batch_size, train=True, shuffle = True):

    dataloader = DataLoader(
        datasets.USPS('/app/research/fsl_da/data/usps', train=train, download=True,
                       transform=transforms.Compose([
                           transforms.Resize((28,28)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5), (0.5))
                           ])),
        batch_size=batch_size, shuffle=shuffle, drop_last = False)
    return dataloader
