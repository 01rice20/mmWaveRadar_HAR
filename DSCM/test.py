import os
import random
import h5py
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer
import torch.utils.data as data
import matplotlib.pyplot as plt
from torchvision import transforms, datasets, models
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from time import time
from torchsummary import summary
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from time import time
from torchvision.datasets import ImageFolder
from model.function import *
from model.module import *
import argparse
from thop import profile

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

def test(radar=60):

    premodel1 = []
    premodel2 = []
    premodel1 = UNet(3, 64)
    premodel2 = UNet(3, 64)
    
    num_class = 5
    batch_size = 8
    drop = 0.1
    dense_1 = 128
    dense_2 = 64

    premodel1, premodel2, dataset, _, _, test_sampler, nj = load_data_weight(premodel1, premodel2, radar)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, shuffle=False, pin_memory = True)
    model = Classificate(premodel1, premodel2, num_class, dense_1, dense_2, drop)
    model_path = '../weights/model.pth'
    model.load_state_dict(torch.load(model_path))
    premodel1 = premodel1.to(device)
    premodel2 = premodel2.to(device)
    model = model.to(device)
    model.eval()
    t_pred, t_true = [], []
    dummy_input = torch.randn(1, 3, 128, 128).to(device)

    flops, params = profile(model, (dummy_input, dummy_input))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))

    for doppler, ranges, labels in test_dataloader:
        doppler = doppler.to(device)
        ranges = ranges.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(doppler, ranges)
            _, predicted = torch.max(outputs, 1)
            t_pred.append(predicted)
            # print("predicted: ", predicted)
            t_true.append(labels)
            # print("labels: ", labels)

    t_pred = torch.cat(t_pred, dim=0).cpu().numpy()
    t_true = torch.cat(t_true, dim=0).cpu().numpy()
    accuracy = accuracy_score(t_true, t_pred)

    print(f'test_accuracy={round(accuracy, 5)}')

if __name__ == "__main__":
    set_seed()
    test()