import os
import random
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn.functional as F
from time import time
from torchsummary import summary
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from time import time
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from model.function import *
from model.attention import *

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, out_c),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, out_c),
            nn.ReLU()
        )
  
    def forward(self, x):
        x = self.conv1(x)
        concate = x.clone()
        x = self.conv2(x)

        return concate, x

class Encoder(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        real_c = int(out_c/2)
        self.conv = DoubleConv(in_c, real_c)
        self.max_pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        concate, x = self.conv(x)
        x = torch.cat([concate, x], dim=1)
        x = self.max_pool(x)

        return x

class Decoder(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        real_c = int(out_c/2)
        self.conv = DoubleConv(in_c, real_c)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        concate, x = self.conv(x)
        x = torch.cat([concate, x], dim=1)
        x = self.upsample(x)

        return x

class autoencoder(nn.Module):
    def __init__(self, channel, num_filter):
        super().__init__()
        self.encoder0 = DoubleConv(channel, num_filter)
        self.encoder1 = Encoder(num_filter, num_filter*2)
        self.encoder2 = Encoder(num_filter*2, num_filter*4)
        self.encoder3 = Encoder(num_filter*4, num_filter*8)
        self.encoder4 = Encoder(num_filter*8, num_filter*8)
        
        self.decoder1 = Decoder(num_filter*8, num_filter*8) 
        self.decoder2 = Decoder(num_filter*8, num_filter*4) 
        self.decoder3 = Decoder(num_filter*4, num_filter*2) 
        self.decoder4 = Decoder(num_filter*2, num_filter)
        self.decoder0 = nn.Conv2d(num_filter, channel, kernel_size=1)

    def forward(self, x):
        _, x = self.encoder0(x)
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.encoder4(x)

        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)
        x = self.decoder4(x)
        x = self.decoder0(x)
        
        return x

class FineTune(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_c),
        )

        self.simatt = SimpleAttention(in_c)
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_c)
        )
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=2, stride=2)

        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        ones(self.rate1)
        ones(self.rate2)

    def forward(self, x):
        set_seed(42)
        concate = x
        concate = self.downsample(concate)
        
        # Add Learning Parameters
        simattx = self.conv1(x)
        simattx = self.simatt(simattx)
        x = self.conv(x)
        x = self.rate1*x + self.rate2*simattx

        return x + concate

class Classificate(nn.Module):
    def __init__(self, model1, model2, model3, num_classes, dense_1, dense_2, drop):
        super().__init__()
        # dense1_inp 1000 for resatt, 3*128*128 for original model, 128*8*8 for convblock
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        
        # New Method
        self.convblock1 = FineTune(9, 32)
        self.convblock2 = FineTune(32, 64)
        self.convblock3 = FineTune(64, 64)
        self.convblock4 = FineTune(64, 128)

        # Classificate Layer for layer2(64*32*32), layer3(64*16*16)
        self.dense_layer0 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*16*16, dense_1),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(dense_1, dense_2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(dense_2, num_classes)
        )
        self.dense_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*8*8, dense_1),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(dense_1, dense_2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(dense_2, num_classes)
        )
        
    def forward(self, x1, x2, x3):
        set_seed(42)
        x1 = self.model1(x1)
        x2 = self.model2(x2)
        x3 = self.model3(x3)

        # Concatentation
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        predict = x.clone()
        x = self.convblock4(x)
        x = self.dense_layer(x)
        predict = self.dense_layer0(predict)
        
        return x, predict

class EncoderX(nn.Module):
    def __init__(self, depth, num_filter):
        super(EncoderX, self).__init__()
        self.depth = depth
        self.num_filter = num_filter
        self.layers = nn.ModuleList()

        for i in range(depth):
            if i != 0:
                input_channels = num_filter * 2
            else:
                input_channels = 3 

            conv1 = nn.Conv2d(input_channels, num_filter, kernel_size=3, stride=1, padding=1)
            conv2 = nn.Conv2d(num_filter, num_filter, kernel_size=9, stride=1, padding=4)
            self.layers.extend([conv1, nn.ReLU(), conv2, nn.ReLU()])

    def forward(self, x):
        intermediates = []
        for i in range(self.depth):
            conv1 = self.layers[i * 4]
            conv2 = self.layers[i * 4 + 2]
            x = conv1(x)
            x = torch.relu(x)
            intermediate = x.clone()
            intermediates.append(intermediate)
            x = conv2(x)
            x = torch.relu(x)
            x = torch.cat([intermediate, x], dim=1)
            x = nn.functional.max_pool2d(x, kernel_size=2)

        return intermediates, x

class DecoderX(nn.Module):
    def __init__(self, depth, num_filter):
        super(DecoderX, self).__init__()
        self.depth = depth
        self.num_filter = num_filter
        self.layers = nn.ModuleList()
        input_channels = self.num_filter * 2

        for i in range(depth):
            conv3 = nn.Conv2d(input_channels, num_filter, kernel_size=3, stride=1, padding=1)
            conv4 = nn.Conv2d(num_filter, num_filter, kernel_size=9, stride=1, padding=4)

            self.layers.extend([conv3, nn.ReLU(), conv4, nn.ReLU()])

        self.layers.append(nn.Conv2d(num_filter*2, 3, kernel_size=3, stride=1, padding=1))

    def forward(self, intermediates, x):
        intermediate2 = []
        for i in range(self.depth):
            conv3 = self.layers[i * 4]
            conv4 = self.layers[i * 4 + 2]
            x = conv3(x)
            x = torch.relu(x)
            intermediate = x.clone()
            intermediate2.append(intermediate)
            x = conv4(x)
            x = torch.relu(x)
            x = torch.cat([intermediate, x], dim=1)
            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            # x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        
        x = self.layers[-1](x)

        return x

class MyModel(nn.Module):
    def __init__(self, depth, num_filter):
        super(MyModel, self).__init__()
        self.encoder = EncoderX(depth, num_filter)
        self.decoder = DecoderX(depth, num_filter)

    def forward(self, x):
        set_seed(42)
        intermediates, x = self.encoder(x)
        x = self.decoder(intermediates, x)
        
        return x

class BaselineClassificate(nn.Module):
    def __init__(self, model, num_classes, dense_1, dense_2, drop):
        super(BaselineClassificate, self).__init__()
        # pre-train Model
        self.model = model
  
        self.flatten = nn.Flatten()
        self.dense_1 = nn.Sequential(nn.Linear(3*128*128, dense_1), nn.ReLU(), nn.Dropout(drop))    # For Original Model
        self.dense_2 = nn.Sequential(nn.Linear(dense_1, dense_2), nn.ReLU(), nn.Dropout(drop))
        self.output_layer = nn.Linear(dense_2, num_classes)
        
    def forward(self, x):
        set_seed(42)
        x = self.model(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.output_layer(x)
        
        return x
