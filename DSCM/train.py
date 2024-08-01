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
# import wandb
from time import time
from torchvision.datasets import ImageFolder
from model.function import *
from model.module import *
import argparse
from thop import profile
from model.attention import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
# wandb.init(project="final_score")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

def train(radar):
    cnt = 0
    epochs = 100
    batch_size = [16, 32]

    dense_1 = [64, 64, 128, 128, 256, 256]
    dense_2 = [32, 64, 64, 128, 128, 256]
    learn_rate = [2e-4, 1e-4]
    drop = 0.1
    num_filter = 64
    num_class = 11
    acc_hist = []
    patience = 3
    result = {}
    bestscore = best_score()

    premodel1 = []
    premodel2 = []
    premodel3 = []
    premodel1 = autoencoder(3, 64)
    premodel2 = autoencoder(3, 64)
    premodel3 = autoencoder(3, 64)

    # if torch.cuda.device_count() > 1:
    #     premodel1 = nn.DataParallel(premodel1, device_ids=[0, 1]).to(device)
    #     premodel2 = nn.DataParallel(premodel2, device_ids=[0, 1]).to(device)
    #     premodel3 = nn.DataParallel(premodel3, device_ids=[0, 1]).to(device)

    premodel1, premodel2, premodel3, dataset, train_sampler, val_sampler, test_sampler, nj = load_data_weight_test(premodel1, premodel2, premodel3, radar)
    for x in range(len(batch_size)):
        dataloader = DataLoader(dataset, batch_size=batch_size[x], sampler=train_sampler, shuffle=False, pin_memory = True)
        val_dataloader = DataLoader(dataset, batch_size=batch_size[x], sampler=val_sampler, shuffle=False, pin_memory = True)
        test_dataloader = DataLoader(dataset, batch_size=batch_size[x], sampler=test_sampler, shuffle=False, pin_memory = True)

        for k in range(len(dense_1)):
            for i in range(len(learn_rate)):
                model = []
                early_stopping = EarlyStopping(patience)
                model = Classificate(premodel1, premodel2, premodel3, num_class, dense_1[k], dense_2[k], drop).to(device)
                optimizer = optim.Adam(model.parameters(), lr=learn_rate[i], weight_decay=1e-6)
                train_loss, train_cnt, val_loss, val_cnt, accuracy = 0, 0, 0, 0, 0
                # dummy_input = torch.randn(1, 3, 128, 128).to(device)
                # flops, params = profile(model, (dummy_input, dummy_input, dummy_input))
                # print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
                for epoch in range(epochs):
                    model.train()
                    for spectrograms, ranges, velocitys, labels in dataloader:
                        spectrograms = spectrograms.to(device)
                        ranges = ranges.to(device)
                        velocitys = velocitys.to(device)
                        labels = labels.to(device)
                        optimizer.zero_grad()
                        outputs, output0 = model(spectrograms, ranges, velocitys)
                        loss = final_loss(outputs, output0, labels, nj, epoch)
                        _, predicted = torch.max(outputs, 1)
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item()
                        train_cnt += 1
                    
                    print(f'Epoch {epoch + 1}, {radar} GHz, batch_size={batch_size[x]}, '
                            f'dense_1={dense_1[k]}, dense_2={dense_2[k]}, learn_rate={learn_rate[i]}, '
                            f'train_loss={train_loss/train_cnt}')
                    # wandb.log({"train_loss": train_loss/train_cnt})
                    
                    if((epoch + 1) % 5 == 0):
                        model.eval()
                        v_pred, v_true = [], []
                        istest = False
                        for spectrograms, ranges, velocitys, labels in val_dataloader:
                            with torch.no_grad():
                                spectrograms = spectrograms.to(device)
                                ranges = ranges.to(device)
                                velocitys = velocitys.to(device)
                                labels = labels.to(device)
                                outputs, output0 = model(spectrograms, ranges, velocitys)
                                loss = final_loss(outputs, output0, labels, nj, epoch)
                                val_loss += loss.item()
                                val_cnt += 1
                                _, predicted = torch.max(outputs, 1)
                                v_pred.append(predicted)
                                v_true.append(labels)

                        v_pred = torch.cat(v_pred, dim=0).cpu().numpy()
                        v_true = torch.cat(v_true, dim=0).cpu().numpy()
                        accuracy = score(v_pred, v_true, istest)
                        vloss_hist = val_loss/val_cnt
                        print(f'val_loss={vloss_hist}, val_acc={round(accuracy, 5)}')
                        # wandb.log({"val_loss": vloss_hist, "val_accuracy": accuracy})
                        
                        if early_stopping(accuracy):
                            print("Early stopping")
                            break

                model.eval()
                t_pred, t_true = [], []
                istest = True
                isfinal = False
                for spectrograms, ranges, velocitys, labels in test_dataloader:
                    spectrograms = spectrograms.to(device)
                    ranges = ranges.to(device)
                    velocitys = velocitys.to(device)
                    labels = labels.to(device)
                    with torch.no_grad():
                        outputs, output0 = model(spectrograms, ranges, velocitys)
                        _, predicted = torch.max(outputs, 1)
                        t_pred.append(predicted)
                        t_true.append(labels)
    
                t_pred = torch.cat(t_pred, dim=0).cpu().numpy()
                t_true = torch.cat(t_true, dim=0).cpu().numpy()
                cm, precision, recall, f1, accuracy = score(t_pred, t_true, istest)
                bestscore(cm, precision, recall, f1, accuracy, isfinal)

                print(f'test_precision={round(precision, 5)}, test_recall={round(recall, 5)}, '
                        f'test_f1={round(f1, 5)}, test_accuracy={round(accuracy, 5)}')
                # wandb.log({"accuracy": accuracy})
    isfinal = True
    result = bestscore(cm, precision, recall, f1, accuracy, isfinal)
    # wandb.log({"Best Precision": result['precision'], "Best Recall": result['recall'], 
    #             "Best F1": result['f1'], "Best Recall": result['recall'], "Best accuracy": result['accuracy']})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=int, help="Data parameter")
    args = parser.parse_args()
    
    set_seed(42)
    train(args.data)