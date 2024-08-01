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
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import torch.nn.functional as F
from time import time
from torchsummary import summary
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler, Dataset, random_split, Subset
from time import time
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from collections import Counter


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False 
    torch.backends.cudnn.benchmark = False

def custom_transform(img):
    img = img.crop((img.size[0]//10, 0, img.size[0], img.size[1]))
    img = img.crop((0, 0, img.size[0]-img.size[0]//10, img.size[1]))
    angle = np.random.uniform(-10, 10)
    img = img.rotate(angle)
    
    return img

class single_dataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        img = self.dataset[idx][0]
        label = self.dataset[idx][1]
        if self.transform:
            img = self.transform(img)
        
        return (img, label)

class multi_dataset(Dataset):
    def __init__(self, dataset, dataset_r, dataset_v, transform):
        self.dataset = dataset
        self.dataset_r = dataset_r
        self.dataset_v = dataset_v
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        # img_path = self.dataset.samples[idx][0]  # Assuming dataset.samples contains (path, label)
        # img_r_path = self.dataset_r.samples[idx][0]
        # img_v_path = self.dataset_v.samples[idx][0]
        # label = self.dataset[idx][1]

        # print(f"Absolute path for img: {os.path.abspath(img_path)}")
        # print(f"Absolute path for img_r: {os.path.abspath(img_r_path)}")
        # print(f"Absolute path for img_v: {os.path.abspath(img_v_path)}")
        # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

        img = self.dataset[idx][0]
        img_r = self.dataset_r[idx][0]
        img_v = self.dataset_v[idx][0]
        label = self.dataset[idx][1]
        if self.transform:
            img = self.transform(img)
            img_r = self.transform(img_r)
            img_v = self.transform(img_v)
        
        return (img, img_r, img_v, label)

class multi_dataset_ablasion(Dataset):
    def __init__(self, dataset, dataset_r, transform):
        self.dataset = dataset
        self.dataset_r = dataset_r
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        img = self.dataset[idx][0]
        img_r = self.dataset_r[idx][0]
        label = self.dataset[idx][1]
        if self.transform:
            img = self.transform(img)
            img_r = self.transform(img_r)
        
        return (img, img_r, label)

def load_data(radar, data):
    root = []
    file = []
    if(radar == 10):
        if(data == 1):
            root = "../dataset/spectrogram_10/"
            file = "spectrogram_10"
        elif(data == 2):
            root = "../dataset/range_10/"
            file = "range_10"
        elif(data == 3):
            root = "../dataset/velocity_10/"
            file = "velocity_10"
    elif (radar == 24):
        if(data == 1):
            root = "../dataset/spectrogram_24/"
            file = "spectrogram_24"
        elif(data == 2):
            root = "../dataset/range_24/"
            file = "range_24"
        elif(data == 3):
            root = "../dataset/velocity_24/"
            file = "velocity_24"
    elif(radar == 77):
        if(data == 1):
            root = "../dataset/spectrogram_77/"
            file = "spectrogram_77"
        elif(data == 2):
            root = "../dataset/range_77/"
            file = "range_77"
        elif(data == 3):
            root = "../dataset/velocity_77/"
            file = "velocity_77"
    elif(radar == 60):
        if(data == 1):
            root = "../dataset/spectrogram_60/"
            file = "spectrogram_60"
        elif(data == 2):
            root = "../dataset/range_60/"
            file = "range_60"
        elif(data == 3):
            root = "../dataset/velocity_60/"
            file = "velocity_60"
    elif(radar == 111):
        if(data == 1):
            root = "../dataset/spectrogram_111/"
            file = "spectrogram_111"
        elif(data == 2):
            root = "../dataset/range_111/"
            file = "range_111"
        elif(data == 3):
            root = "../dataset/velocity_111/"
            file = "velocity_111"

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(root=root, transform=transform)

    train_ratio = 0.8
    num_data = len(dataset)
    indices = list(range(num_data))
    random.shuffle(indices)
    num_train = int(train_ratio * num_data)
    train_indices = indices[:num_train]
    test_indices = indices[num_train:]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    return dataset, train_sampler, test_sampler, file

def load_data_weight(model1, model2, model3, radar):

    root = []
    root_r = []
    root_v = []
    nj = []

    if(radar == 10):
        root = "../dataset/spectrogram_10/"
        root_r = "../dataset/range_10/"
        root_v = "../dataset/velocity_10/"
        model1.load_state_dict(torch.load('../weights/spectrogram_10.pth'))
        model2.load_state_dict(torch.load('../weights/range_10.pth'))
        model3.load_state_dict(torch.load('../weights/velocity_10.pth'))
    elif (radar == 24):
        root = "../dataset/spectrogram_24/"
        root_r = "../dataset/range_24/"
        root_v = "../dataset/velocity_24/"
        model1.load_state_dict(torch.load('../weights/spectrogram_24.pth'))
        model2.load_state_dict(torch.load('../weights/range_24.pth'))
        model3.load_state_dict(torch.load('../weights/velocity_24.pth'))
    elif(radar == 60):
        root = "../../dataset/spectrogram/spectrogram_60/"
        root_r = "../../dataset/range/range_60/"
        root_v = "../../dataset/velocity/velocity_60/"
        model1.load_state_dict(torch.load('../weights/spectrogram_60.pth'))
        model2.load_state_dict(torch.load('../weights/range_60.pth'))
        model3.load_state_dict(torch.load('../weights/velocity_60.pth'))
    elif(radar == 77):
        root = "../../dataset/spectrogram/spectrogram_77/"
        root_r = "../../dataset/range/range_resize77/"
        root_v = "../../dataset/velocity/velocity_resize77/"
        model1.load_state_dict(torch.load('../../weights/unet_multi/spectrogram/77GHz_b16_l64.pth'))
        model2.load_state_dict(torch.load('../../weights/unet_multi/range/77GHz_b16_l64.pth'))
        model3.load_state_dict(torch.load('../../weights/unet_multi/velocity/77GHz_b16_l64.pth'))
    elif(radar == 111):
        root = "../../dataset/spectrogram/spectrogram_111/"
        root_r = "../../dataset/range/range_resize111/"
        root_v = "../../dataset/velocity/velocity_resize111/"
        model1.load_state_dict(torch.load('../../weights/unet_multi/spectrogram/111GHz_b16_l64.pth'))
        model2.load_state_dict(torch.load('../../weights/unet_multi/range/111GHz_b16_l64.pth'))
        model3.load_state_dict(torch.load('../../weights/unet_multi/velocity/111GHz_b16_l64.pth'))

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    dataset = ImageFolder(root=root, transform=None)
    dataset_r = ImageFolder(root=root_r, transform=None)
    dataset_v = ImageFolder(root=root_v, transform=None)
    multi_modal_dataset = multi_dataset(dataset, dataset_r, dataset_v, transform)

    train_ratio = 0.8
    num_data = len(multi_modal_dataset)
    indices = list(range(num_data))
    random.shuffle(indices)
    num_train = int(train_ratio * num_data)
    train_indices = indices[:num_train]
    test_indices = indices[num_train:]
    train_dataset = SubsetRandomSampler(train_indices)
    test_dataset = SubsetRandomSampler(test_indices)

    # train_set_size = int(len(dataset) * 0.8)
    # valid_set_size = len(dataset) - train_set_size
    # train_dataset, test_dataset = random_split(
    #     dataset = dataset, 
    #     lengths = [train_set_size, valid_set_size],
    #     generator=torch.Generator().manual_seed(42)
    # )
    
    label_counter = Counter([item[3] for item in multi_modal_dataset])
    for label, count in label_counter.items():
        nj.append(count)

    return model1, model2, model3, multi_modal_dataset, train_dataset, test_dataset, test_dataset, nj

def load_data_weight_test(model1, model2, model3, radar):

    root = []
    root_r = []
    root_v = []
    nj = []

    if(radar == 10):
        root = "../dataset/spectrogram_10/"
        root_r = "../dataset/range_10/"
        root_v = "../dataset/velocity_10/"
        model1.load_state_dict(torch.load('../weights/spectrogram_10.pth'))
        model2.load_state_dict(torch.load('../weights/range_10.pth'))
        model3.load_state_dict(torch.load('../weights/velocity_10.pth'))
    elif (radar == 24):
        root = "../dataset/spectrogram_24/"
        root_r = "../dataset/range_24/"
        root_v = "../dataset/velocity_24/"
        model1.load_state_dict(torch.load('../weights/1/24.pth'))
        model2.load_state_dict(torch.load('../weights/2/24.pth'))
        model3.load_state_dict(torch.load('../weights/3/24.pth'))
    elif(radar == 60):
        root = "../../dataset/spectrogram/spectrogram_60/"
        root_r = "../../dataset/range/range_60/"
        root_v = "../../dataset/velocity/velocity_60/"
        model1.load_state_dict(torch.load('../weights/spectrogram_60.pth'))
        model2.load_state_dict(torch.load('../weights/range_60.pth'))
        model3.load_state_dict(torch.load('../weights/velocity_60.pth'))
    elif(radar == 77):
        root = "../../dataset/spectrogram/spectrogram_77/"
        root_r = "../../dataset/range/range_resize77/"
        root_v = "../../dataset/velocity/velocity_resize77/"
        model1.load_state_dict(torch.load('../../weights/unet_multi/spectrogram/77GHz_b16_l64.pth'))
        model2.load_state_dict(torch.load('../../weights/unet_multi/range/77GHz_b16_l64.pth'))
        model3.load_state_dict(torch.load('../../weights/unet_multi/velocity/77GHz_b16_l64.pth'))
    elif(radar == 111):
        root = "../../dataset/spectrogram/spectrogram_111/"
        root_r = "../../dataset/range/range_resize111/"
        root_v = "../../dataset/velocity/velocity_resize111/"
        model1.load_state_dict(torch.load('../../weights/unet_multi/spectrogram/111GHz_b16_l64.pth'))
        model2.load_state_dict(torch.load('../../weights/unet_multi/range/111GHz_b16_l64.pth'))
        model3.load_state_dict(torch.load('../../weights/unet_multi/velocity/111GHz_b16_l64.pth'))

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    dataset = ImageFolder(root=root, transform=None)
    dataset_r = ImageFolder(root=root_r, transform=None)
    dataset_v = ImageFolder(root=root_v, transform=None)
    multi_modal_dataset = multi_dataset(dataset, dataset_r, dataset_v, transform)

    train_ratio = 0.8
    num_data = len(multi_modal_dataset)
    indices = list(range(num_data))
    random.shuffle(indices)
    num_train = int(train_ratio * num_data)
    train_indices = indices[:num_train]
    test_indices = indices[num_train:]
    train_dataset = SubsetRandomSampler(train_indices)
    test_dataset = SubsetRandomSampler(test_indices)

    # train_set_size = int(len(dataset) * 0.8)
    # valid_set_size = len(dataset) - train_set_size
    # train_dataset, test_dataset = random_split(
    #     dataset = dataset, 
    #     lengths = [train_set_size, valid_set_size],
    #     generator=torch.Generator().manual_seed(42)
    # )
    
    label_counter = Counter([item[3] for item in multi_modal_dataset])
    for label, count in label_counter.items():
        nj.append(count)

    return model1, model2, model3, multi_modal_dataset, train_dataset, test_dataset, test_dataset, nj

class best_score:
    def __init__(self):
        self.best_metrics = {}

    def __call__(self, cm, precision, recall, f1, accuracy, isfinal):
        if (isfinal):
            print(self.best_metrics['cm'])
            print(self.best_metrics['precision'])
            print(self.best_metrics['recall'])
            print(self.best_metrics['f1'])
            print(self.best_metrics['accuracy'])
            
            return self.best_metrics
        elif(self.best_metrics == {} or self.best_metrics['accuracy'] < accuracy):
            self.best_metrics['cm'] = cm
            self.best_metrics['precision'] = precision
            self.best_metrics['recall'] = recall
            self.best_metrics['f1'] = f1
            self.best_metrics['accuracy'] = accuracy

def score(all_predictions, all_labels, istest):
    cm = confusion_matrix(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=1)
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=1)
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=1)
    accuracy = accuracy_score(all_labels, all_predictions)
   
    if istest:
        return cm, precision, recall, f1, accuracy
    else:
        return accuracy

def ShowPic(inputs, outputs, name):
    fig, axs = plt.subplots(2, 16, figsize=(16, 9))
    inputs = inputs.detach().cpu().numpy()
    outputs = outputs.detach().cpu().numpy()
    
    for i in range(16):
        input_image = inputs[i].transpose(1, 2, 0)
        output_image = outputs[i].transpose(1, 2, 0)
        image_combined = np.concatenate([input_image, output_image], axis=1)

        axs[0, i].imshow(input_image)
        axs[0, i].axis('off')
        axs[1, i].imshow(output_image)
        axs[1, i].axis('off')

    plt.savefig(name, bbox_inches='tight', pad_inches=0)
    plt.close()

def DrawLossPlot(valloss_hist, cnt):
    x_axis = range(len(valloss_hist))
    plt.plot(x_axis, valloss_hist, marker='o')
    min_val = min(valloss_hist)
    min_idx = valloss_hist.index(min_val)
    plt.plot(min_idx, min_val, marker='o', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('validation loss')
    plt.title('valloss_hist')
    plt.savefig('./valloss/valloss_hist_plot' + str(cnt) + '.png')
    plt.clf()

class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score):

        # Count Validation Accuracy
        if self.best_score is None:
            self.best_score = current_score
        elif current_score > self.best_score:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1

        if self.counter == self.patience:
            self.early_stop = True

        return self.early_stop

def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)

def FocalLoss(output, label):
    alpha = 0.75
    gamma = 2
    ce_loss = F.cross_entropy(output, label, reduction='none')
    pt = torch.exp(-ce_loss)
    focal = (alpha * (1 - pt) ** gamma * ce_loss).mean()
    
    return focal

def DWBWeight(label, nj, predict):
    nj = torch.tensor(nj).cuda()
    max_n = torch.max(nj)
    weight = (torch.log(max_n / nj[label]) + 1) ** (1 - predict)

    return weight.cuda()

def DWBLoss(output_softmax, predict, label, nj):
    score, _ = torch.topk(output_softmax, k=2, dim=1, largest=True)
    hard_score = torch.sum(score[:, 1:], dim=1)
    weight = DWBWeight(label, nj, predict)
    dwb = -torch.mean(weight * torch.log(predict) - hard_score)

    return dwb

def final_loss(output, output2, label, nj, epoch):

    if(epoch <= 10):
        loss = F.cross_entropy(output, label)

        return loss
    else:
        output_softmax = F.softmax(output, dim=1)
        predict = torch.gather(output_softmax, 1, label.view(-1, 1))
        dwb_loss = DWBLoss(output_softmax, predict, label, nj)
        loss2 = F.cross_entropy(output2, label)

        return dwb_loss + loss2