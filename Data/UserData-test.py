# # -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import torch.nn.init as init
import time
from torch.autograd import Variable
import torchvision
from torchvision import datasets,transforms,utils, models
import torchvision.datasets.mnist as mnist
import numpy as np
import os
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from UserData import UserData
plt.switch_backend('agg')
data_dir='/home/priv-lab1/Database/Ship_cls/images/'
label_txt='/home/priv-lab1/Database/Ship_cls/'
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop (224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

dsets = {x: UserData(img_path=data_dir,
                                    txt_path=(label_txt + x + '.txt'),
                                    data_transforms=data_transforms,
                                    dataset=x) for x in ['test', 'train']}
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=10,
                                               shuffle=True, num_workers=25)#num_workers  用多少个子进程加载数据,这里用了25个子进程
                for x in ['test', 'train']}
data_loader_train=dset_loaders['test']
data_loader_test=dset_loaders['train']
def image_visible():
    images, labels = next(iter(data_loader_train))#得到一个所有地images和label的列表
    print("one batch dim:", images.size())
    # for i, (images, labels) in enumerate(data_loader_train):
    img = utils.make_grid(images[0:4])  # 绘制四个
    # 如果要使用matplotlib显示图片，必须将(channel,height,weight)转换为(height,weight,channel)
    img=img.numpy().transpose(1,2,0)
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    img=img*std+mean
    #数据预览
    print([labels[i] for i in range(4)])
    plt.imshow(img)
    plt.savefig('img_visible.png')
if __name__ == '__main__':
    image_visible()
