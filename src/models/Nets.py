#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
import argparse
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from qtorch.quant import Quantizer, quantizer
from qtorch.optim import OptimLP
from torch.optim import SGD
from qtorch import FloatingPoint
from tqdm import tqdm
import math
from utils.options import args_parser

args=args_parser()

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x

class MLP_Quant(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, quant):
        super(MLP_Quant, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.quant=quant()

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.quant(x)
        #x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        x = self.quant(x)
        return x


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

cfg={
    'CNNModel':[32,32,'M',64,64,'M',128,128,'M']
}

class CNNQuant(nn.Module):
    def __init__(self,model_name,quant):
        super(CNNQuant,self).__init__()
        self.quant = quant()
        self.conv_1 = nn.Conv2d(3,32,kernel_size=3,padding=1)
        self.conv_2 = nn.Conv2d(32,32,kernel_size=3,padding=1)
        self.conv_3 = nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.conv_4 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.conv_5 = nn.Conv2d(64,128,kernel_size=3,padding=1)
        self.conv_6 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax()
        self.dropout=nn.Dropout(p=0.5)
        self.maxpool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.features = self._make_layers(cfg[model_name])
        self.flatten = nn.Linear(2048,512)
        self.flatten2 = nn.Linear(4096,512)
        self.classifier = nn.Linear(512, 10)
        self.classifier2 = nn.Linear(512, 10)


    def forward(self, x):
        out = self.conv_1(x)
        out = self.relu(out)
        out = self.quant(self.conv_2(out))
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.dropout(out)
        out = self.quant(self.conv_3(out))
        out = self.relu(out)
        out = self.quant(self.conv_4(out))
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.dropout(out)
        out = self.quant(self.conv_5(out))
        out = self.relu(out)
        out = self.quant(self.conv_6(out))
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.dropout(out)
        out = out.view(out.size(0),-1)
        #print(np.shape(out))
        out = self.quant(self.flatten(out))
        #out = self.softmax(out)
        out = self.classifier(out)
        return out

    def _make_layers(self,cfg):
        layers = []
        in_channels=3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2),
                           nn.Dropout(p=0.5)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3,padding=1),
                           nn.ReLU(inplace=True),
                           ]
                in_channels=x
        return nn.Sequential(*layers)

class CNNQuant100(nn.Module):
    def __init__(self,model_name,quant):
        super(CNNQuant100,self).__init__()
        self.quant = quant()
        self.conv_1 = nn.Conv2d(3,32,kernel_size=3,padding=1)
        self.conv_2 = nn.Conv2d(32,32,kernel_size=3,padding=1)
        self.conv_3 = nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.conv_4 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.conv_5 = nn.Conv2d(64,128,kernel_size=3,padding=1)
        self.conv_6 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax()
        self.dropout=nn.Dropout(p=0.5)
        self.maxpool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.features = self._make_layers(cfg[model_name])
        self.flatten = nn.Linear(2048,512)
        self.flatten2 = nn.Linear(4096,512)
        self.classifier = nn.Linear(512, 100)
        self.classifier2 = nn.Linear(512, 100)


    def forward(self, x):
        out = self.conv_1(x)
        out = self.relu(out)
        out = self.quant(self.conv_2(out))
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.dropout(out)
        out = self.quant(self.conv_3(out))
        out = self.relu(out)
        out = self.quant(self.conv_4(out))
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.dropout(out)
        out = self.quant(self.conv_5(out))
        out = self.relu(out)
        out = self.quant(self.conv_6(out))
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.dropout(out)
        out = out.view(out.size(0),-1)
        #print(np.shape(out))
        out = self.quant(self.flatten(out))
        #out = self.softmax(out)
        out = self.classifier(out)
        return out

    def _make_layers(self,cfg):
        layers = []
        in_channels=3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2),
                           nn.Dropout(p=0.5)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3,padding=1),
                           nn.ReLU(inplace=True),
                           ]
                in_channels=x
        return nn.Sequential(*layers)
