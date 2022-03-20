#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import tqdm
from qtorch.quant import quantizer, Quantizer
from qtorch import FloatingPoint
from qtorch.optim import OptimLP
import torch.nn.functional as F



bit_8 = FloatingPoint(exp=5, man=2)
bit_16 = FloatingPoint(exp=5, man=10)
bit_32 = FloatingPoint(exp=8, man=23)

#8bit quantizer
weight_quant_8 = quantizer(forward_number=bit_8,
                        forward_rounding="nearest")
grad_quant_8 = quantizer(forward_number=bit_8,
                        forward_rounding="nearest")
momentum_quant_8 = quantizer(forward_number=bit_8,
                        forward_rounding="stochastic")
acc_quant_8 = quantizer(forward_number=bit_8,
                        forward_rounding="stochastic")

#16bit quantizer
weight_quant_16 = quantizer(forward_number=bit_16,
                        forward_rounding="nearest")
grad_quant_16 = quantizer(forward_number=bit_16,
                        forward_rounding="nearest")
momentum_quant_16 = quantizer(forward_number=bit_16,
                        forward_rounding="stochastic")
acc_quant_16 = quantizer(forward_number=bit_16,
                        forward_rounding="stochastic")
#32bit quantizer
weight_quant = quantizer(forward_number=bit_32,
                        forward_rounding="nearest")
grad_quant = quantizer(forward_number=bit_32,
                        forward_rounding="nearest")
momentum_quant = quantizer(forward_number=bit_32,
                        forward_rounding="stochastic")
acc_quant = quantizer(forward_number=bit_32,
                        forward_rounding="stochastic")


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net, weight_quant, grad_quant, momentum_quant, acc_quant):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        optimizer = OptimLP(optimizer,
                            weight_quant=weight_quant,
                            grad_quant=grad_quant,
                            momentum_quant=momentum_quant,
                            acc_quant=acc_quant,
                            #grad_scaling=1 / 1000  # do loss scaling
                            )

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def train_sequential(self, net, weight_quant, grad_quant, momentum_quant, acc_quant):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        optimizer = OptimLP(optimizer,
                            weight_quant=weight_quant,
                            grad_quant=grad_quant,
                            momentum_quant=momentum_quant,
                            acc_quant=acc_quant,
                            #grad_scaling=1 / 1000  # do loss scaling
                            )

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net, sum(epoch_loss) / len(epoch_loss)

