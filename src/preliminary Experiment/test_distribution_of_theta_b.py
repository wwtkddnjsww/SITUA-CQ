#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from src.utils.sampling import mnist_iid, mnist_noniid, cifar_iid, non_iid, non_iid_preset
from src.utils.options import args_parser
from src.models.Update import *
from src.models.Nets import *
from src.models.Fed import FedAvg
from src.models.test import test_img
from qtorch.auto_low import sequential_lower, lower
from qtorch import FixedPoint, FloatingPoint
from qtorch.quant import float_quantize

from src.preset_distribution_list import *

experiment_iter = 1
num_iter = len(kl_total)  # number of percent
fraction_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

if __name__ == '__main__':
    for diver_iter in range(9,10):
        for clients_fraction in fraction_list:
            for main_iteration in range(1):
                # parse args
                args = args_parser()
                args.frac = clients_fraction
                args.device = torch.device(
                    'cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

                # load dataset and split users
                if args.dataset == 'mnist':
                    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
                    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
                    dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
                    # sample users
                    if args.iid:
                        dict_users = mnist_iid(dataset_train, args.num_users)
                    else:
                        dict_users = non_iid_preset(dataset_train, args.num_users, 600, kl_iid_100client)
                elif args.dataset == 'cifar':
                    trans_cifar = transforms.Compose(
                        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                    dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
                    dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
                    if args.iid:
                        dict_users = cifar_iid(dataset_train, args.num_users)
                    else:
                        dict_users = non_iid_preset(dataset_train, args.num_users, 1000, kl_total[2])

                else:
                    exit('Error: unrecognized dataset')
                img_size = dataset_train[0][0].shape

                act_error_quant = lambda: Quantizer(forward_number=bit_32, backward_number=bit_32,
                                                    forward_rounding="nearest", backward_rounding="nearest")
                act_error_quant_16 = lambda: Quantizer(forward_number=bit_16, backward_number=bit_16,
                                                       forward_rounding="nearest", backward_rounding="nearest")
                act_error_quant_8 = lambda: Quantizer(forward_number=bit_8, backward_number=bit_8,
                                                      forward_rounding="nearest", backward_rounding="nearest")

                # build model
                if args.model == 'cnn' and args.dataset == 'cifar':
                    net_glob = CNNCifar(args=args).to(args.device)
                elif args.model == 'cnn' and args.dataset == 'mnist':
                    net_glob = CNNMnist(args=args).to(args.device)
                elif args.model == 'mlp':
                    len_in = 1
                    for x in img_size:
                        len_in *= x
                    net_glob = CNNQuant(model_name='CNNModel', quant=act_error_quant).to(args.device)
                    net_glob_16 = sequential_lower(net_glob, forward_number=bit_16, backward_number=bit_16,
                                                   forward_rounding="nearest",
                                                   backward_rounding="nearest")
                    net_glob_8 = sequential_lower(net_glob, forward_number=bit_8, backward_number=bit_8,
                                                  forward_rounding="nearest",
                                                  backward_rounding="nearest")
                    # net_glob_16 = MLP_Quant(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes, quant=act_error_quant_16).to(args.device)
                    # net_glob_8 = MLP_Quant(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes, quant=act_error_quant_8).to(args.device)
                else:
                    exit('Error: unrecognized model')

                net_glob.train()
                net_glob_16.train()
                net_glob_8.train()

                # copy weights
                w_glob = net_glob.state_dict()

                # training
                loss_train = []
                cv_loss, cv_acc = [], []
                val_loss_pre, counter = 0, 0
                net_best = None
                best_loss = None
                val_acc_list, net_list = [], []
                acc_history, loss_history = [], []
                if args.all_clients:
                    print("Aggregation over all clients")
                    w_locals = [w_glob for i in range(args.num_users)]
                for iter in range(args.epochs):
                    loss_locals = []
                    if not args.all_clients:
                        w_locals = []
                    m = max(int(args.frac * args.num_users), 1)
                    idxs_users = np.random.choice(range(args.num_users), m, replace=False)

                    bit_idx = 0
                    bit_thresh = int(args.num_users * args.frac)
                    for idx in idxs_users:
                        if bit_idx < bit_thresh:
                            print('bit 32 selected')
                            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device),
                                                  weight_quant=weight_quant, grad_quant=grad_quant,
                                                  momentum_quant=momentum_quant, acc_quant=acc_quant)

                            if args.all_clients:
                                w_locals[idx] = copy.deepcopy(w)
                            else:
                                w_locals.append(copy.deepcopy(w))
                            loss_locals.append(copy.deepcopy(loss))
                        else:
                            print('bit 8 Selected')
                            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                            w, loss = local.train(net=copy.deepcopy(net_glob_8).to(args.device),
                                                  weight_quant=weight_quant_8, grad_quant=grad_quant_8,
                                                  momentum_quant=momentum_quant_8, acc_quant=acc_quant_8)

                            if args.all_clients:
                                w_locals[idx] = copy.deepcopy(w)
                            else:
                                w_locals.append(copy.deepcopy(w))
                            loss_locals.append(copy.deepcopy(loss))
                        bit_idx += 1
                    # update global weights
                    # print(w_locals)
                    w_glob = FedAvg(w_locals)
                    # print(w_glob)
                    # print(w_glob)
                    # copy weight to net_glob
                    net_glob.load_state_dict(w_glob)
                    net_glob_8 = sequential_lower(net_glob, forward_number=bit_8, backward_number=bit_8,
                                                  forward_rounding="nearest", backward_rounding="nearest")
                    net_glob_16 = sequential_lower(net_glob, forward_number=bit_16, backward_number=bit_16,
                                                   forward_rounding="nearest", backward_rounding="nearest")

                    # net_glob_8.load_state_dict(net_glob2.state_dict())

                    # print loss
                    loss_avg = sum(loss_locals) / len(loss_locals)
                    print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
                    loss_train.append(loss_avg)
                    round_acc, round_loss = test_img(net_glob, dataset_test, args)
                    acc_history.append(round_acc)
                    loss_history.append(round_loss)

                # testing
                net_glob.eval()
                acc_train, loss_train = test_img(net_glob, dataset_train, args)
                acc_test, loss_test = test_img(net_glob, dataset_test, args)
                print("Training accuracy: {:.2f}".format(acc_train))
                print("Testing accuracy: {:.2f}".format(acc_test))