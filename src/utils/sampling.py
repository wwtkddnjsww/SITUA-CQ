#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import random

from utils.options import args_parser

import numpy as np
from torchvision import datasets, transforms


args = args_parser()

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def non_iid(dataset, num_users, num_data_per_user, alpha = 1.0):
    if args.dataset == 'cifar':
        x_train, y_train = dataset.data, dataset.targets
        y_train = np.array(y_train)

    elif args.dataset == 'mnist':
        x_train, y_train = dataset.data, dataset.targets

    else:
        print("nonono")
    print(y_train)
    num_label = max(y_train)+1
    label_splitted = []
    for k in range(num_label):
        idx_k = np.where(y_train==k)[0]
        print('idx:',idx_k)
        label_splitted.append(idx_k)
    lab_index = [0 for i in range(num_label)]


    user_dist = []
    for i in range(num_users):
        temp  = np.random.dirichlet(np.repeat(alpha,num_label))
        temp = temp*num_data_per_user
        user_dist.append(temp)
    user_dist = np.round(user_dist)
    user_dist = user_dist.astype(np.int64)
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_idx = 0
    for data_dist in user_dist:
        class_idx = 0
        for num_of_user_class_data in data_dist:
            label_len = len(label_splitted[class_idx])
            if (lab_index[class_idx] + num_of_user_class_data) < label_len:
                add_data = label_splitted[class_idx][(lab_index[class_idx]):(lab_index[class_idx] + num_of_user_class_data)]
            else:
                add_data1 = label_splitted[class_idx][(lab_index[class_idx]):label_len]
                add_data2 = label_splitted[class_idx][0:(lab_index[class_idx] + num_of_user_class_data)%label_len]
                add_data = np.concatenate((add_data1,add_data2),axis = 0)
            dict_users[dict_idx] = np.concatenate((dict_users[dict_idx],add_data),axis=0)

            class_idx +=1
        random.shuffle(dict_users[dict_idx])
        dict_idx+=1
    return dict_users

def non_iid_preset(dataset, num_users, num_data_per_user,user_dist):
    if args.dataset == 'cifar':
        x_train, y_train = dataset.data, dataset.targets
        y_train = np.array(y_train)

    elif args.dataset == 'mnist':
        x_train, y_train = dataset.data, dataset.targets

    else:
        print("nonono")
    print(y_train)
    num_label = max(y_train)+1
    label_splitted = []
    for k in range(num_label):
        idx_k = np.where(y_train==k)[0]
        print('idx:',idx_k)
        label_splitted.append(idx_k)
    lab_index = [0 for i in range(num_label)]
    for lab in range(len(label_splitted)):
        random.shuffle(label_splitted[lab])

    user_dist = user_dist[:int(args.num_users)]

    user_dist = np.array(user_dist)*num_data_per_user
    user_dist = np.round(user_dist)
    user_dist = user_dist.astype(np.int64)
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_idx = 0
    for data_dist in user_dist:
        class_idx = 0
        for num_of_user_class_data in data_dist:
            label_len = len(label_splitted[class_idx])
            if (lab_index[class_idx] + num_of_user_class_data) < label_len:
                add_data = label_splitted[class_idx][(lab_index[class_idx]):(lab_index[class_idx] + num_of_user_class_data)]

            else:
                last_idx = (lab_index[class_idx] + num_of_user_class_data)%label_len
                add_data1 = label_splitted[class_idx][(lab_index[class_idx]):label_len]
                add_data2 = label_splitted[class_idx][0:last_idx]
                add_data = np.concatenate((add_data1,add_data2),axis = 0)
            lab_index[class_idx] = (lab_index[class_idx] + num_of_user_class_data)
            lab_index[class_idx] = lab_index[class_idx]%label_len
            dict_users[dict_idx] = np.concatenate((dict_users[dict_idx],add_data),axis=0)

            class_idx +=1
        random.shuffle(dict_users[dict_idx])
        dict_idx+=1
    return dict_users

def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]


    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    print(dict_users)
    return dict_users


def cifar_iid(dataset, num_users,times=1):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)*times
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
