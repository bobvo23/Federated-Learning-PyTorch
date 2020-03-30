#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
from sampling import cub_iid, cub_noniid, cub_noniid_hard
import datasets.cub200 as cub
import configs.config as cf


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar10':
        data_dir = '../data/cifar10/'
        apply_transform_train = transforms.Compose(
            [
             transforms.RandomCrop(24),
             transforms.RandomHorizontalFlip(0.5),
             transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

        apply_transform_test = transforms.Compose(
            [transforms.CenterCrop(24),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform_train)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform_test)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or args.dataset == 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    elif args.dataset == 'cub200':
        data_dir = '../data/cub200/'
        apply_transform_train = transforms.Compose([
            transforms.Resize(int(cf.imresize[args.net_type])),
            transforms.RandomRotation(10),
            transforms.RandomCrop(cf.imsize[args.net_type]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
        ])

        apply_transform_test = transforms.Compose([
            transforms.Resize(cf.imresize[args.net_type]),
            transforms.CenterCrop(cf.imsize[args.net_type]),
            transforms.ToTensor(),
            transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
        ])
        train_dataset = cub.CUB200(data_dir, year=2011, train=True, download=True,
                                       transform=apply_transform_train)

        test_dataset = cub.CUB200(data_dir, year=2011, train=False, download=True,
                                      transform=apply_transform_test)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cub_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.hard:
                # Chose uneuqal splits for every user
                user_groups = cub_noniid_hard(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = cub_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        if 'partitionings' in key:
            continue
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def average_weights_split(w, local_mc_w, idxs_classifiers, num_partitionings=100, num_partitions=2 ):
    """
    Returns the average of weights for combinatorial global model.
    shared feature extractor should be averaged over all clients
    but meta-classifiers should be averaged over agents correspond to the meta-classifiers
    """
    w_avg = copy.deepcopy(w[0])

    #histogram = torch.histc(idxs_classifiers, bin=num_partitionings)
    for key in w_avg.keys():
        if 'feature_extractor' in key:
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
            w_avg[key] = torch.div(w_avg[key], len(w))

    for classifier_idx in local_mc_w.keys():
        if len(local_mc_w[classifier_idx]) != 0:
            local_w_avg = copy.deepcopy(local_mc_w[classifier_idx][0])
            for key in local_w_avg.keys():
                for i in range(1, len(local_mc_w[classifier_idx])):
                    local_w_avg[key] += local_mc_w[classifier_idx][i][key]
                w_avg[key] = torch.div(local_w_avg[key], len(local_mc_w[classifier_idx]))

    return w_avg



def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return

def split_weights(w, meta_classifier_idx):
    weight = w
    keys = w.keys()
    fe_dict = {}
    comb_dict = {}
    for key in keys:
        if 'feature_extractor' in key:
            fe_dict[key] = w[key]
        elif str(meta_classifier_idx) in key and str('meta_classifier') in key:
            comb_dict[key] = w[key]
    return fe_dict, comb_dict


def adjust_learning_rate(optimizers, args, epoch):
    # if args.dataset == 'cub200':
    #     if epoch in args.schedule:
    #         args.lr = args.lr * 0.1
    #         for optimizer in optimizers:
    #             for param_group in optimizer.param_groups:
    #                 param_group['lr'] = args.lr
    # else:
    args.lr = args.lr * args.gamma
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr