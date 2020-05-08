
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=2,
                        help="number of rounds of training")
    parser.add_argument('--target-acc', type=float, default=86.00,
                        help='early stopping criteria')
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='fraction of clients that participate in each training round: C')
    parser.add_argument('--local_ep', type=int, default=5,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=16,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='learning rate decay')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='models names: mlp, cnn...')
    parser.add_argument('--net-type', type=str, default='resnet', help='submodel name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel for cnn option')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset: mnist,fmnist,cifar10...")
    #TODO:1 clean up this option, seem redundant or complicated 
    parser.add_argument('--method', type=str, default='no-fed/baseline', help="name \
                            of method")
    #toclarify:2 is it the output classes of the dataset? can we get it from the data instead?
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID, i.e. 0 . Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer: sgd, adam")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    #toclarify: is it the patient param or mamimum round?
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--workers', type=int, default=4, help='num of workers')
    #toclarify: how to use nargs and the list?
    parser.add_argument('--schedule', type=int, nargs='+', default=[20, 30],
                        help='Decrease learning rate at these epochs.')

    args = parser.parse_args()
    return args
