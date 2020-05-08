#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F
import configs.config as cf


class MLP(nn.Module):
    #Modeling after the FedAvg Model to compare the performance
    #2 hidden layers with 200 hidden units each
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu1 = nn.ReLU()
        #self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_hidden)
        self.relu2 = nn.ReLU()
        self.layer_hidden2 = nn.Linear(dim_hidden, dim_out)
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        #x = self.dropout(x)
        x = self.relu1(x)
        x = self.layer_hidden(x)
        x = self.relu2(x)
        x = self.layer_hidden2(x)
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
        #return F.log_softmax(x, dim=1)
        return x


class CNNFashion_Mnist(nn.Module):
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 3 * 3, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, args.num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #return F.log_softmax(x, dim=1)
        return x


class CNNCifar_combi(nn.Module):
    def __init__(self, args):
        super(CNNCifar_combi, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        #self.fc3 = nn.Linear(84, args.num_classes)
        self.fc3 = CombinatorialClassifier(args.num_classes, args.num_partitionings, args.num_partitions, 84,
                                           local_partitionings=args.local_partitionings)

    def forward(self, x, classifier_idx):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x, classifier_idx)
        return F.log_softmax(x, dim=1)


class CNNComb(nn.Module):
    def __init__(self, args, fe, feat_dim, partitionings):
        super(CNNComb, self).__init__()
        self.args = args
        self.feature_extractor = fe
        self.feat_dim = feat_dim
        #self.partitionings = partitionings
        self.comb_classifier = CombinatorialClassifier(cf.num_classes[self.args.dataset], self.args.num_partitionings,
                                                        self.args.num_partitions, self.feat_dim, additive=False, attention=False)
        self.comb_classifier.set_partitionings(partitionings)
        #self.comb_classifier = nn.Linear(192, args.num_classes)

    def forward(self, x, classifier_idx=None):
        x = self.feature_extractor(x)
        if classifier_idx is not None:
            x = self.comb_classifier(x, classifier_idx)
            #x = self.comb_classifier(x)
        else:
            x = self.comb_classifier(x)

        return x