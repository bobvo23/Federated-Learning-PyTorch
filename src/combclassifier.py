from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from math import *
import numpy as np
import copy


class CombinatorialClassifier(nn.Module):
    partition_weight = None

    def __init__(self, num_classes, num_partitionings, num_partitions, feature_dim, additive=False, attention=False,
                 mode='softmax', combination='logit', local_partitionings=1):
        super(CombinatorialClassifier, self).__init__()
        #self.classifiers = nn.Linear(feature_dim, num_partitions * num_partitionings)
        self.classifiers = nn.ModuleDict({'meta_classifier_%d' % i: nn.Linear(feature_dim, num_partitions)for i in range(num_partitionings)} )
        self.num_classes = num_classes
        self.num_partitionings = num_partitionings
        self.num_partitions = num_partitions
        self.attention = attention
        self.mode = mode
        self.combination = combination
        self.local_partitionings = local_partitionings
        #self.layer_norm = nn.LayerNorm(num_classes, eps=1e-6, elementwise_affine=False)
        if self.attention:

            self.AtModule = nn.Sequential(
                nn.Linear(feature_dim, num_partitionings // 4, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(num_partitionings // 4, num_partitionings, bias=False),
                #nn.Softmax()
            )
            print("attention module activated")
        #Adds a persistent buffer to the module.
        #This is typically used to register a buffer that should not to be considered a model parameter.
        #For example, BatchNorm’s running_mean is not a parameter, but is part of the persistent state.

        self.register_buffer('partitionings', -torch.ones(num_partitionings, num_classes).long())
        self.register_buffer('partitionings_inference', -torch.ones(num_partitionings, num_classes).long())

        self.additive = additive
        print("mode : ", self.mode, 'combination : ', self.combination)

    def set_partitionings(self, partitionings_map):
        self.partitionings.copy_(torch.LongTensor(partitionings_map).t())
        arange = torch.arange(self.num_partitionings).view(-1, 1).type_as(self.partitionings)
        #arange를 더해준다.? -> 01110, 23332
        self.partitionings_inference = torch.add(self.partitionings, (arange * self.num_partitions))

    def rescale_grad(self):
        for params in self.classifiers.parameters():
            if self.partition_weight is None:
                params.grad.mul_(self.num_partitionings)
            else:
                params.grad.mul_(self.partition_weight.sum())

    def forward(self, input, classifier_idx=None, output_sum=True, return_meta_dist=False, with_feat=False):
        assert self.partitionings.sum() > 0, 'Partitionings is never given to the module.'

        if classifier_idx is not None:
            all_output = self.classifiers['meta_classifier_%d' % classifier_idx](input)
            all_output = all_output.view(-1, self.local_partitionings, self.num_partitions)

            all_output = F.log_softmax(all_output, dim=2)

            all_output = all_output.view(-1, self.local_partitionings * self.num_partitions)
            output = all_output.index_select(1, self.partitionings[classifier_idx].view(-1))
            output = output.view(-1, self.local_partitionings, self.num_classes)

        else:
            outputs = []
            for i in range(self.num_partitionings):
                meta_output = self.classifiers['meta_classifier_%d' % i](input)
                outputs.append(meta_output)
            all_output = torch.cat(outputs, dim=1)
            all_output = all_output.view(-1, self.num_partitionings, self.num_partitions)

            all_output = F.log_softmax(all_output, dim=2)

            all_output = all_output.view(-1, self.num_partitionings * self.num_partitions)
            output = all_output.index_select(1, self.partitionings_inference.view(-1))
            output = output.view(-1, self.num_partitionings, self.num_classes)

            return output

        if output_sum:
            output = output.sum(1)

        return output


