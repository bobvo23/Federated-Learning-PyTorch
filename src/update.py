#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from configs import config as cf
import torch.nn.functional as F
from utils import adjust_learning_rate


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    Splitting a subset of the dataset based on the index of the user
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        #return torch.tensor(image), torch.tensor(label)
        return image.clone().detach().requires_grad_(True), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        #idxs all samples of that user
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        #debug
        print("Local device: ", self.device, args.gpu)
        # Default criterion set to NLL loss function
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        #self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        #idxs_train = idxs[:]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]
        #TODO: 3 implement, max batchsize = max dataset size
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        '''
        Update weights for 1 user
        :param model: current global model
        '''
        
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=cf.momentum[self.args.dataset], weight_decay=cf.weight_decay[self.args.dataset])
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)


        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                #log_probs = model(images)
                outputs = model(images)
                #log_probs = F.log_softmax(outputs, dim=1)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tLr: {:.4f}'.format(
                        global_round, iter, (batch_idx+1) * len(images),
                        len(self.trainloader.dataset),
                        100. * (batch_idx+1) / len(self.trainloader), loss.item(), self.args.lr))
                #self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss

#Comment out Combi part
# class LocalUpdate_combi(LocalUpdate):
#     def __init__(self, args, dataset, idxs, logger):
#         super(LocalUpdate_combi, self).__init__(args, dataset, idxs, logger)
#         self.criterion = nn.NLLLoss().to(self.device)

#     def update_weights(self, model, global_round, classifier_idx):
#         # Set mode to train model
#         model.train()
#         epoch_loss = []

#         # Set optimizer for the local updates
#         if self.args.optimizer == 'sgd':
#             optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
#                                         momentum=cf.momentum[self.args.dataset], weight_decay=cf.weight_decay[self.args.dataset])
#         elif self.args.optimizer == 'adam':
#             optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
#                                          weight_decay=1e-4)

#         for iter in range(self.args.local_ep):
#             batch_loss = []
#             for batch_idx, (images, labels) in enumerate(self.trainloader):
#                 images, labels = images.to(self.device), labels.to(self.device)

#                 model.zero_grad()
#                 optimizer.zero_grad()
#                 log_probs = model(images, classifier_idx)
#                 loss = self.criterion(log_probs, labels)
#                 loss.backward()
#                 optimizer.step()

#                 if self.args.verbose and (batch_idx % 10 == 0):
#                     print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLr: {:.4f}'.format(
#                         global_round, iter, batch_idx * len(images),
#                         len(self.trainloader.dataset),
#                         100. * batch_idx / len(self.trainloader), loss.item(), self.args.lr))
#                 #self.logger.add_scalar('loss', loss.item())
#                 batch_loss.append(loss.item())
#             epoch_loss.append(sum(batch_loss)/len(batch_loss))

#         return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

#     def inference(self, model):
#         """ Returns the inference accuracy and loss.
#         """

#         model.eval()
#         loss, total, correct = 0.0, 0.0, 0.0

#         for batch_idx, (images, labels) in enumerate(self.testloader):
#             images, labels = images.to(self.device), labels.to(self.device)

#             # Inference
#             outputs = model(images)
#             batch_loss = self.criterion(outputs, labels)
#             loss += batch_loss.item()

#             # Prediction
#             _, pred_labels = torch.max(outputs, 1)
#             pred_labels = pred_labels.view(-1)
#             correct += torch.sum(torch.eq(pred_labels, labels)).item()
#             total += len(labels)

#         accuracy = correct/total
#         return accuracy, loss


def test_inference(args, model, test_dataset, combi=False):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    if combi:
        meta_correct = torch.zeros(args.num_partitionings, dtype=torch.long).to(device)

    if args.dataset == 'cub200':
        if combi:
            criterion = nn.NLLLoss().to(device)
        else:
            criterion = nn.CrossEntropyLoss().to(device)
        testloader = DataLoader(test_dataset, batch_size=32,
                                shuffle=False)
    else:
        if combi:
            criterion = nn.NLLLoss().to(device)
        else:
            criterion = nn.CrossEntropyLoss().to(device)
        testloader = DataLoader(test_dataset, batch_size=128,
                                shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        if combi:
            meta_outputs = outputs
            outputs = meta_outputs.sum(dim=1)

            # accuracies per meta-classifiers
            max_value, _ = torch.max(meta_outputs, 2)
            idx = torch.stack([labels.unsqueeze(1) for i in range(args.num_partitionings)], dim=1)
            meta_labels = torch.gather(meta_outputs, 2, idx).squeeze()
            meta_correct += torch.eq(max_value, meta_labels).sum(0)


        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)




    accuracy = correct*100/total
    if combi:
        accuracy_per_meta_classifiers = meta_correct*100/total
        return accuracy, loss, accuracy_per_meta_classifiers
    else:
        return accuracy, loss

# class LocalUpdate_combi_tc(LocalUpdate):
    def __init__(self, args, dataset, idxs, logger):
        super(LocalUpdate_combi_tc, self).__init__(args, dataset, idxs, logger)
        self.criterion = nn.NLLLoss().to(self.device)

    def update_weights(self, model, global_round, classifier_idx):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=args.momentum, weight_decay=cf.weight_decay[self.args.dataset])
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                #optimizer.zero_grad()
                log_probs = model(images)
                log_probs = log_probs.sum(dim=1)
                loss = self.criterion(log_probs, labels) / self.args.num_partitionings
                loss.backward()
                model.comb_classifier.rescale_grad()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLr: {:.4f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item(), self.args.lr))
                #self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
#TODO:2 seems that test lost is not normalized, need to fix?
        accuracy = correct/total
        return accuracy, loss