#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


from tqdm import tqdm
import matplotlib.pyplot as plt
import errno
import torch
import torch.backends.cudnn as cudnn
import os
import wandb
from torch.utils.data import DataLoader
import torchvision.models as models
import random
import numpy as np

from utils import get_dataset, adjust_learning_rate
from opt.single_base_options import args_parser
from update import test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
import configs.config as cf

args = args_parser()
#if args.gpu:
#    torch.cuda.set_device(args.gpu)



wandb.init(project='federated_combinatorial')
wandb.config.update(args, allow_val_change=True)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
device = 'cuda' if args.gpu else 'cpu'

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def main():

    model_path = 'results/%s/%s/%s/seed_%d' % (args.dataset, args.method, args.net_type, args.seed)
    if not os.path.isdir(model_path):
        mkdir_p(model_path)
    # load datasets
    train_dataset, test_dataset, _ = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar10':
            global_model = CNNCifar(args=args)
        elif args.dataset == 'cub200':
            if args.net_type == 'resnet':
                global_model = models.resnet50(pretrained=True)
                global_model.fc = torch.nn.Linear(global_model.fc.in_features, cf.num_classes[args.dataset])
    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)
    wandb.watch(global_model)

    # Training
    # Set optimizer and criterion
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                    momentum=cf.momentum[args.dataset], weight_decay=5e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr,
                                     weight_decay=1e-4)

    trainloader = DataLoader(train_dataset, batch_size=args.local_bs * (args.num_users * args.frac), shuffle=True, num_workers=args.workers,
                             pin_memory=use_cuda, drop_last=True)
    if args.dataset == 'cub200':
        criterion = torch.nn.CrossEntropyLoss().to(device)
    else:
        criterion = torch.nn.NLLLoss().to(device)
    epoch_loss = []
    test_acc_lst = []
    best_acc = 0
    args.lr = cf.lr[args.dataset]

    for epoch in tqdm(range(args.epochs)):
        global_model.train()
        batch_loss = []

        # adjest learning rate per global round
        if epoch != 0:
            adjust_learning_rate([optimizer], args, epoch)

        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = global_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            #if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLr: {:.4f}'.format(
                    epoch+1, batch_idx * len(images), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item(), args.lr))
            batch_loss.append(loss.item())

            wandb.log({'Train Loss': loss.item()})

        loss_avg = sum(batch_loss)/len(batch_loss)
        print('\nTrain loss: \n', loss_avg)
        epoch_loss.append(loss_avg)

        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        test_acc_lst.append(test_acc)


        #save model
        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': global_model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, dir=model_path, filename='checkpoint.pth.tar')

        print('\nTrain Epoch: {}, Test acc: {:.2f}%, Best Test acc: {:.2f}%'.format(epoch + 1, test_acc, best_acc))

        # log training loss, test accuracy at wandb
        wandb.log({'Test Acc': test_acc,
                   'Best Acc': best_acc})

        # if model achieves target test acc, stop training
        if best_acc >= args.target_acc:
            print('Total Global round: ', epoch+1)
            break

    if not os.path.isdir(os.path.join(model_path, 'save')):
        mkdir_p(os.path.join(model_path, 'save'))
    # Plot loss
    plt.figure()
    plt.plot(range(len(epoch_loss)), epoch_loss)
    plt.xlabel('epochs')
    plt.ylabel('Train loss')
    plt.savefig(os.path.join(model_path, 'save/nn_{}_{}_{}_loss.png'.format(args.dataset, args.model,
                                                 args.epochs)))

    # Plot test acc per epoch
    plt.figure()
    plt.plot(range(len(test_acc_lst)), test_acc_lst)
    plt.xlabel('epochs')
    plt.ylabel('Test accuracy')
    plt.savefig(os.path.join(model_path, 'save/nn_{}_{}_{}_acc.png'.format(args.dataset, args.model,
                                                                            args.epochs)))
    # testing
    #test_acc, test_loss = test_inference(args, global_model, test_dataset)
    print('Test on', len(test_dataset), 'samples')
    print("Best Test Accuracy: {:.2f}%".format(best_acc))



def save_checkpoint(state, dir, filename):
    filepath = os.path.join(dir, filename)
    torch.save(state, filepath)

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >236_comb_fromZeroNoise.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

if __name__ == '__main__':
    main()