#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
import wandb
import torch
import errno
import random
import torch.backends.cudnn as cudnn
import torchvision.models as models
import configs.config as cf

from tqdm import tqdm

from tensorboardX import SummaryWriter


from options.fed_combi_options import args_parser
from update import LocalUpdate_combi, test_inference, LocalUpdate_combi_tc
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar_combi, CNNCifar, CNNComb
from utils import get_dataset, exp_details, split_weights, average_weights_split, average_weights
from combclassifier import CombinatorialClassifier

args = args_parser()
wandb.init(project='federated_combinatorial')


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
device = 'cuda' if args.gpu else 'cpu'

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

# Use CUDA
if use_cuda:
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True
    cudnn.deterministic = False

args.partitionings_path = '../data/%s_partitions/part_%d_kmeans.pth.tar' % \
                          (args.dataset, args.num_partitions)

#args.lr = cf.lr[args.dataset]
args.local_bs = cf.train_batch[args.dataset]

wandb.config.update(args, allow_val_change=True)

def main():
    start_time = time.time()

    # define paths
    model_path = 'results/%s/%s/iid_%d/%s/seed_%d' % (args.dataset, args.method, args.iid, args.net_type, args.seed)
    if not os.path.isdir(model_path):
        mkdir_p(model_path)
    if not os.path.isdir(os.path.join(model_path, 'logs')):
        mkdir_p(os.path.join(model_path, 'logs'))
    logger = SummaryWriter(os.path.join(model_path, 'logs'))

    exp_details(args)

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar10':
            _model = CNNCifar(args=args)
            feat_dim = _model.fc3.in_features
            #_model = models.resnet18(pretrained=True)
            #feat_dim = _model.fc.in_features
            _model.fc3 = torch.nn.Sequential()
            #global_model = CNNCifar_combi(args=args)
        elif args.dataset == 'cub200':
            if args.net_type == 'resnet':
                _model = models.resnet50(pretrained=True)
                feat_dim = _model.fc.in_features
                _model.fc = torch.nn.Sequential()

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
    assert os.path.isfile(args.partitionings_path), 'Error: no partitionings found!'
    partitionings = torch.load(args.partitionings_path)[args.seed * args.num_partitionings + 20
                                                        :(args.seed+1)*args.num_partitionings + 20].t()

    global_model = CNNComb(args=args, fe=_model, feat_dim=feat_dim, partitionings=partitionings)
    print('    Total params: %.2fM' % (sum(p.numel() for p in global_model.parameters()) / 1000000.0))



    global_model.to(device)
    global_model.train()
    print(global_model)
    wandb.watch(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    test_acc_lst = []
    print_every = 2
    val_loss_pre, counter = 0, 0
    best_acc = 0


    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        local_meta_classifier_weights = {str(i): [] for i in range(args.num_partitionings)}
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        #idxs_classifiers = idxs_users % args.num_partitionings
        idxs_classifiers = np.random.choice(range(args.num_partitionings), m, replace=False)

        if epoch != 0:
            args.lr *= args.gamma

        for idx, meta_classifier_idx in zip(idxs_users, idxs_classifiers):
            local_model = LocalUpdate_combi_tc(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch, classifier_idx=meta_classifier_idx)

            #w_fe, w_metaclassifier = split_weights(w, meta_classifier_idx)
            local_weights.append(copy.deepcopy(w))
            #local_meta_classifier_weights[str(meta_classifier_idx)].append(copy.deepcopy(w_metaclassifier))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        #global_weights = average_weights_split(local_weights, local_meta_classifier_weights, idxs_classifiers)
        global_weights = average_weights(local_weights)
        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        # global_model.eval()
        # for c in range(args.num_users):
        #     local_model = LocalUpdate_combi(args=args, dataset=train_dataset,
        #                               idxs=user_groups[c], logger=logger)
        #     acc, loss = local_model.inference(model=global_model)
        #     list_acc.append(acc)
        #     list_loss.append(loss)
        # train_accuracy.append(sum(list_acc)/len(list_acc))

        test_acc, test_loss, test_meta_acc = test_inference(args, global_model, test_dataset, combi=True)
        test_acc_lst.append(test_acc)

        # save model
        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': global_model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'meta_classifiers_acc': test_meta_acc
            }, dir=model_path, filename='checkpoint.pth.tar')

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            #print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
            print('Test Accuracy: {:.2f}%, Best Test Acc: {:.2f}%, Mean Test Meta Acc: {:.2f}% \n'.format(test_acc, best_acc, np.mean(test_meta_acc.cpu().numpy())))

        wandb.log({
            "Train Loss": loss_avg,
            "Test Acc": test_acc,
            "lr": args.lr,
            "Test Meta Acc": np.mean(test_meta_acc.cpu().numpy()),
            'Best Acc': best_acc
        })

        if best_acc >= args.target_acc:
            print('Total Global round: ', epoch+1)
            break

    # Test inference after completion of training
    test_acc, test_loss, _ = test_inference(args, global_model, test_dataset, combi=True)

    print(f' \n Results after {args.epochs} global rounds of training:')
    #print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(test_acc))
    print("|---- Best Test Accuracy: {:.2f}%".format(best_acc))

    # # Saving the objects train_loss and train_accuracy:
    # file_name = './save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
    #     format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #            args.local_ep, args.local_bs)
    #
    # with open(file_name, 'wb') as f:
    #     pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))

    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Test Accuracy vs Communication rounds')
    plt.plot(range(len(test_acc_lst)), test_acc_lst, color='k')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))


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