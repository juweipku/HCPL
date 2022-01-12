import os
import math
import time
import copy
import random
import shutil
import pickle
import numpy as np
import networkx as nx

import torch
import torch.optim as optim
import torch.nn.functional as F

from utils.utils import *
from utils.train_util import train_initial, test
from utils.pseudo_labeling_util import pseudo_labeling

from arguments import arg_parse
from datasets import *
from aug import permute_edges, mask_nodes
from models.deepergnn import Net
import logging


def create_mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path) 


def main(logger=None):
    # print key configurations
    print('########################################################################')
    print('########################################################################')
    print(f'dataset:                                  {args.dataset}')
    print(f'number of pseudo-labeling iterations:     {args.iterations}')
    print(f'number of epochs:                         {args.epochs}')
    print(f'lr:                                       {args.lr}')
    print(f'value of tau_p:                           {args.tau_p}')
    print(f'value of kappa_p:                         {args.kappa_p}')
    print('########################################################################')
    print('########################################################################')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    if args.seed != -1:
        set_seed(args)

    lcc_mask = None
    if args.dataset == "cora" or args.dataset == "citeseer" or args.dataset == "pubmed":
        dataset = get_planetoid_dataset(args.dataset, args.normalize_features)
        permute_masks = None if args.public_splits else random_planetoid_splits
        # permute_masks = random_planetoid_splits if args.random_splits else None
        if permute_masks is not None:
            data = permute_masks(dataset[0], dataset.num_classes, lcc_mask, args)
        else:
            data = tag_unlabel(dataset[0], args)
        print('dataset.num_classes:', dataset.num_classes)
    elif args.dataset == "cs" or args.dataset == "physics":
        dataset = get_coauthor_dataset(args.dataset, args.normalize_features)
        permute_masks = random_coauthor_amazon_splits
        data = permute_masks(dataset[0], dataset.num_classes, lcc_mask, args)
        print('dataset.num_classes:', dataset.num_classes)
    elif args.dataset == "computers" or args.dataset == "photo":
        dataset = get_amazon_dataset(args.dataset, args.normalize_features)
        permute_masks = random_coauthor_amazon_splits
        # select largest connected component
        data_ori = dataset[0]
        data_nx = to_networkx(data_ori)
        data_nx = data_nx.to_undirected()
        print("Original #nodes:", data_nx.number_of_nodes())
        data_nx = data_nx.subgraph(max(nx.connected_components(data_nx), key=len))
        print("#Nodes after lcc:", data_nx.number_of_nodes())
        lcc_mask = list(data_nx.nodes)
        data = permute_masks(dataset[0], dataset.num_classes, lcc_mask, args)
        print('dataset.num_classes:', dataset.num_classes)

    print('size of train set:', int(data.train_mask.sum().item()))
    print('size of valid set:', int(data.val_mask.sum().item()))
    print('size of test set:', int(data.test_mask.sum().item()))
    print('size of unlabeled set:', int(data.unlabel.sum().item()))
    data.ground_truth = copy.deepcopy(data.y)
    args.num_features = max(dataset.num_features, 1)
    args.num_classes = dataset.num_classes
    print('data.num_nodes:', data.num_nodes)
    # DONE: modify args.select_unlabel
    select_unlabel_base = int(data.unlabel.sum().item()/args.iterations)

    ground_truth_label = data.ground_truth
    best_val_loss = 9999999
    best_val_acc = 0
    best_itr = 0
    for itr in range(args.iterations):
        if itr > 0:
            pseudo_lbl_dict = f'{args.out}/{args.expt_name}/{args.dataset}/{args.dataset}_pseudo_labeling_iteration_{str(itr)}.pkl'
        else:
            pseudo_lbl_dict = None

        if pseudo_lbl_dict is not None:
            data = pseudo_data(data, pseudo_lbl_dict, ground_truth_label)

        # adjust args.select_unlabel dynamically
        train_size = int(data.train_mask.sum().item())
        print('size of train set:', train_size)
        args.select_unlabel = min(select_unlabel_base, int(args.tag_ratio * train_size))
        print('args.select_unlabel:', args.select_unlabel)
        data = data.to(args.device)

        model = Net(args, dataset).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        test_acc = 0
        bad_counter = 0
        best_epoch = -1
        start_epoch = 0
        for epoch in range(start_epoch, args.epochs):
            data_aug = copy.deepcopy(data)
            if args.mask_aug_ratio_train > 0.0:
                data_aug = mask_nodes(data_aug, args.mask_aug_ratio_train)
                data_aug = permute_edges(data_aug, args.edge_aug_ratio_train)
            train_loss, val_loss, acc_val, acc_test = train_initial(args, data_aug, model, optimizer, epoch)

            # if val_loss < best_val_loss:
            if acc_val > best_val_acc:
                best_val_acc = acc_val
                best_val_loss = val_loss
                test_acc = acc_test
                bad_counter = 0
                best_epoch = epoch
                create_mkdir(os.path.join('./save_models/', args.expt_name, args.dataset))
                torch.save(model.state_dict(), os.path.join('./save_models/', args.expt_name, args.dataset,
                                                            args.dataset + str(itr) +'.pkl'))
            else:
                bad_counter += 1

            if bad_counter == args.patience:
                print('Early stop model validation loss: ', best_val_loss, ', accuracy: ', test_acc)
                break

        if best_epoch >= 0:
            best_itr = itr
            print('Loading {}th epoch'.format(best_epoch))
            model.load_state_dict(torch.load(os.path.join('./save_models/', args.expt_name, args.dataset,
                                                            args.dataset + str(itr) +'.pkl')))
        else:
            print('Loading {}th iteration'.format(best_itr))
            model.load_state_dict(torch.load(os.path.join('./save_models/', args.expt_name, args.dataset,
                                                            args.dataset + str(best_itr) +'.pkl')))
        _, best_acc = test(args, data, model)
        logger.info(str(best_acc))

        # pseudo-label generation and selection
        pseudo = pseudo_labeling(args, data, model, itr)
        if pseudo == None:
            return None
        else:
            pseudo_label_dict = pseudo

        create_mkdir(os.path.join('./outputs/', args.expt_name, args.dataset))
        with open(os.path.join('./outputs/', args.expt_name,
                  f'{args.dataset}/{args.dataset}_pseudo_labeling_iteration_{str(itr+1)}.pkl'), "wb") as f:
            pickle.dump(pseudo_label_dict, f)

    shutil.rmtree(os.path.join('./outputs/', args.expt_name, f'{args.dataset}'))
    shutil.rmtree(os.path.join('./save_models/', args.expt_name, args.dataset))


if __name__ == '__main__':
    args = arg_parse()
    create_mkdir(args.log_dir)
    log_path = os.path.join(args.log_dir, args.log_file)
    print('logging into %s' % log_path)

    # DONE: use logging
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # handler.setFormatter(formatter)
    logger.addHandler(handler)

    localtime = time.asctime(time.localtime(time.time()))
    logger.info("%s" % localtime)

    # record arguments
    logger.info("%s" % args.dataset)
    args_str = ""
    for k, v in sorted(vars(args).items()):
        args_str += "%s" % k + "=" + "%s" % v + "; "
    logger.info(args_str)
    logger.info("args.iterations: %s" % args.iterations)
    logger.info("args.hidden: %s" % args.hidden)
    logger.info("args.percentiles_holder: %s" % args.percentiles_holder)
    logger.info("args.uncertainty_percentiles: %s" % args.uncertainty_percentiles)
    logger.info("args.train_class_size: %s" % args.train_class_size)

    main(logger)

    logger.info("\n")

