import os.path as osp
import pickle
import numpy as np
import torch
from torch.utils.data import random_split
from torch_geometric.datasets import Planetoid, Coauthor, Amazon, WikiCS
import torch_geometric.transforms as T
from torch_geometric.utils import *


def get_planetoid_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', name)
    dataset = Planetoid(path, name, split="public")

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform
    return dataset


def get_coauthor_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', name)
    dataset = Coauthor(path, name)
    
    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform
    return dataset


def get_amazon_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', name)
    dataset = Amazon(path, name)
    
    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform
    return dataset


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, num_classes, lcc_mask, args):
    # Set new random planetoid splits:
    # * 20 * num_classes labels for training
    # * 500 labels for validation
    # * 1000 labels for testing

    indices = []
    if lcc_mask is not None:
        for i in range(num_classes):
            index = (data.y[lcc_mask] == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)
    else:
        for i in range(num_classes):
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

    train_index = torch.cat([i[:args.train_class_size] for i in indices], dim=0)
    val_index = torch.cat([i[args.train_class_size:60] for i in indices], dim=0)

    rest_index = torch.cat([i[60:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index, size=data.num_nodes)
    # DONE: unlabel = val + test
    data.unlabel = torch.logical_or(data.val_mask, data.test_mask)
    data.indexs = torch.arange(data.num_nodes)
    return data


def tag_unlabel(data, args):
    if not args.unlabel_part:
        data.unlabel = ~data.train_mask
    else:
        data.unlabel = torch.logical_or(data.val_mask, data.test_mask)
    data.indexs = torch.arange(data.num_nodes)
    return data

    
def random_coauthor_amazon_splits(data, num_classes, lcc_mask, args):
    # Set random coauthor/co-purchase splits:
    # * 20 * num_classes labels for training
    # * 30 * num_classes labels for validation
    # rest labels for testing

    indices = []
    if lcc_mask is not None:
        for i in range(num_classes):
            index = (data.y[lcc_mask] == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)
    else:
        for i in range(num_classes):
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

    train_index = torch.cat([i[:args.train_class_size] for i in indices], dim=0)
    val_index = torch.cat([i[args.train_class_size:60] for i in indices], dim=0)

    rest_index = torch.cat([i[60:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index, size=data.num_nodes)
    # DONE: unlabel = val + test
    data.unlabel = torch.logical_or(data.val_mask, data.test_mask)
    data.indexs = torch.arange(data.num_nodes)
    return data


def generate_split(num_samples: int, train_ratio: float, val_ratio: float):
    train_len = int(num_samples * train_ratio)
    val_len = int(num_samples * val_ratio)
    test_len = num_samples - train_len - val_len

    train_set, test_set, val_set = random_split(torch.arange(0, num_samples), (train_len, test_len, val_len))

    idx_train, idx_test, idx_val = train_set.indices, test_set.indices, val_set.indices
    train_mask = torch.zeros((num_samples,)).to(torch.bool)
    test_mask = torch.zeros((num_samples,)).to(torch.bool)
    val_mask = torch.zeros((num_samples,)).to(torch.bool)

    train_mask[idx_train] = True
    test_mask[idx_test] = True
    val_mask[idx_val] = True
    return train_mask, test_mask, val_mask


def pseudo_data(data, pseudo_lbl_dict, ground_truth_label):
    pseudo_lbl_dict = pickle.load(open(pseudo_lbl_dict, 'rb'))
    pseudo_idx = pseudo_lbl_dict['pseudo_idx']
    pseudo_target = pseudo_lbl_dict['pseudo_target']
    lbl_idx = data.indexs[data.train_mask].tolist()
    assert len(list(set(pseudo_idx).intersection(set(lbl_idx)))) == 0
    lbl_idx = np.array(lbl_idx + pseudo_idx)

    train_index = torch.tensor(lbl_idx)
    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    unlabel_index = data.indexs[data.unlabel].tolist()
    unlabel_index = torch.tensor(list(set(unlabel_index).difference(set(pseudo_idx))))
    print(f'Remaining Unlabeled Data: {len(unlabel_index)}')
    data.unlabel = index_to_mask(unlabel_index, size=data.num_nodes)
    target = data.y.tolist()
    for i, idx in enumerate(pseudo_idx):
        target[idx] = pseudo_target[i]
    data.y = torch.tensor(target, device=data.y.device)
    return data
