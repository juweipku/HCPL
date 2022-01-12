import random
import time
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
# from .misc import AverageMeter, accuracy


def train_initial(args, data, model, optimizer, epoch):
    model.train()
    optimizer.zero_grad()

    data = data.to(args.device)
    logits = model(args, data)

    loss_train = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask], reduction='mean')
    # loss_train = F.nll_loss(logits[data.train_mask], data.y[data.train_mask])
    loss_train.backward()
    optimizer.step()
    
    acc_train = accuracy(logits[data.train_mask], data.y[data.train_mask])

    
    model.eval()
    output = model(args, data)
    loss_val = F.cross_entropy(output[data.val_mask], data.ground_truth[data.val_mask], reduction='mean')
    # loss_val = F.nll_loss(output[data.val_mask], data.y[data.val_mask])
    acc_val = accuracy(output[data.val_mask], data.ground_truth[data.val_mask])
    acc_test = accuracy(output[data.test_mask], data.ground_truth[data.test_mask])

    if epoch % 20 == 0: 
        print('Epoch: {:04d}'.format(epoch+1),
        'loss_train: {:.4f}'.format(loss_train.item()),
        'acc_train: {:.4f}'.format(acc_train.item()),
        'loss_val: {:.4f}'.format(loss_val.item()),
        'acc_val: {:.4f}'.format(acc_val.item()),
        'acc_test: {:.4f}'.format(acc_test.item()))
    return loss_train, loss_val, acc_val, acc_test


def test(args, data, model):
    model.eval()
    with torch.no_grad():
        output = model(args, data)
        # loss_test = F.nll_loss(output[data.test_mask], data.y[data.test_mask])
        loss_test = F.cross_entropy(output[data.test_mask], data.ground_truth[data.test_mask], reduction='mean')
        acc_test = accuracy(output[data.test_mask], data.ground_truth[data.test_mask])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
    return loss_test.item(), acc_test.item()


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)