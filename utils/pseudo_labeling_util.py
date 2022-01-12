import random
import time
import copy
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from aug import permute_edges, mask_nodes


def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Net') or m.__class__.__name__.startswith('Encoder'):
            m.train()


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def pseudo_labeling(args, data, model, itr):
    print("{} Iteration".format(itr+1))

    pseudo_idx = []
    pseudo_target = []
    pseudo_maxstd = []
    gt_target = []
    idx_list = []
    gt_list = []
    target_list = []

    model.eval()
    with torch.no_grad():
        data = data.to(args.device)
        out_prob = []
        for _ in range(args.aug_sample_times):
            data_aug = copy.deepcopy(data)
            data_aug = mask_nodes(data_aug, args.mask_aug_ratio)
            data_aug = permute_edges(data_aug, args.edge_aug_ratio)
            outputs = model(args, data_aug)[data.unlabel]
            out_prob.append(F.softmax(outputs, dim=1))  # for selecting positive pseudo-labels
        out_prob = torch.stack(out_prob)
        out_std = torch.std(out_prob, dim=0)
        out_prob = torch.mean(out_prob, dim=0)
        max_value, max_idx = torch.max(out_prob, dim=1)
        max_std = out_std.gather(1, max_idx.view(-1, 1))

        acc_unlabel = accuracy(out_prob, data.y[data.unlabel])
        print("Unlable set results:", "accuracy= {:.4f}".format(acc_unlabel.item()))

        idx_list.extend(data.indexs[data.unlabel].cpu().numpy().tolist())
        gt_list.extend(data.ground_truth[data.unlabel].cpu().numpy().tolist())
        target_list.extend(max_idx.cpu().numpy().tolist())

        itr += 1
        # sets mu: percentiles threshold
        uncertainty_percentiles = 0 + args.uncertainty_percentiles * itr
        if uncertainty_percentiles < 0:
            uncertainty_percentiles = 0

        # DONE: insert curriculum labeling
        # sets mu: percentiles threshold
        percentiles_holder = 100 - args.percentiles_holder * itr
        if percentiles_holder < 0:
            percentiles_holder = 0
        curriculum_threshold = np.percentile(np.asarray(max_value.cpu().detach().numpy()),
                                  percentiles_holder)  # From smallest to largest
        print('percentiles_holder:', percentiles_holder)
        print('curriculum_threshold:', curriculum_threshold)
        print('max_score:', max(max_value))
        print('shape of max_value:', max_value.shape)

        # selecting positive pseudo-labels
        if args.no_curriculum:
            selected_idx = (max_value >= args.tau_p)
            print('#max_value>=args.tau_p:', int((max_value >= args.tau_p).sum().item()))
        elif args.no_uncertainty:
            selected_idx = (max_value >= max(args.tau_p, curriculum_threshold))
            print('#max_value>=max(args.tau_p, threshold):',
                  int((max_value >= max(args.tau_p, curriculum_threshold)).sum().item()))
        else:
            selected_idx = (max_value >= max(args.tau_p, curriculum_threshold)) * \
                           (max_std.squeeze(1) < uncertainty_percentiles)
            print('#max_value>=args.tau_p:', int((max_value >= args.tau_p).sum().item()))
            print('#max_value>=max(args.tau_p, threshold):',
                  int((max_value >= max(args.tau_p, curriculum_threshold)).sum().item()))
            print('#max_std<uncertainty_percentiles:',
                  int((max_std.squeeze(1) < uncertainty_percentiles).sum().item()))
        
        if sum(selected_idx).item() == 0:
            return None

        pseudo_maxstd.extend(max_std.squeeze(1)[selected_idx].cpu().numpy().tolist())
        pseudo_target.extend(max_idx[selected_idx].cpu().numpy().tolist())
        pseudo_idx.extend(data.indexs[data.unlabel][selected_idx].cpu().numpy().tolist())
        gt_target.extend(data.ground_truth[data.unlabel][selected_idx].cpu().numpy().tolist())

    pseudo_maxstd = np.array(pseudo_maxstd)
    pseudo_target = np.array(pseudo_target)
    pseudo_idx = np.array(pseudo_idx)
    gt_target = np.array(gt_target)


    pseudo_labeling_acc = (pseudo_target == gt_target) * 1
    pseudo_labeling_acc = (sum(pseudo_labeling_acc) / len(pseudo_labeling_acc)) * 100
    print(f'Uncertainty Pseudo-Labeling Accuracy: {pseudo_labeling_acc}, Total Selected: {len(pseudo_idx)}')

    arg_idx = pseudo_idx.tolist()
    arg_target = pseudo_target.tolist()
    arg_idx = arg_idx[:args.select_unlabel]
    arg_target = arg_target[:args.select_unlabel]
    pseudo_label_dict = {'pseudo_idx': arg_idx, 'pseudo_target': arg_target}
    print(f'Total Selected: {len(arg_idx)}')

    if len(arg_idx) == 0:
        return None

    return pseudo_label_dict

