import argparse
import numpy as np
import torch
import random
import time
import sys 
import pdb 
import os.path as osp

import torch.optim as optim

from Models.BaseModel import DCGNet
from utils import *
from pre_trainer import PreTrainer

from data_states import label_shift
from test_adapter import *
from loss_utils import class_counter

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--patience', type=int, default=200)
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--seed', type=int, default=42, help='Random seed.')

parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--with_bn', type=int, default=1, help='0 for without bn, 1 for 1d bn, 2 for ASRNorm')

parser.add_argument('--gnn', type=str, default='GCN', help='gnn type')
parser.add_argument('--noise_alpha', type=float, default=5.0, help='the noise scale')
parser.add_argument('--lambda1', type=float, default=0.1, help='the weight of MSE Loss')
parser.add_argument('--proj_nlayer', type=int, default=1, help='the number of projector layers')
parser.add_argument('--diff_nlayer', type=int, default=1, help='the number of differentiator layers')
parser.add_argument('--gat_heads', type=int, default=8, help='the number of GAT heads')

parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--ood', type=int, default=1)
parser.add_argument('--tune', type=int, default=0)
parser.add_argument('--run', type=int, default=1)
parser.add_argument('--test_val', type=int, default=0, help='set to 1 to evaluate performance on validation data')

args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)
device = 'cuda' if args.gpu_id >= 0 else "cpu"
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print('===========')
# reset_args(args)
print(args)

##============= data loader ====================##
if args.ood:
    path = 'GraphOOD-EERM/'
    if args.dataset == 'elliptic':
        path = path + 'temp_elliptic'
        sys.path.append(path)
        from main_as_utils import datasets_tr, datasets_val, datasets_te
        data = [datasets_tr, datasets_val, datasets_te]
    elif args.dataset == 'fb100':
        path = path + 'multigraph'
        sys.path.append(path)
        from main_as_utils_fb import datasets_tr, datasets_val, datasets_te
        data = [datasets_tr, datasets_val, datasets_te]
    elif args.dataset == 'amazon-photo':
        path = path + 'synthetic'
        sys.path.append(path)
        from main_as_utils_photo import dataset_tr, dataset_val, datasets_te
        data = [dataset_tr, dataset_val, datasets_te]
    else:
        if args.dataset == 'cora':
            path = path + 'synthetic'
        elif args.dataset == 'ogb-arxiv':
            path = path + 'temp_arxiv'
        elif args.dataset == 'twitch-e':
            path = path + 'multigraph'
        else:
            raise NotImplementedError
        sys.path.append(path)
        from main_as_utils import dataset_tr, dataset_val, datasets_te
        data = [dataset_tr, dataset_val, datasets_te]
else:
    data = get_dataset(args.dataset, args.normalize_features)

from utils import eval_acc, eval_f1, eval_rocauc
if args.dataset == 'twitch-e':
    eval_func = eval_rocauc
elif args.dataset == 'elliptic':
    eval_func = eval_f1
elif args.dataset in ['cora', 'amazon-photo', 'ogb-arxiv', 'fb100']:
    eval_func = eval_acc
else:
    raise NotImplementedError

# pdb.set_trace()

def label_shift_degree(data, args):
    # pdb.set_trace()
    train_data, test_data = data[0], data[2]
    if type(train_data) is not list:
        train_y = train_data.label.view(-1)
    else:
        train_y = []
        if args.dataset == 'elliptic':
            for dat in train_data:
                train_y.append(dat.label[dat.mask].view(-1))
            train_y = torch.cat(train_y, dim=0)
        elif args.dataset == 'fb100':
            for dat in train_data:
                train_y.append(dat.label.view(-1))
            train_y = torch.cat(train_y, dim=0)
    # pdb.set_trace()
    num_class = max(train_y)+1
    ps_y = class_counter(train_y, num_class)
    ps_y = ps_y / train_y.shape[0]

    degrees = []
    y_te = []
    for i, dat in enumerate(test_data):
        if args.dataset in ['ogb-arxiv']:
            # pdb.set_trace()
            test_yi = dat.label[dat.test_mask].view(-1)
            pt_yi = class_counter(test_yi, num_class) / test_yi.shape[0]
            t1 = [abs(ps_y[j]-pt_yi[j]) for j in range(ps_y.shape[0])]
            degrees.append(sum(t1))
        elif args.dataset in ['cora', 'amazon-photo', 'twitch-e', 'fb100']:
            test_yi = dat.label
            pt_yi = class_counter(test_yi, num_class) / test_yi.shape[0]
            t1 = [abs(ps_y[j]-pt_yi[j]) for j in range(ps_y.shape[0])]
            degrees.append(sum(t1))
        elif args.dataset in ['elliptic']:
            test_yi = dat.label[dat.mask].view(-1)
            y_te.append(test_yi)
            if i % 4 == 0 or i == len(test_data) - 1:
                # pdb.set_trace()
                y_te = torch.cat(y_te, dim=0)
                pt_yi = class_counter(y_te, num_class) / y_te.shape[0]
                t1 = [abs(ps_y[j]-pt_yi[j]) for j in range(ps_y.shape[0])]
                degrees.append(sum(t1))
                y_te = []
    return degrees

degrees = label_shift_degree(data, args)
print('Label shift degrees:', degrees)
pdb.set_trace()



def get_model(args, data, verbose=True):
    if type(data[0]) is not list:
        feat, labels = data[0].graph['node_feat'], data[0].label 
        edge_index = data[0].graph['edge_index']

    else:
        feat, labels = data[0][0].graph['node_feat'], data[0][0].label 
        edge_index = data[0][0].graph['edge_index']

    save_mem = False
    model = DCGNet(nfeat=feat.shape[1], nhid=args.hidden, nclass=max(labels)+1, nlayers=args.nlayers, dropout=args.dropout, 
                    save_mem=save_mem, gnn=args.gnn, with_bn=args.with_bn, with_bias=True, device=device, args=args).to(device)

    if verbose:
        print(model)
    return model



def pretrain(args, model, data, verbose=True): ###########
    # pdb.set_trace()
    pre_trainer = PreTrainer(model, device, args)
    train_iters = 1000 if args.dataset == 'ogb-arxiv' else args.epochs
    pre_trainer.fit_inductive(data, train_iters=train_iters, patience=args.patience, verbose=verbose)
    accs, res = evaluate(model, data)
    return model, res, accs

def test_time_adapt(model, data, args):
    # pdb.set_trace()
    model = configure_model(model)
    params, param_names = collect_params(model)
    optimizer = optim.Adam(params, lr=1e-4, weight_decay=1e-5)
    
    accs =[]
    y_te, out_te = [], []
    y_te_all, out_te_all = [], []
    # adapted_model = Adaptor(model, optimizer, device)
    for ii, test_data in enumerate(data[2]):
        adapted_model = Adaptor(model, optimizer, device)
        x, edge_index = test_data.graph['node_feat'], test_data.graph['edge_index']
        x, edge_index = x.to(device), edge_index.to(device)
        labels = test_data.label.to(device)
        output = adapted_model(x, edge_index)
        if args.dataset in ['ogb-arxiv']:
            acc_test = eval_func(labels[test_data.test_mask], output[test_data.test_mask])
            accs.append(acc_test)
            y_te_all.append(labels[test_data.test_mask])
            out_te_all.append(output[test_data.test_mask])
        elif args.dataset in ['cora', 'amazon-photo', 'twitch-e', 'fb100']:
            acc_test = eval_func(labels, output)
            accs.append(acc_test)
            y_te_all.append(labels)
            out_te_all.append(output)
        elif args.dataset in ['elliptic']:
            acc_test = eval_func(labels[test_data.mask], output[test_data.mask])
            y_te.append(labels[test_data.mask])
            out_te.append(output[test_data.mask])
            y_te_all.append(labels[test_data.mask])
            out_te_all.append(output[test_data.mask])
            if ii % 4 == 0 or ii == len(data[2])-1:
                acc_te = eval_func(torch.cat(y_te, dim=0), torch.cat(out_te, dim=0))
                accs += [float(f'{acc_te:.4f}')]
                y_te, out_te = [], []
        else:
            raise NotImplementedError
    print('Test accs:', accs)
    acc_te = eval_func(torch.cat(y_te_all, dim=0), torch.cat(out_te_all, dim=0))
    print(f'flatten test: {acc_te}')
    res = f"test accuracy: {accs}  " + f"flatten test: {acc_te}"
    return acc_te, accs


def evaluate(model, data):
    # pdb.set_trace()
    model.eval()
    accs = []
    y_te, out_te = [], []
    y_te_all, out_te_all = [], []
    for ii, test_data in enumerate(data[2]):
        x, edge_index = test_data.graph['node_feat'], test_data.graph['edge_index']
        x, edge_index = x.to(device), edge_index.to(device)
        output = model.predict(x, edge_index)

        labels = test_data.label.to(device) #.squeeze()
        # eval_func = model.eval_func
        if args.dataset in ['ogb-arxiv']:
            acc_test = eval_func(labels[test_data.test_mask], output[test_data.test_mask])
            accs.append(acc_test)
            y_te_all.append(labels[test_data.test_mask])
            out_te_all.append(output[test_data.test_mask])
        elif args.dataset in ['cora', 'amazon-photo', 'twitch-e', 'fb100']:
            acc_test = eval_func(labels, output)
            accs.append(acc_test)
            y_te_all.append(labels)
            out_te_all.append(output)
        elif args.dataset in ['elliptic']:
            acc_test = eval_func(labels[test_data.mask], output[test_data.mask])
            y_te.append(labels[test_data.mask])
            out_te.append(output[test_data.mask])
            y_te_all.append(labels[test_data.mask])
            out_te_all.append(output[test_data.mask])
            if ii % 4 == 0 or ii == len(data[2]) - 1:
                acc_te = eval_func(torch.cat(y_te, dim=0), torch.cat(out_te, dim=0))
                accs += [float(f'{acc_te:.4f}')]
                y_te, out_te = [], []
        else:
            raise NotImplementedError
    print('Test accs:', accs)
    acc_te = eval_func(torch.cat(y_te_all, dim=0), torch.cat(out_te_all, dim=0))
    print(f'flatten test: {acc_te}')
    res = f"test accuracy: {accs}  " + f"flatten test: {acc_te}"
    return accs, acc_te

def train_evaluate(model, data):
    pdb.set_trace()
    model.eval()
    accs = 0.0
    x, edge_index = data[0].graph['node_feat'], data[0].graph['edge_index']
    x, edge_index = x.to(device), edge_index.to(device)
    y = data[0].label.to(device)
    output = model.predict(x, edge_index)
    accs = eval_func(y, output)
    print('Train accs:', accs)



run_accs = []
st_time = time.time()
for _ in range(args.run):
    model = get_model(args, data, verbose=False)
    model, test_acc, accs1 = pretrain(args, model, data)
    pre_weights = deepcopy(model.state_dict())
    adapted_acc, accs2 = test_time_adapt(model, data, args)
    if adapted_acc < test_acc:
        run_accs.append(test_acc.item())
        model.load_state_dict(pre_weights)
        accs = accs1
    else:
        run_accs.append(adapted_acc.item())
        accs = accs2

    torch.save(model.state_dict(), f"checkpoints/{args.dataset}_{args.gnn}_{args.seed}s.pt")

    # del(model)
    # tmp = [str(ii) for ii in accs]
    # with open("boxplot_res.txt", 'a+') as f:
    #     f.write(f"Dataset: {args.dataset}, seed: {args.seed}. \n")
    #     f.write(', '.join(tmp))
    #     f.write('\n\n')

strs = f"Dataset: {args.dataset}, GNN: {args.gnn}, {args.run} runs, mean flatten acc: {np.mean(run_accs):.6f}, var: {np.var(run_accs):.6f}"
# print(f"Dataset: {args.dataset}, {args.run} runs, mean flatten acc: {np.mean(run_accs):.6f}, var: {np.var(run_accs):.6f}")
print(strs)
print(f"Total Consumed Times: {time.time()-st_time}s.")
# with open("results.txt", "a+") as f:
#     f.write(f"num_layers: {args.nlayers}, hidden dim: {args.hidden}, gnn: {args.gnn}\n")
#     f.write(strs)
#     f.write('\n\n') 