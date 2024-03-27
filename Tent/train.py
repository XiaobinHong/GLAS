import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import argparse
import numpy as np
import random 
import time
import sys 
import pdb
from copy import deepcopy

from models import GCN
from utils import *
from tent import *

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
parser.add_argument('--with_bn', action='store_true', help='if with batch norm')

parser.add_argument('--gnn', type=str, default='GCN', help='gnn type')
parser.add_argument('--gat_heads', type=int, default=8, help='the number of GAT heads')

# parser.add_argument('--ASRNorm', action='store_true', help='if use ASRNorm')

parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--ood', type=int, default=1)
parser.add_argument('--tune', type=int, default=0)

args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)
device = 'cuda' if args.gpu_id >= 0 else "cpu"
# device = "cpu"
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print('===========')
# reset_args(args)
print(args)

from utils import eval_acc, eval_f1, eval_rocauc
if args.dataset == 'twitch-e':
	eval_func = eval_rocauc
elif args.dataset == 'elliptic':
	eval_func = eval_f1
elif args.dataset in ['cora', 'amazon-photo', 'ogb-arxiv', 'fb100']:
	eval_func = eval_acc
else:
	raise NotImplementedError

##============= data loader ====================##
if args.ood:
	path = '../GraphOOD-EERM/'
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

pdb.set_trace()

def get_model(args, data, verbose=True):
	if type(data[0]) is not list:
		feat, labels = data[0].graph['node_feat'], data[0].label 
		edge_index = data[0].graph['edge_index']

	else:
		feat, labels = data[0][0].graph['node_feat'], data[0][0].label 
		edge_index = data[0][0].graph['edge_index']

	save_mem = False

	model = GCN(nfeat=feat.shape[1], nhid=args.hidden, nclass=max(labels).item()+1, nlayers=args.nlayers, dropout=args.dropout, save_mem=save_mem,
					with_bn=args.with_bn, with_bias=True, device=device, args=args).to(device)
	if verbose:
		print(model)
	return model


def sup_loss(y, pred):
	if args.dataset in ('twitch-e', 'fb100', 'elliptic'):
		if y.shape[1] == 1:
			true_label = F.one_hot(y, y.max()+1).squeeze(1)
		else:
			true_label = y 
		criterion = nn.BCEWithLogitsLoss()
		loss = criterion(pred, true_label.squeeze(1).to(torch.float))
	else:
		out = F.log_softmax(pred, dim=1)
		target = y.squeeze(1)
		criterion = nn.NLLLoss()
		loss = criterion(out, target)
	return loss


def train(args, model, data, verbose=True):
	# pdb.set_trace()
	if verbose:
		print(f"======backbone: {model.name}======")

	optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	train_data = data[0]
	val_data = data[1]
	test_data = data[2]
	early_stopping = args.patience
	best_acc_val = float('-inf')

	if type(train_data) is not list:
		x, y = train_data.graph['node_feat'].to(device), train_data.label.to(device)
		edge_index = train_data.graph['edge_index'].to(device)

		x_val, y_val = val_data.graph['node_feat'].to(device), val_data.label.to(device)
		edge_index_val = val_data.graph['edge_index'].to(device)

	for epoch in range(args.epochs):
		model.train()
		optimizer.zero_grad()
		if type(train_data) is not list:
			pred = model(x, edge_index)
			loss_train = sup_loss(y, pred)
		else:
			loss_train = 0.0
			for graph_id, dat in enumerate(train_data):
				# pdb.set_trace()
				x, y = dat.graph['node_feat'].to(device), dat.label.to(device)
				edge_index = dat.graph['edge_index'].to(device)
				pred = model(x, edge_index)
				if args.dataset == 'elliptic':
					loss_train = loss_train + sup_loss(y[dat.mask], pred[dat.mask])
				else:
					loss_train = loss_train + sup_loss(y, pred)
				# del(x, y)
			loss_train = loss_train / len(train_data)
		loss_train.backward()
		optimizer.step()

		if verbose and epoch % 10 == 0:
			print(f'Epoch: {epoch:03d}, Training Loss: {loss_train.item():.6f}')

		model.eval()
		if args.dataset in ['ogb-arxiv']:
			pred = model(x_val, edge_index_val)
			acc_val = eval_func(y_val[val_data.test_mask], pred[val_data.test_mask])
		elif args.dataset in ['cora', 'amazon-photo', 'twitch-e']:
			pred = model(x_val, edge_index_val)
			acc_val = eval_func(y_val, pred)
		elif args.dataset in ['fb100']:
			y_val, out_val = [], []
			for i, dat in enumerate(val_data):
				x_val = dat.graph['node_feat'].to(device)
				edge_index_val = dat.graph['edge_index'].to(device)
				pred = model(x_val, edge_index_val)
				y_val.append(dat.label.to(device))
				out_val.append(pred)
			acc_val = eval_func(torch.cat(y_val), torch.cat(out_val, dim=0))
		elif args.dataset in ['elliptic']:
			y_val, out_val = [], []
			for i, dat in enumerate(val_data):
				x_val = dat.graph['node_feat'].to(device)
				edge_index_val = dat.graph['edge_index'].to(device)
				pred = model(x_val, edge_index_val)
				y_val.append(dat.label[dat.mask].to(device))
				out_val.append(pred[dat.mask])
			acc_val = eval_func(torch.cat(y_val, dim=0), torch.cat(out_val, dim=0))
		else:
			raise NotImplementedError

		if best_acc_val < acc_val:
			best_acc_val = acc_val
			weights = deepcopy(model.state_dict())
			patience = early_stopping
		else:
			patience -= 1 
		if epoch > early_stopping and patience <= 0:
			break
	if verbose:
		print(f"======early stopping at {epoch} epoch, acc_val = {best_acc_val:.6f}")
	model.load_state_dict(weights)
	filename = f'checkpoints/{args.dataset}_{args.gnn}_s{args.seed}.pt'
	torch.save(model.state_dict(), filename)
	return model


def test_time_adapt(model, data):
	pdb.set_trace()
	model = configure_model(model)
	params, param_names = collect_params(model)
	optimizer = optim.Adam(params, lr=1e-3, weight_decay=1e-5)
	tented_model = TENT(model, optimizer)
	for ii, test_data in enumerate(data[2]):
		x, edge_index = test_data.graph['node_feat'], test_data.graph['edge_index']
		x, edge_index = x.to(device), edge_index.to(device)
		output = tented_model(x, edge_index)
	return tented_model


def evaluate(model, data):
	# pdb.set_trace()
	model.eval()
	accs = []
	y_te, out_te = [], []
	y_te_all, out_te_all = [], []
	for ii, test_data in enumerate(data[2]):
		x, edge_index = test_data.graph['node_feat'], test_data.graph['edge_index']
		x, edge_index = x.to(device), edge_index.to(device)
		output = model(x, edge_index)
		labels = test_data.label.to(device) #.squeeze()
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
				accs += [float(f'{acc_te:.2f}')]
				y_te, out_te = [], []
		else:
			raise NotImplementedError
	print('Test accs:', accs)
	acc_te = eval_func(torch.cat(y_te_all, dim=0), torch.cat(out_te_all, dim=0))
	print(f'flatten test: {acc_te}')
	res = f"test accuracy: {accs}  " + f"flatten test: {acc_te}"
	return res

filename = f'checkpoints/{args.dataset}_{args.gnn}_s{args.seed}.pt'
model = get_model(args, data)
if args.debug and osp.exists(filename):
	model.load_state_dict(torch.load(filename, map_location=device))
else:
	model = train(args, model, data, verbose=True)
res = evaluate(model, data)

tented_model = test_time_adapt(model, data)
res = evaluate(tented_model, data)