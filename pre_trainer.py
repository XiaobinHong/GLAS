import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from copy import deepcopy
from deeprobust.graph import utils
from torch_geometric.utils import dropout_adj

from loss_utils import *
from Models.BaseModel import DCGNet

from utils import get_gpu_memory_map

import pdb

class PreTrainer(nn.Module):
	def __init__(self, model, device, args):
		super(PreTrainer, self).__init__()
		self.args = args
		self.model = model
		self.device = device

		self.MSELoss = nn.MSELoss()

		from utils import eval_acc, eval_f1, eval_rocauc
		if args.dataset == 'twitch-e':
			self.eval_func = eval_rocauc
		elif args.dataset == 'elliptic':
			self.eval_func = eval_f1
		elif args.dataset in ['cora', 'amazon-photo', 'ogb-arxiv', 'fb100']:
			self.eval_func = eval_acc
		else:
			raise NotImplementedError

		self.model.eval_func = self.eval_func

	def fit_inductive(self, data, train_iters=500, initialize=True, verbose=False, patience=500, **kwargs):
		if initialize:
			self.model.gnn_encoder.initialize()
			self.model.projector.initialize()
			self.model.differentiator.initialize()
			self.model.classifier.reset_parameters()

		self.train_data = data[0]
		self.val_data = data[1]
		self.test_data = data[2]

		self.train_with_early_stopping(train_iters, patience, verbose)


	def fit_with_val(self, pyg_data, train_iters=500, initialize=True, patience=500, verbose=False, **kwargs):
		if initialize:
			self.model.gnn_encoder.initialize()
			self.model.projector.initialize()
			self.model.differentiator.initialize()
			self.model.classifier.reset_parameters()

		self.data = pyg_data.to(self.device)
		self.data.train_mask = self.data.train_mask + self.data.val1_mask
		self.data.val_mask = self.data.val2_mask

		self.train_with_early_stopping(train_iters, patience, verbose)

	def train_with_early_stopping(self, train_iters, patience, verbose):
		if verbose:
			print(f"==== backbone: {self.model.gnn_encoder.name} ====")
		# pdb.set_trace()

		mem_st = get_gpu_memory_map()

		optimizer = optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

		train_data, val_data = self.train_data, self.val_data
		early_stopping = patience
		best_acc_val = float('-inf')

		if type(train_data) is not list:
			x, y = train_data.graph['node_feat'].to(self.device), train_data.label.to(self.device)
			edge_index = train_data.graph['edge_index'].to(self.device)

			x_val, y_val = val_data.graph['node_feat'].to(self.device), val_data.label.to(self.device)
			edge_index_val = val_data.graph['edge_index'].to(self.device)

			if self.args.dataset == 'elliptic':
				self.CELoss = CE(y[train_data.mask], max(y[train_data.mask]).item()+1)
				self.BSCELoss = BSCE(y[train_data.mask], max(y[train_data.mask]).item()+1)
			else:
				self.CELoss = CE(y, max(y).item()+1)
				self.BSCELoss = BSCE(y, max(y).item()+1)

		for epoch in range(train_iters):
			# pdb.set_trace()
			self.model.train()
			optimizer.zero_grad()
			if type(train_data) is not list:
				if hasattr(self, 'dropedge') and self.dropedge != 0:
					edge_index, _ = dropout_adj(edge_index, p=self.dropedge)
				pred1, pred2, z = self.model.forward(x, edge_index)

				if self.args.dataset == 'elliptic':
					loss1 = self.BSCELoss(pred1[train_data.mask], y[train_data.mask])
					loss2 = self.CELoss(pred2[train_data.mask], y[train_data.mask])
				else:
					loss1 = self.BSCELoss(pred1, y)
					loss2 = self.CELoss(pred2, y)
				loss3 = self.MSELoss(z, torch.zeros_like(z))
				loss_train = loss1 + loss2 + self.args.lambda1 * loss3


			else:
				loss_train = 0 
				for graph_id, dat in enumerate(train_data):
					x, y = dat.graph['node_feat'].to(self.device), dat.label.to(self.device)
					edge_index = dat.graph['edge_index'].to(self.device)
					if hasattr(self, 'dropedge') and self.dropedge != 0:
						edge_index, _ =dropout_adj(edge_index, p=self.dropedge)

					pred1, pred2, z = self.model.forward(x, edge_index)
					if self.args.dataset == 'elliptic':
						# pdb.set_trace()
						self.CELoss = CE(y[dat.mask], max(y[dat.mask]).item()+1)
						self.BSCELoss = BSCE(y[dat.mask], max(y[dat.mask]).item()+1)
						loss1 = self.BSCELoss(pred1[dat.mask], y[dat.mask])
						loss2 = self.CELoss(pred2[dat.mask], y[dat.mask])
					else:
						self.CELoss = CE(y, max(y).item()+1)
						self.BSCELoss = BSCE(y, max(y).item()+1)
						loss1 = self.BSCELoss(pred1, y)
						loss2 = self.CELoss(pred2, y)
					loss3 = self.MSELoss(z, torch.zeros_like(z))

					tmp_loss = loss1 + loss2 + self.args.lambda1 * loss3
					loss_train = loss_train + tmp_loss

				loss_train = loss_train / len(train_data)
			loss_train.backward()
			optimizer.step()

			if verbose and epoch % 10 == 0:
				print(f'Epoch: {epoch:03d}, Training Loss: {loss_train.item():.6f}')

			self.model.eval()
			eval_func = self.eval_func
			if self.args.dataset in ['ogb-arxiv']:
				# pdb.set_trace()
				pred, _, _ = self.model.forward(x_val, edge_index_val)
				acc_val = eval_func(y_val[val_data.test_mask], pred[val_data.test_mask])
			elif self.args.dataset in ['cora', 'amazon-photo', 'twitch-e']:
				pred, _, _ = self.model.forward(x_val, edge_index_val)
				acc_val = eval_func(y_val, pred)
			elif self.args.dataset in ['fb100']:
				y_val, out_val = [], []
				for i, dataset in enumerate(val_data):
					x_val = dataset.graph['node_feat'].to(self.device)
					edge_index_val = dataset.graph['edge_index'].to(self.device)
					out, _, _ = self.model.forward(x_val, edge_index_val)
					y_val.append(dataset.label.to(self.device))
					out_val.append(out)
				acc_val = eval_func(torch.cat(y_val, dim=0), torch.cat(out_val, dim=0))
			elif self.args.dataset in ['elliptic']:
				# pdb.set_trace()
				y_val, out_val = [], []
				for i, dataset in enumerate(val_data):
					x_val = dataset.graph['node_feat'].to(self.device)
					edge_index_val = dataset.graph['edge_index'].to(self.device)
					out, _, _ = self.model.forward(x_val, edge_index_val)
					y_val.append(dataset.label[dataset.mask].to(self.device))
					out_val.append(out[dataset.mask])
				acc_val = eval_func(torch.cat(y_val, dim=0), torch.cat(out_val, dim=0))
			else:
				raise NotImplementedError

			if best_acc_val < acc_val:
				best_acc_val = acc_val
				weights = deepcopy(self.model.state_dict())
				patience = early_stopping
			else:
				patience -= 1
			if epoch > early_stopping and patience <= 0:
				break
		if verbose:
			print(f"==== early stopping at {epoch} epoch, acc_val = {best_acc_val:.6f}")
		gpu_mem = get_gpu_memory_map()
		print(f'Mem used: {int(gpu_mem[self.args.gpu_id])-int(mem_st[self.args.gpu_id])}MB')
		self.model.load_state_dict(weights)


	def test(self):
		self.model.eval()
		test_mask = self.data.test_mask
		labels = self.data.y 
		pred1, pred2, z = self.model.forward(self.data.x, self.data.edge_index)
		CELoss = CE(labels[test_mask], max(labels[test_mask]).item()+1)
		BSCELoss = BSCE(labels[test_mask], max(labels[test_mask]).item()+1)
		loss1 = BSCELoss(pred1, labels)
		loss2 = CELoss(pred2, labels)
		loss3 = self.MSELoss(z, torch.zeros_like(z))
		loss_test = loss1 + loss2 + self.args.lambda1 * loss3
		acc_test = utils.accuracy(pred1[test_mask], labels[test_mask])
		print(f"Test Results. Loss = {loss_test.item():.6f}, Accuracy = {acc_test.item():.6f}")
		return acc_test.item()
