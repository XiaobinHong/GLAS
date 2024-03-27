import torch.nn as nn
import torch.nn.functional as F
import math 
import torch
import torch.optim as optim

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import GCNConv
from torch_sparse import coalesce, SparseTensor

class GCN(nn.Module):
	def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.5, save_mem=True,
				with_bn=True, with_bias=True, device=None, args=None):
		super(GCN, self).__init__()

		assert device is not None, "Please specify device!"
		self.device = device
		self.args = args

		self.layers = nn.ModuleList()
		if with_bn:
			self.bns = nn.ModuleList()

		if nlayers == 1:
			self.layers.append(GCNConv(nfeat, nclass, bias=with_bias, normalize=not save_mem))
		else:
			self.layers.append(GCNConv(nfeat, nhid, bias=with_bias, normalize=not save_mem))
			if with_bn:
				self.bns.append(nn.BatchNorm1d(nhid))
				# self.bns.append(ASRNormBN(nhid))

			for i in range(nlayers-2):
				self.layers.append(GCNConv(nhid, nhid, bias=with_bias, normalize=not save_mem))
				if with_bn:
					self.bns.append(nn.BatchNorm1d(nhid))
			self.layers.append(GCNConv(nhid, nclass, bias=with_bias, normalize=not save_mem))

		self.dropout = dropout
		self.output = None 
		self.best_model = None 
		self.best_output = None 
		self.with_bn = with_bn 
		self.name = f'{nlayers} layers GCN'


	def forward(self, x, edge_index, edge_weight=None):
		x, edge_index, edge_weight = self._ensure_contiguousness(x, edge_index, edge_weight)
		if edge_weight is not None:
			adj = SparseTensor.from_edge_index(edge_index, edge_weight, sparse_sizes=2*x.shape[:1]).t()

		for ii, layer in enumerate(self.layers):
			if edge_weight is not None:
				x = layer(x, adj)
			else:
				x = layer(x, edge_index)
			if ii != len(self.layers)-1:
				if self.with_bn:
					x = self.bns[ii](x)
				x = F.relu(x)
				x = F.dropout(x, p=self.dropout, training=self.training)
		return x 
		# return F.log_softmax(x, dim=1)

	@torch.no_grad()
	def get_embed(self, x, edge_index, edge_weight=None):
		x, edge_index, edge_weight = self._ensure_contiguousness(x, edge_index, edge_weight)
		if edge_weight is not None:
			adj = SparseTensor.from_edge_index(edge_index, edge_weight, sparse_sizes=2*x.shape[:1]).t()

		for ii, layer in enumerate(self.layers):
			if edge_weight is not None:
				x = layer(x, adj)
			else:
				x = layer(x, edge_index)
			if ii != len(self.layers) - 1:
				if self.with_bn:
					x = self.bns[ii](x)
				x = F.relu(x)
				x = F.dropout(x, p=self.dropout, training=self.training)
		return x

	def _ensure_contiguousness(self, x, edge_idx, edge_weight):
		if not x.is_sparse:
			x = x.contiguous()
		if hasattr(edge_idx, 'contiguous'):
			edge_idx = edge_idx.contiguous()
		if edge_weight is not None:
			edge_weight = edge_weight.contiguous()
		return x, edge_idx, edge_weight

	def initialize(self):
		for m in self.layers:
			m.reset_parameters()
		if self.with_bn:
			for bn in self.bns:
				bn.reset_parameters()