import torch
import torch.nn as nn 
import torch.nn.functional as F 
import pdb

from torch_geometric.nn import GINConv
from torch_sparse import coalesce, SparseTensor

from .AsrNorm import ASRNormBN

class GIN(nn.Module):
	def __init__(self, nfeat, nhid, nlayers=2, dropout=0.5, with_bn=True, with_bias=True, save_mem=True, device=None, args=None):
		super(GIN, self).__init__()
		self.gin_layers = nn.ModuleList()
		if with_bn > 0:
			self.bns = nn.ModuleList()

		self.nlayers = nlayers
		if nlayers == 1:
			self.gin_layers.append(GINConv(nn.Linear(nfeat, nclass, bias=with_bias)))
		else:
			self.gin_layers.append(GINConv(nn.Linear(nfeat, nhid, bias=with_bias)))
			if with_bn == 1:
				self.bns.append(nn.BatchNorm1d(nhid))
			elif with_bn == 2:
				self.bns.append(ASRNormBN(nhid))

			for i in range(nlayers-2):
				self.gin_layers.append(GINConv(nn.Linear(nhid, nhid, bias=with_bias)))

				if with_bn > 0:
					self.bns.append(nn.BatchNorm1d(nhid))

		self.gin_layers.append(GINConv(nn.Linear(nhid, nhid, bias=with_bias)))

		self.dropout = dropout
		self.with_bn = with_bn
		self.args = args
		self.name = f"{nlayers}-layers GIN"

	def forward(self, x, edge_index, edge_weight=None):
		# pdb.set_trace()
		x, edge_index, edge_weight = self._ensure_contiguousness(x, edge_index, edge_weight)
		if edge_weight is not None:
			adj = SparseTensor.from_edge_index(edge_index, edge_weight, sparse_size=2*x.shape[:1]).t()

		for ii, layer in enumerate(self.gin_layers):
			if edge_weight is not None:
				x = layer(x, adj)
			else:
				x = layer(x, edge_index)

			if ii != len(self.gin_layers)-1:
				if self.with_bn > 0:
					x = self.bns[ii](x)
				x = F.relu(x)
				x = F.dropout(x, p=self.dropout, training=self.training)

			x = self.gin_layers[-1](x, edge_index)
			return x

	def initialize(self):
		for m in self.gin_layers:
			m.reset_parameters()
		if self.with_bn > 0:
			for bn in self.bns:
				bn.reset_parameters()

	def _ensure_contiguousness(self,
							   x,
							   edge_idx,
							   edge_weight):
		if not x.is_sparse:
			x = x.contiguous()
		if hasattr(edge_idx, 'contiguous'):
			edge_idx = edge_idx.contiguous()
		if edge_weight is not None:
			edge_weight = edge_weight.contiguous()
		return x, edge_idx, edge_weight