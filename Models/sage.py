import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_sparse import SparseTensor, matmul
# from torch_geometric.nn import SAGEConv, GATConv, APPNP, MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import scipy.sparse
import numpy as np

from .AsrNorm import ASRNormBN

class SAGE(nn.Module):
	def __init__(self, nfeat, nhid, nlayers=2, dropout=0.5, device=None, with_bn=True, args=None):
		super(SAGE, self).__init__()

		self.args = args
		self.layers = nn.ModuleList()
		self.layers.append(SAGEConv(nfeat, nhid))
		if with_bn > 0:
			self.bns = nn.ModuleList()
			if with_bn == 1:
				self.bns.append(nn.BatchNorm1d(nhid))
			elif with_bn == 2:
				self.bns.append(ASRNormBN(nhid))

		for _ in range(nlayers-2):
			self.layers.append(SAGEConv(nhid, nhid))
			if with_bn > 0:
				self.bns.append(nn.BatchNorm1d(nhid))
		self.layers.append(SAGEConv(nhid, nhid))

		self.with_bn = with_bn
		self.device = device
		self.dropout = dropout
		self.name = f"{nlayers}-layers SAGE"

	def initialize(self):
		for layer in self.layers:
			layer.reset_parameters()
		for bn in self.bns:
			bn.reset_parameters()


	def forward(self, x, edge_index, edge_weight=None):
		if edge_weight is not None:
			adj = SparseTensor.from_edge_index(edge_index, edge_weight, sparse_sizes=2*x.shape[:1]).t()

		for i, layer in enumerate(self.layers[:-1]):
			if edge_weight is not None:
				x = layer(x, adj)
			else:
				x = layer(x, edge_index, edge_weight)
			if self.with_bn > 0:
				x = self.bns[i](x)
			x = F.dropout(x, p=self.dropout, training=self.training)
			x = F.relu(x)
		if edge_weight is not None:
			x = self.layers[-1](x, adj)
		else:
			x = self.layers[-1](x, edge_index, edge_weight)
		return x


from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, Size

from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing


class SAGEConv(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 bias: bool = True, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'mean')
        super(SAGEConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if 0:
            if isinstance(x, Tensor):
                x: OptPairTensor = (x, x)
            # propagate_type: (x: OptPairTensor)
            out = self.propagate(edge_index, x=x, size=size)
            out = self.lin_l(out)
        else: # for  fb100 dataset
            if isinstance(x, Tensor):
                x: OptPairTensor = (x, x)
            out = self.lin_l(x[0])
            # propagate_type: (x: OptPairTensor)
            out = self.propagate(edge_index, x=(out, out), size=size)

        x_r = x[1]
        if x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        # Deleted the following line to make propagation differentiable
        # adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)