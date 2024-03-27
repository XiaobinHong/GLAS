import torch 
import torch.nn as nn
import torch.nn.functional as F


from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, APPNP, MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import scipy.sparse
import numpy as np
from .AsrNorm import ASRNormBN

class GPRGNN(nn.Module):
	def __init__(self, nfeat, nhid, Init='PPR', dropout=0.5, device=None, with_bn=True, 
				K=10, alpha=0.1, Gamma=None, ppnp='GPR_prop', args=None):
		super(GPRGNN, self).__init__()

		self.lin1 = nn.Linear(nfeat, nhid)
		if with_bn == 1:
			self.bn1 = nn.BatchNorm1d(nhid)
		elif with_bn == 2:
			self.bn1 = ASRNormBN(nhid)
		self.lin2 = nn.Linear(nhid, nhid)
		self.args = args

		if ppnp == 'PPNP':
			self.prop1 = APPNP(K, alpha)
		elif ppnp == 'GPR_prop':
			self.prop1 = GPR_prop(K, alpha, Init, Gamma)

		self.Init = Init
		self.dropout = dropout
		self.name = "2 layers GPRGNN"
		self.device = device
		self.with_bn = with_bn

	def initialize(self):
		self.lin1.reset_parameters()
		if self.with_bn > 0:
			self.bn1.reset_parameters()
		self.lin2.reset_parameters()
		self.prop1.reset_parameters()

	def get_embed(self, x, edge_index, edge_weight=None):
		self.eval()
		x = F.dropout(x, p=self.dropout, training=self.training)
		x = F.relu(self.lin1(x))

		if edge_weight is not None:
			adj = SparseTensor.from_edge_index(edge_index, edge_weight, sparse_sizes=2*x.shape[:1]).t()
			if self.dropout == 0.0:
				x = self.prop1(x, adj)
			else:
				x = F.dropout(x, p=self.dropout, training = self.training)
				x = self.prop1(x, adj)
		else:
			if self.dropout == 0.0:
				x = self.prop1(x, edge_index, edge_weight)
			else:
				x = F.dropout(x, p=self.dropout, training=self.training)
				x = self.prop1(x, edge_index, edge_weight)
		return x

	def forward(self, x, edge_index, edge_weight=None):
		x = F.dropout(x, p=self.dropout, training=self.training)
		x = F.relu(self.lin1(x))
		if self.with_bn > 0:
			x = self.bn1(x)
		x = F.dropout(x, p=self.dropout, training=self.training)
		x = self.lin2(x)

		if edge_weight is not None:
			adj = SparseTensor.from_edge_index(edge_index, edge_weight, sparse_sizes=2*x.shape[:1]).t()
			if self.dropout == 0.0:
				x = self.prop1(x, adj)
			else:
				x = F.dropout(x, p=self.dropout, training = self.training)
				x = self.prop1(x, adj)
		else:
			if self.dropout == 0.0:
				x = self.prop1(x, edge_index, edge_weight)
			else:
				x = F.dropout(x, p=self.dropout, training = self.training)
				x = self.prop1(x, edge_index, edge_weight)
		return x


class GPR_prop(MessagePassing):
    '''
    GPRGNN, from original repo https://github.com/jianhao2016/GPRGNN
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like
            TEMP = 0.0*np.ones(K+1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = nn.Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, edge_index, edge_weight=None):
        if isinstance(edge_index, torch.Tensor):
            edge_index, norm = gcn_norm(
                edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
            norm = None

        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)