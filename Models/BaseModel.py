import torch 
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from deeprobust.graph import utils
from torch_geometric.utils import dropout_adj

from .gcn import GCN 
from .gat import GAT
from .gpr import GPRGNN
from .sage import SAGE
from .gin import GIN
from .subnet import projector, differentiator

import pdb

class DCGNet(nn.Module):
	def __init__(self, nfeat, nhid, nclass, nlayers, dropout=0.5, save_mem=True,
				gnn='GCN', with_bn=True, with_bias=True, device=None, args=None):
		super(DCGNet, self).__init__()
		
		if gnn == 'GCN':
			self.gnn_encoder = GCN(nfeat=nfeat, nhid=nhid, nlayers=nlayers, dropout=dropout, save_mem=save_mem,
					with_bn=with_bn, with_bias=with_bias, device=device, args=args)
		elif gnn == 'GAT':
			self.gnn_encoder = GAT(nfeat=nfeat, nhid=nhid, nlayers=nlayers, heads=args.gat_heads, output_head=1, dropout=0., 
				save_mem=save_mem, with_bn=with_bn, with_bias=with_bias, device=device, args=args)
		elif gnn == 'GPR':
			self.gnn_encoder = GPRGNN( nfeat=nfeat, nhid=nhid, Init='PPR', dropout=dropout, device=device, 
				with_bn=args.with_bn, K=10, alpha=0.1, Gamma=None, ppnp='GPR_prop', args=args)
		elif gnn == 'SAGE':
			self.gnn_encoder = SAGE(nfeat=nfeat, nhid=nhid, nlayers=nlayers, dropout=dropout, 
				device=device, with_bn=args.with_bn, args=args)
		elif gnn == 'GIN':
			self.gnn_encoder = GIN(nfeat=nfeat, nhid=nhid, nlayers=nlayers, dropout=dropout, with_bn=with_bn,
				with_bias=with_bias, save_mem=save_mem, device=device, args=args)
		else:
			raise NotImplementedError

		self.projector = projector(nhid=nhid, nlayer1=args.proj_nlayer)

		self.differentiator = differentiator(nhid=nhid, nlayer2=args.diff_nlayer)

		self.classifier = nn.Linear(nhid, nclass, bias=False)

		self.dropout = dropout
		self.save_mem = save_mem
		self.with_bn = with_bn
		self.device = device
		self.args = args
		self.output = None
		self.eval_func = None

	def forward(self, x, edge_index):
		# pdb.set_trace()
		z1 = self.gnn_encoder(x, edge_index)
		mag_norm = self.args.noise_alpha / torch.sqrt(torch.tensor(z1.size(1)))
		noise_z = torch.zeros_like(z1).uniform_(-mag_norm, mag_norm)
		noise_z = noise_z.to(self.device)
		z2 = z1 +  noise_z

		z1 = self.projector(z1)
		z2 = self.projector(z2)

		# z1 = torch.sigmoid(z1) * z1
		# z2 = torch.sigmoid(z2) * z2

		pred1 = self.classifier(z1)
		pred2 = self.classifier(z2)

		# z = torch.norm(z1-z2, p=2, dim=0)
		z = self.differentiator(z1-z2)
		# return F.log_softmax(pred1, dim=1), F.log_softmax(pred2, dim=1), z
		return pred1, pred2, z

	@torch.enable_grad()
	def get_embed(self, x, edge_index):
		z1 = self.gnn_encoder(x, edge_index)
		mag_norm = self.args.noise_alpha / torch.sqrt(torch.tensor(z1.size(1)))
		noise_z = torch.zeros_like(z1).uniform_(-mag_norm, mag_norm)
		noise_z = noise_z.to(self.device)
		z2 = z1 + noise_z

		z1 = self.projector(z1)
		z2 = self.projector(z2)

		z = self.differentiator(z1-z2)
		return z1, z2, z

	@torch.no_grad()
	def predict(self, x, edge_index):
		z1 = self.gnn_encoder(x, edge_index)
		z1 = self.projector(z1)
		pred1 = self.classifier(z1)
		return pred1

	def initialize(self):
		self.gnn_encoder.initialize()
		self.projector.initialize()
		self.differentiator.initialize()
		self.classifier.reset_parameters()