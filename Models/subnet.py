import torch 
import torch.nn as nn
import torch.nn.functional as F

class projector(nn.Module):
	def __init__(self, nhid, nlayer1=1, dropout=0.5):
		super(projector, self).__init__()
		self.layers = nn.ModuleList()
		for i in range(nlayer1):
			self.layers.append(nn.Linear(nhid, nhid, bias=True))
		self.dropout = dropout
		self.nlayers = nlayer1

	def forward(self, x):
		if self.nlayers == 1:
			x = self.layers[0](x)
		else:
			for i in range(self.nlayers-1):
				x = self.layers[i](x)
				x = F.dropout(x, p=self.dropout, training=self.training)
				x = F.relu(x)
			x = self.layers[-1](x)
		return x

	def initialize(self):
		for m in self.layers:
			m.reset_parameters()


class differentiator(nn.Module):
	def __init__(self, nhid, nlayer2=1, dropout=0.5):
		super(differentiator, self).__init__()
		self.layers = nn.ModuleList()
		for i in range(nlayer2):
			self.layers.append(nn.Linear(nhid, nhid, bias=True))
		self.dropout = dropout
		self.nlayers = nlayer2

	def forward(self, x):
		if self.nlayers == 1:
			x = self.layers[0](x)
		else:
			for i in range(self.nlayers-1):
				x = self.layers[i](x)
				x = F.dropout(x, p=self.dropout, training=self.training)
				x = F.relu(x)
			x = self.layers[-1](x)
		return x

	def initialize(self):
		for m in self.layers:
			m.reset_parameters()