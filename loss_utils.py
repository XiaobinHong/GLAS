import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import numpy as np

import pdb

def class_counter(labels, num_class):
	# pdb.set_trace()
	labels = labels.view(-1).cpu().numpy()
	class_weights = np.zeros(num_class)
	label_count = Counter(labels)
	for idx, (label, count) in enumerate(sorted(label_count.items(), key=lambda x: -x[1])):
		class_weights[label] = count
	return class_weights

class CE(nn.Module):
	def __init__(self, labels, num):
		super(CE, self).__init__()
		self.num = num
		self.num_cls_list = class_counter(labels, num)
		self.weight_list = None

	def forward(self, pred, label, **kwargs):
		loss = F.cross_entropy(pred, label.view(-1))

		return loss

class BSCE(CE):
	def __init__(self, labels, num):
		super(BSCE, self).__init__(labels, num)
		self.bsce_weight = torch.FloatTensor(self.num_cls_list).cuda()

	def forward(self, pred, label, **kwargs):
		logits = pred + self.bsce_weight.unsqueeze(0).expand(pred.shape[0], -1).log()
		loss = F.cross_entropy(logits, label.view(-1))

		return loss