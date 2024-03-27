from copy import deepcopy

import torch  
import torch.nn as nn
import torch.nn.functional as F
import torch.jit 
import time

from Models.subnet import projector
from loss_utils import class_counter
import pdb

class Adaptor(nn.Module):
	def __init__(self, model, optimizer, device, beta=0.1, tau=0.5, steps=1, episodic=False):
		super().__init__()
		self.model = model 
		self.optimizer = optimizer
		self.steps = steps
		assert steps > 0, "Requires >= 1 step(s) to forward and update"
		self.episodic = episodic
		self.MSELoss = nn.MSELoss()
		self.mu_s = deepcopy(self.model.classifier.weight.data)
		self.num_class = self.model.classifier.weight.shape[0]
		self.device = device
		self.prop = (torch.ones((self.num_class,1))*(1/self.num_class)).to(device)
		self.beta = beta  ## momentum update parameter for learning target proportions
		self.tau = tau ## temperature parameter
		self.eps = 1e-6

		self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)

	def pairwise_cosine_dist(self, x, y):
		x = F.normalize(x, p=2, dim=1)
		y = F.normalize(y, p=2, dim=1)
		return 1 - torch.matmul(x, y.T)

	def get_pos_logits(self, sim_mat, prop):
		log_prior = torch.log(prop + self.eps)
		return sim_mat/self.tau + log_prior

	def update_prop(self, prop):
		return (1 - self.beta) * self.prop + self.beta * prop 

	def forward(self, x, edge_index):
		'''
		forward and adapt projector
		'''
		# pdb.set_trace()
		if self.episodic:
			self.reset()
		for _ in range(self.steps):
			outputs = self.forward_and_adapt(x, edge_index)
		return outputs

	# @torch.enable_grad()
	# def forward_and_adapt(self, x, edge_index):
	# 	'''
	# 	main test time adaptation mathod:
	# 	1. target node embedding KNN cluster
	# 	2. the mutual normalized information between target and source prototype
	# 	3. differentiator loss
	# 	'''
	# 	# pdb.set_trace()
	# 	z1, z2, z = self.model.get_embed(x, edge_index)
	# 	z1_norm = F.normalize(z1, p=2, dim=1)
	# 	y_hat_t, mu_t = KMeans(z1_norm, device=self.device, K=self.num_class)

	# 	py_t = class_counter(y_hat_t, self.num_class)
	# 	py_t = torch.FloatTensor(py_t).to(self.device)
	# 	py_t = py_t / py_t.sum()

	# 	mi_loss = self.MSELoss(py_t*mu_t.t(), self.py_s*self.mu_s.t())
	# 	# mi_loss = 
	# 	diff_loss = self.MSELoss(z, torch.zeros_like(z))
	# 	loss = mi_loss + diff_loss

	# 	loss.backward()
	# 	self.optimizer.step()
	# 	self.optimizer.zero_grad()
	# 	pdb.set_trace()
	# 	pred = self.model.classifier(z1)
	# 	# return pred
	# 	return F.log_softmax(0.1*torch.matmul(z1, mu_t.t()) + pred, dim=1)


	@torch.enable_grad()
	def forward_and_adapt(self, x, edge_index):
		'''
		main test time adaptation mathod:
		1. update target label proportions
		2. propor entropy optimization
		3. differentiator loss
		'''
		# pdb.set_trace()
		z1, z2, z = self.model.get_embed(x, edge_index)

		sim_mat = torch.matmul(self.mu_s, z1.T)
		old_logits = self.get_pos_logits(sim_mat.detach(), self.prop)
		s_dist_old = F.softmax(old_logits, dim=0)
		prop = s_dist_old.mean(1, keepdim=True)
		self.prop = self.update_prop(prop)

		new_logits = self.get_pos_logits(sim_mat, self.prop)
		mi_loss = softmax_entropy(new_logits.T).mean(0) - softmax_entropy(new_logits.mean(0).view(1,-1))

		diff_loss = self.MSELoss(z, torch.zeros_like(z))
		loss = mi_loss + diff_loss

		loss.backward()
		self.optimizer.step()
		self.optimizer.zero_grad()
		pred = self.model.classifier(z1)
		return pred
		# return F.log_softmax(pred, dim=1)

	def reset(self):
		if self.model_state is None or self.optimizer_state is None:
			raise Exception("cannot reset without saved model/optimizer state")
		load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
	"""Entropy of softmax distribution from logits."""
	return -(x.softmax(1) * x.log_softmax(1)).sum(1)

# def KMeans(x, device, K=10, Niter=10, verbose=False):
#     """Implements Lloyd's algorithm for the Euclidean metric."""

#     start = time.time()
#     N, D = x.shape  # Number of samples, dimension of the ambient space
#     c = x[:K, :].clone()  # Simplistic initialization for the centroids
#     x_i = x.clone().unsqueeze(1)
#     c_j = c.clone().unsqueeze(0)

#     # K-means loop:
#     # - x  is the (N, D) point cloud,
#     # - cl is the (N,) vector of class labels
#     # - c  is the (K, D) cloud of cluster centroids
#     for i in range(Niter):
#         # E step: assign points to the closest cluster -------------------------
#         D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
#         cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

#         # M step: update the centroids to the normalized cluster average: ------
#         # Compute the sum of points per cluster:
#         c.zero_()
#         c.scatter_add_(0, cl[:, None].repeat(1, D), x)

#         # Divide by the number of points per cluster:
#         Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
#         c /= Ncl  # in-place division to compute the average

#     if verbose:  # Fancy display -----------------------------------------------
#         if use_cuda:
#             torch.cuda.synchronize()
#         end = time.time()
#         print(
#             f"K-means for the Euclidean metric with {N:,} points in dimension {D:,}, K = {K:,}:"
#         )
#         print(
#             "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
#                 Niter, end - start, Niter, (end - start) / Niter
#             )
#         )

#     return cl, c


def collect_params(model):
	params = []
	names = []
	# pdb.set_trace()
	for nm, m in model.named_modules():
		if isinstance(m, projector):
			for np, p in m.named_parameters():
				if np[-6:] == 'weight' or np[-4:] == 'bias':
					params.append(p)
					names.append(f"{nm}.{np}")
	return params, names

def copy_model_and_optimizer(model, optimizer):
	model_state = deepcopy(model.state_dict())
	optimizer_state = deepcopy(optimizer.state_dict())
	return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
	model.load_state_dict(model_state, strict=True)
	optimizer.load_state_dict(optimizer_state)

def configure_model(model):
	model.train()
	model.requires_grad_(False)
	for m in model.modules():
		if isinstance(m, projector):
			m.requires_grad_(True)
			m.track_running_stats = False
			m.running_mean = None
			m.running_var = None
	return model

def check_model(model):
	is_training = model.training 
	assert is_training, "Needs train mode: call model.train()"
	param_grads = [p.requires_grad for p in model.parameters()]
	has_any_params = any(param_grads)
	has_all_params = all(param_grads)
	assert has_any_params, "Needs params to update: " \
							"check which require grad"
	assert not has_all_params, "Should not update all params: " \
								"check which require grad"
	has_proj = any([isinstance(m, projector) for m in model.modules()])
	assert has_proj, "Needs update the params of projector"