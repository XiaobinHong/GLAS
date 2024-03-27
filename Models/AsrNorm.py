import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import pdb

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class ASRNormBN(nn.Module):
    def __init__(self, dim, cop_dim=12, eps=1e-6):
        super(ASRNormBN, self).__init__()
        self.eps = eps
        self.standard_encoder = nn.Linear(dim, dim // cop_dim)
        self.rescale_encoder = nn.Linear(dim, dim // cop_dim)
        self.standard_mean_decoder = nn.Linear(dim // cop_dim, dim)
        self.standard_var_decoder = nn.Linear(dim // cop_dim, dim)
        self.rescale_mean_decoder = nn.Linear(dim // cop_dim, dim)
        self.rescale_var_decoder = nn.Linear(dim // cop_dim, dim)

        self.lambda_1 = nn.Parameter(torch.zeros(dim)-5)
        self.lambda_2 = nn.Parameter(torch.zeros(dim)-5)

        self.bias_1 = nn.Parameter(torch.zeros(dim))
        # training image net in one hour suggest to initialize as 0
        self.bias_2 = nn.Parameter(torch.zeros(dim))
        self.drop_out = nn.Dropout(p=0.5)

    def init(self):
        pass

    def forward(self, x):
        '''

        :param x: B, d
        :return:
        '''
        # pdb.set_trace()
        lambda_1 = torch.sigmoid(self.lambda_1)
        lambda_2 = torch.sigmoid(self.lambda_2)

        real_mean = x.mean(0)
        real_var = x.std(0)

        asr_mean = self.standard_mean_decoder(
            F.relu(self.standard_encoder(self.drop_out(real_mean.view(1, -1))))).squeeze()
        asr_var = F.relu(
            self.standard_var_decoder(F.relu(self.standard_encoder(self.drop_out(real_var.view(1, -1)))))).squeeze()
        mean = lambda_1 * asr_mean + (1 - lambda_1) * real_mean
        var = lambda_2 * asr_var + (1 - lambda_2) * real_var

        x = (x - mean) / (var + self.eps)

        asr_mean = torch.tanh(self.rescale_mean_decoder(
            F.relu(self.rescale_encoder(self.drop_out(real_mean.view(1, -1)))))).squeeze() + self.bias_1
        asr_var = torch.sigmoid(
            self.rescale_var_decoder(
                F.relu(self.rescale_encoder(self.drop_out(real_var.view(1, -1)))))).squeeze() + self.bias_2
        x = x * asr_var + asr_mean
        return x

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()