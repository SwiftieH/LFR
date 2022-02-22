import torch
import torch.nn.functional as F
import math
from torch.nn import Parameter
import random

class GCNConvLFR(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, num_comps, bias=True, filters = None, no_share_para = False):
        super(GCNConvLFR, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.no_share_para = no_share_para
        if self.no_share_para:
            self.weight_LFR = Parameter(torch.FloatTensor(in_features, out_features))
        if filters is None:
            self.filter = Parameter(torch.FloatTensor(num_comps))
            torch.nn.init.uniform_(self.filter, 0.9, 1.1)
        else:
            self.filter = torch.FloatTensor(filters)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.no_share_para:
            self.weight_LFR.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def forward_LFR(self, input, supports):
        if self.no_share_para:
            transformed_features = torch.mm(input, self.weight_LFR)
        else:
            transformed_features = torch.mm(input, self.weight)
        output = torch.mm(torch.mm(supports[0].to_dense(), torch.diag(self.filter.to(input.device))),
                                          torch.mm(supports[1].to_dense(), transformed_features))
        if self.bias is not None:
            return output + self.bias
        else:
            return output

