#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
import math
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch_geometric.utils import get_laplacian
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Parameter
from torch.nn import Linear
from torch_geometric.nn import MessagePassing

class Agcn_prop(MessagePassing):
    def __init__(self, K, in_features, **kwargs):
        super(Agcn_prop, self).__init__(aggr='add', **kwargs)
        self.in_features = in_features
        self.K = K
        self.temp = Parameter(torch.Tensor(self.K))
        self.scores = nn.ParameterList()
        self.bias = nn.ParameterList()
        for i in range(self.K + 1):
            self.scores.append(Parameter(torch.FloatTensor(self.in_features, 1)))
            self.bias.append(Parameter(torch.FloatTensor(1)))

        self.reset_parameters()
    def reset_parameters(self):
        self.temp.data.fill_(1)
        for s in self.scores:
            # s.data.fill_(0)
            stdv = 1. / math.sqrt(s.size(1))
            s.data.uniform_(-stdv, stdv)
        for b in self.bias:
            b.data.fill_(0.0)

    def forward(self, input, edge_index, edge_weight=None):
        TEMP = F.tanh(self.temp)  # (-1,1)
        s = []
        si = F.sigmoid(input @ self.scores[0] + self.bias[0])
        s.append(si)
        # avg.append(torch.mean(si))
        # std.append(torch.std(si))
        hidden = (s[0]) * input
        #S = torch.sigmoid(self.s)
        # L=I-D^(-0.5)AD^(-0.5)
        edge_index1, norm1 = get_laplacian(edge_index, edge_weight, normalization='sym', dtype=input.dtype,
                                           num_nodes=input.size(self.node_dim))
        for i in range(self.K):
            #Lx
            mid = self.propagate(edge_index1, x=input, norm=norm1, size=None)
            k_i  = TEMP[i]
            # x - k_iLx=(I-k_iL)x
            input = torch.sub(input, k_i*mid)
            si = F.sigmoid(input @ self.scores[i + 1] + self.bias[i + 1])
            s.append(si)
            # avg.append(torch.mean(si))
            # std.append(torch.std(si))
            hidden = hidden + (s[i + 1]) * input
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

class AGCN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass,args):
        super(AGCN, self).__init__()
        self.lin1 = Linear(nfeat, nhid)
        self.lin2 = Linear(nhid, nclass)
        self.prop1 = Agcn_prop(args.K,nclass)
        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, x, edge_index):

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            #x, GAM,avg,std = self.prop1(x, l_sym)
            x = self.prop1(x,edge_index)
            return F.log_softmax(x, dim=1)
            #return x
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            #x, GAM,avg,std = self.prop1(x, l_sym)
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)

###################################################################
