import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from scipy.sparse import identity
class Lf_prop(Module):
    def __init__(self, K, in_features):
        super(Lf_prop, self).__init__()
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


    def forward(self, input, l_sym):
        #GAM = []
        #avg=[]
        #std=[]
        TEMP = F.tanh(self.temp) #(-1,1)
        s = []
        si = F.sigmoid(input @ self.scores[0] + self.bias[0])
        s.append(si)
        #avg.append(torch.mean(si))
        #std.append(torch.std(si))
        hidden = (s[0]) * input
        for i in range(self.K):
            input2 = torch.spmm(l_sym, input)
            k_i = TEMP[i]
            input = torch.sub(input, k_i * input2)
            si = F.sigmoid(input @ self.scores[i+1] + self.bias[i+1])
            s.append(si)
            #avg.append(torch.mean(si))
            #std.append(torch.std(si))
            hidden = hidden + (s[i+1])*input
            #GAM.append(k_i)
        #return hidden, GAM, avg,std
        return hidden
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class AGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass,args):
        super(AGCN, self).__init__()
        self.lin1 = Linear(nfeat, nhid)
        self.lin2 = Linear(nhid, nclass)
        self.prop1 = Lf_prop(args.K,nclass)
        self.dprate = args.dprate
        self.dropout = args.dropout

    def forward(self, x, l_sym):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            #x, GAM,avg,std = self.prop1(x, l_sym)
            x = self.prop1(x, l_sym)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            #x, GAM,avg,std = self.prop1(x, l_sym)
            x = self.prop1(x, l_sym)
            return F.log_softmax(x, dim=1)



