from typing import Optional
from torch_geometric.typing import OptTensor
import numpy as np
import math
import torch
import torch_sparse
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import os.path as osp
import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.autograd import Variable
from torch.nn import Linear
from torch_geometric.nn import GATConv, GCNConv, ChebConv, FAConv
from torch_geometric.nn import MessagePassing, APPNP
from torch_geometric.utils import to_scipy_sparse_matrix
import scipy.sparse as sp
from scipy.special import comb
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.utils import get_laplacian

#########################SGC##############################
def sgc_precompute(features, adj, degree=2):
    for i in range(degree):
        features = torch.spmm(adj, features)
    return features

class SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, nfeat, nclass,args):
        super(SGC, self).__init__()

        self.W = nn.Linear(nfeat, nclass)

    def forward(self, x,adj):
        x = self.W(x)
        x= sgc_precompute(x,adj)

        return F.log_softmax(x, dim=1)
###########################################################
###########################MLP begin###################################
class MLP(torch.nn.Module):
    def __init__(self, nfeat,nclass, args):
        super(MLP, self).__init__()

        self.lin1 = Linear(nfeat, args.hidden)
        self.lin2 = Linear(args.hidden, nclass)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=1)
###########################MLP end#####################################

###########################GCN begin###################################
class GCN_Net(torch.nn.Module):
    def __init__(self, nfeat,nclass, args):
        super(GCN_Net, self).__init__()
        self.conv1 = GCNConv(nfeat, args.hidden)
        self.conv2 = GCNConv(args.hidden, nclass)
        #self.conv3 = GCNConv(nclass, nclass)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        #self.conv3.reset_parameters()

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
###########################GCN end#####################################

###########################Cheby begin###################################
class ChebNet(torch.nn.Module):
    def __init__(self, nfeat,nclass, args):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(nfeat, 32, K=2)
        self.conv2 = ChebConv(32, nclass, K=2)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

###########################Cheby end#####################################

###########################GAT begin###################################
class GAT_Net(torch.nn.Module):
    def __init__(self, nfeat,nclass, args):
        super(GAT_Net, self).__init__()
        self.conv1 = GATConv(
            nfeat,
            args.hidden,
            heads=args.heads,
            dropout=args.dropout)
        self.conv2 = GATConv(
            args.hidden * args.heads,
            nclass,
            heads=args.output_heads,
            concat=False,
            dropout=args.dropout)
        self.conv3 = GATConv(
            nclass,
            nclass,
            heads=args.output_heads,
            concat=False,
            dropout=args.dropout)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv3(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)
###########################GAT end#####################################

###########################APPNP begin###################################
class APPNP_Net(torch.nn.Module):
    def __init__(self, nfeat,nclass, args):
        super(APPNP_Net, self).__init__()
        self.lin1 = Linear(nfeat, args.hidden)
        self.lin2 = Linear(args.hidden, nclass)
        self.prop1 = APPNP(args.K, args.alpha)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)
###########################APPNP end#####################################

###########################FAGNN begin###################################
class FAGCN_Net(torch.nn.Module):
    def __init__(self, nfeat,nclass, args):
        super(FAGCN_Net, self).__init__()
        in_channels = nfeat
        out_channels = nclass
        self.eps = args.epsilon
        self.layer_num = 4
        self.dropout = args.dropout
        self.hidden = args.hidden

        self.layers = torch.nn.ModuleList()
        for _ in range(self.layer_num):
            self.layers.append(FAConv(self.hidden,self.eps,self.dropout))

        self.lin1 = torch.nn.Linear(in_channels, self.hidden)
        self.lin2 = torch.nn.Linear(self.hidden, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.lin1.weight, gain=1.414)
        torch.nn.init.xavier_normal_(self.lin2.weight, gain=1.414)

    def forward(self, x,edge_index):
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = torch.relu(self.lin1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        raw = h
        for i in range(self.layer_num):
            h = self.layers[i](h,raw,edge_index)
        h = self.lin2(h)
        return F.log_softmax(h, dim=1)
###########################FAGNN end#####################################

###########################GPR begin###################################
class GPR_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like, note that in this case, alpha has to be a integer. It means where the peak at when initializing GPR weights.
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

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)
class GPRGNN(torch.nn.Module):
    def __init__(self, nfeat,nclass, args):
        super(GPRGNN, self).__init__()
        self.lin1 = Linear(nfeat, args.hidden)
        self.lin2 = Linear(args.hidden, nclass)

        if args.ppnp == 'PPNP':
            self.prop1 = APPNP(args.K, args.alpha)
        elif args.ppnp == 'GPR_prop':
            self.prop1 = GPR_prop(args.K, args.alpha, args.Init, args.Gamma)

        self.Init = args.Init
        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, x, edge_index ):

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
###########################GPR end#####################################

###########################Bern begin###################################
class Bern_prop(MessagePassing):
    def __init__(self, K, bias=True, **kwargs):
        super(Bern_prop, self).__init__(aggr='add', **kwargs)

        self.K = K
        self.temp = Parameter(torch.Tensor(self.K + 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1)

    def forward(self, x, edge_index, edge_weight=None):
        TEMP = F.relu(self.temp)

        # L=I-D^(-0.5)AD^(-0.5)
        edge_index1, norm1 = get_laplacian(edge_index, edge_weight, normalization='sym', dtype=x.dtype,
                                           num_nodes=x.size(self.node_dim))
        # 2I-L
        edge_index2, norm2 = add_self_loops(edge_index1, -norm1, fill_value=2., num_nodes=x.size(self.node_dim))

        tmp = []
        tmp.append(x)
        for i in range(self.K):
            x = self.propagate(edge_index2, x=x, norm=norm2, size=None)
            tmp.append(x)

        out = (comb(self.K, 0) / (2 ** self.K)) * TEMP[0] * tmp[self.K]

        for i in range(self.K):
            x = tmp[self.K - i - 1]
            x = self.propagate(edge_index1, x=x, norm=norm1, size=None)
            for j in range(i):
                x = self.propagate(edge_index1, x=x, norm=norm1, size=None)

            out = out + (comb(self.K, i + 1) / (2 ** self.K)) * TEMP[i + 1] * x
        return out,TEMP

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

class BernNet(torch.nn.Module):
    def __init__(self,nfeat,nclass, args):
        super(BernNet, self).__init__()
        self.lin1 = Linear(nfeat, args.hidden)
        self.lin2 = Linear(args.hidden, nclass)
        self.m = torch.nn.BatchNorm1d(nclass)
        self.prop1 = Bern_prop(args.K)
        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        #x= self.m(x)

        if self.dprate == 0.0:
            x,tmp = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1),tmp
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x,tmp = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1),tmp

###########################Bern end#####################################
###########################H2GCN begin#####################################
class H2GCN_Net(torch.nn.Module):
    def __init__(self, nfeat,nclass, args):
        super(H2GCN_Net, self).__init__()
        dropout = args.dropout
        hidden_dim = args.hidden
        feat_dim = nfeat
        class_dim = nclass
        k = args.k

        self.dropout = dropout
        self.k = k
        self.w_embed = torch.nn.Parameter(
            torch.zeros(size=(feat_dim, hidden_dim)),
            requires_grad=True
        )
        self.w_classify = torch.nn.Parameter(
            torch.zeros(size=((2 ** (self.k + 1) - 1) * hidden_dim, class_dim)),
            requires_grad=True
        )
        self.params = [self.w_embed, self.w_classify]
        self.initialized = False
        self.a1 = None
        self.a2 = None
        self.reset_parameter()

    def reset_parameter(self):
        torch.nn.init.xavier_uniform_(self.w_embed)
        torch.nn.init.xavier_uniform_(self.w_classify)

    @staticmethod
    def _indicator(sp_tensor: torch.sparse.Tensor) -> torch.sparse.Tensor:
        csp = sp_tensor.coalesce()
        return torch.sparse_coo_tensor(
            indices=csp.indices(),
            values=torch.where(csp.values() > 0, torch.tensor(1).cuda(), torch.tensor(0).cuda()),
            size=csp.size(),
            dtype=torch.float
        )

    @staticmethod
    def _spspmm(sp1: torch.sparse.Tensor, sp2: torch.sparse.Tensor) -> torch.sparse.Tensor:
        assert sp1.shape[1] == sp2.shape[0], 'Cannot multiply size %s with %s' % (sp1.shape, sp2.shape)
        sp1, sp2 = sp1.coalesce(), sp2.coalesce()
        index1, value1 = sp1.indices(), sp1.values()
        index2, value2 = sp2.indices(), sp2.values()
        m, n, k = sp1.shape[0], sp1.shape[1], sp2.shape[1]
        indices, values = torch_sparse.spspmm(index1, value1, index2, value2, m, n, k)
        return torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=(m, k),
            dtype=torch.float
        )

    @classmethod
    def _adj_norm(cls, adj: torch.sparse.Tensor) -> torch.sparse.Tensor:
        n = adj.size(0)
        d_diag = torch.pow(torch.sparse.sum(adj, dim=1).values(), -0.5)
        d_diag = torch.where(torch.isinf(d_diag), torch.full_like(d_diag, 0), d_diag)
        d_tiled = torch.sparse_coo_tensor(
            indices=[list(range(n)), list(range(n))],
            values=d_diag,
            size=(n, n)
        )
        return cls._spspmm(cls._spspmm(d_tiled, adj), d_tiled)

    def _prepare_prop(self, adj):
        n = adj.size(0)
        device = adj.device
        self.initialized = True
        sp_eye = torch.sparse_coo_tensor(
            indices=[list(range(n)), list(range(n))],
            values=[1.0] * n,
            size=(n, n),
            dtype=torch.float
        ).cuda()
        # initialize A1, A2
        a1 = self._indicator(adj.cuda() - sp_eye)
        a2 = self._indicator(self._spspmm(adj, adj).cuda() - adj.cuda() - sp_eye)
        # norm A1 A2
        self.a1 = self._adj_norm(a1)
        self.a2 = self._adj_norm(a2)

    def forward(self, x,adj):

        if not self.initialized:
            self._prepare_prop(adj)
        # H2GCN propagation
        rs = [F.relu(torch.mm(x, self.w_embed))]
        for i in range(self.k):
            r_last = rs[-1]
            r1 = torch.spmm(self.a1.cuda(), r_last.cuda())
            r2 = torch.spmm(self.a2.cuda(), r_last.cuda())
            rs.append(torch.cat([r1, r2], dim=1))
        r_final = torch.cat(rs, dim=1)
        r_final = F.dropout(r_final, self.dropout, training=self.training)
        return F.log_softmax(torch.mm(r_final, self.w_classify), dim=1)
###########################H2GCN end#####################################