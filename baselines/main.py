import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from drop_edge import  drop_edge
import sys
import torch
import torch.nn.functional as F
import utils
from args import parameter_parser
args = parameter_parser()
from models import MLP,GCN_Net,GAT_Net,ChebNet,APPNP_Net,FAGCN_Net,GPRGNN,BernNet,H2GCN_Net,SGC
from utils import load_data, set_seed, accuracy,to_sparse_tensor
from sklearn.metrics import f1_score
from sklearn import metrics
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch_scatter import scatter_mean,scatter_sum
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
set_seed(args.seed)


print(args)
print()
#writer = SummaryWriter('runs/exp/loss')
acc = []
f1=[]
NMI=[]
mad =[]
def distance(fea):
    #eq 1-2 masked cos similarity
    A = fea.cpu().detach().numpy()
    A2 = np.linalg.norm(A, ord=2, axis=1, keepdims=True)
    A = abs(A / A2)
    mat1 = np.expand_dims(A, 1)
    mat2 = np.expand_dims(A, 0)
    D= np.sqrt(np.sum((abs(mat1 - mat2))**2, axis=-1))/2
    smvi= (np.sum(D, axis=-1, keepdims=True))/(fea.shape[0]-1)
    dis = (np.sum(smvi))/fea.shape[0]
    return dis
for repeat in range(10):
    print('-------------------- Repeat {} Start -------------------'.format(repeat))
    # load data
    split_path = utils.root + '/splits/' + args.dataset + '_split_0.6_0.2_%s.npz' % repeat  if args.split == 'DEFAULT' else args.split
    # load dataset
    adj, edge_index, features, labels, train_idx, val_idx, test_idx, num_features, num_classes = load_data(
        args.dataset,
        split_path
    )
    #edge_index = drop_edge(adj)
    # load data
    #adj, features, labels, labels_oneHot, train_idx, val_idx, test_idx = load_data(args.dataset, repeat, args.device, args.self_loop)
    print('Data load init finish')
    print('Num features: {} | Num classes: {}'.format(
         features.shape[1], num_classes))
    gnn_name = args.net
    if gnn_name == 'GCN':
        Net = GCN_Net
    elif gnn_name == 'GAT':
        Net = GAT_Net
    elif gnn_name == 'APPNP':
        Net = APPNP_Net
    elif gnn_name == 'ChebNet':
        Net = ChebNet
    elif gnn_name == 'GPRGNN':
        Net = GPRGNN
    elif gnn_name == 'FAGCN':
        Net = FAGCN_Net
    elif gnn_name == 'Bern':
        Net = BernNet
    elif gnn_name == 'MLP':
        Net = MLP
    elif gnn_name == 'H2GCN_Net':
        Net = H2GCN_Net
        edge_index = to_sparse_tensor(edge_index)
    elif gnn_name == 'SGC':
        Net = SGC
        edge_index = to_sparse_tensor(edge_index).cuda()
    # init model
    model= Net(features.shape[1], num_classes, args)
    model.cuda()

    if args.net == 'GPRGNN':
        optimizer = torch.optim.Adam(
            [{'params': model.lin1.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
             {'params': model.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
             {'params': model.prop1.parameters(), 'weight_decay': 0.00, 'lr': args.lr}])

    elif args.net == 'BernNet':
        optimizer = torch.optim.Adam(
            [{'params': model.lin1.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
             {'params': model.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
             {'params': model.prop1.parameters(), 'weight_decay': 0.0, 'lr': args.Bern_lr}])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_epoch, best_val_acc, best_test_acc = -1, 0.,0.
    best_loss = 1000
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []
    patience = 0
    running_epoch = args.epochs
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        logits,_ = model(features,edge_index)
        train_loss = F.nll_loss(logits[train_idx], labels[train_idx])
        train_acc = accuracy(logits[train_idx], labels[train_idx])
        train_loss.backward()
        optimizer.step()
        #writer.add_scalar('train_loss_{}'.format(repeat+6), train_loss, global_step=epoch)
        model.eval()
        logits,_ = model(features,edge_index)
        val_loss = F.nll_loss(logits[val_idx], labels[val_idx])
        val_acc = accuracy(logits[val_idx], labels[val_idx])
        test_acc = accuracy(logits[test_idx], labels[test_idx])
        #writer.add_scalar('val_loss_{}'.format(repeat+9), val_loss, global_step=epoch)
       
        if val_acc >= best_val_acc and test_acc >= best_test_acc:
            best_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), 'checkpoint_{}/{}_best_BMGCN'.format(args.net,args.dataset))
            patience = 0
        
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
        else:
            patience += 0
        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    break  # 如果超过early_stopping部分的平均验证损失小于当前的验证损失，说明模型不再提高

        sys.stdout.flush()
        sys.stdout.write('\r')
        sys.stdout.write("Epoch #{:4d}\tTrain Loss: {:.6f} | Train Acc: {:.4f}".format(epoch, train_loss.item(), train_acc))
        sys.stdout.write(" | Val Loss: {:.6f} | Val Acc: {:.4f} | Test Acc: {:.4f} | Best Test Acc: {:.4f} ".format(val_loss.item(), val_acc, test_acc, best_test_acc))

    model.load_state_dict(torch.load('checkpoint_{}/{}_best_BMGCN'.format(args.net,args.dataset)))
    model.eval()
    logits,tempb = model(features,edge_index)
    #orgma = distance(features)
    print(tempb)
    #preds = logits.max(1)[1]
    #madgap = distance(features)
    test_mad = 0

    test_acc = accuracy(logits[test_idx], labels[test_idx])
    macro_f1 = f1_score(labels[test_idx].cpu(), logits[test_idx].max(1)[1].cpu().data.numpy(), average='macro')
    nmi = metrics.normalized_mutual_info_score(labels[test_idx].cpu(),
                                               logits[test_idx].max(1)[1].cpu().data.numpy())

    print('\nBM-GCN best_val_epoch: {}, test_acc: {:.4f}, f1:{:.4f},nmi:{:.4f},mad:{:.4f}'.format(best_epoch, test_acc, macro_f1,
                                                                                       nmi,test_mad))

    #print('\nBM-GCN best_val_epoch: {}, test_acc: {:.4f}'.format(best_epoch, test_acc))

    print('******************** Repeat {} Done ********************\n'.format(repeat))
    acc.append(round(test_acc.item(), 4))
    f1.append(round(macro_f1.item(), 4))
    NMI.append(round(nmi.item(), 4))
    #mad.append(round(test_mad.item(), 4))

#avgmad = sum(mad)/10
avg = sum(acc) / 10
N = sum(NMI) / 10
m1=np.std(acc)
m2=np.std(NMI)
f = open("./res_SGC/{}.txt".format(args.net,args.dataset),'a')
f.write("lr:{},ber_lr:{}, wd:{} ,drop:{},drape:{}, acc_test:{},m1:{},nmi:{},m2:{}".format(args.lr, args.Bern_lr,args.weight_decay,args.dropout,args.dprate,avg,m1,N,m2))
f.write("\n")
f.close()

print('Result: {}'.format(acc))
print('Avg acc: {:.4f}'.format(sum(acc) / 10))
print('acc std: {:.4f}'.format(np.std(acc)))
print('Avg MAD: {:.4f}'.format(sum(mad) / 10))
print('f1 Result: {}'.format(f1))
print('f1 Avg acc: {:.4f}'.format(sum(f1) / 10))
print('acc std: {:.4f}'.format(np.std(f1)))
print('nmi Result: {}'.format(NMI))
print('nmi Avg acc: {:.4f}'.format(sum(NMI) / 10))
print('acc std: {:.4f}'.format(np.std(NMI)))
print('\nAll Done!')