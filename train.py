import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import time
import sys
import torch
import torch.nn.functional as F
import utils
from args import parameter_parser
from model import AGCN
from utils import load_data, set_seed, accuracy, distance
from sklearn.metrics import f1_score
from sklearn import metrics
import numpy as np

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
args = parameter_parser()
set_seed(args.seed)
print(args)

time_results = []
def distance(fea):
    # eq 1-2 masked cos similarity
    A = fea.cpu().detach().numpy()
    A2 = np.linalg.norm(A, ord=2, axis=1, keepdims=True)
    A = abs(A / A2)
    mat1 = np.expand_dims(A, 1)
    mat2 = np.expand_dims(A, 0)
    D = np.sqrt(np.sum((abs(mat1 - mat2)) ** 2, axis=-1)) / 2
    smvi = (np.sum(D, axis=-1, keepdims=True)) / (fea.shape[0] - 1)
    dis = (np.sum(smvi)) / fea.shape[0]
    return dis

best = 999999999
acc = []
f1 = []
mad = []
NMI = []
for repeat in range(10):
    print('-------------------- Repeat {} Start -------------------'.format(repeat))
    # load data
    split_path = utils.root + '/splits/' + args.dataset + '_split_0.6_0.2_%s.npz' % repeat if args.split == 'DEFAULT' else args.split
    # load dataset
    edge_index, features, labels, train_idx, val_idx, test_idx, feat_dim, class_dim = load_data(
        args.dataset,
        split_path
    )

    # load data
    # adj, features, labels, labels_oneHot, train_idx, val_idx, test_idx = load_data(args.dataset, repeat, args.device, args.self_loop)
    print('Data load init finish')
    print('  Num features: {} | Num classes: {}'.format(
         feat_dim, class_dim))

    # init model
    model = AGCN(feat_dim, args.hidden, class_dim, args)
    model.cuda()

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam([{'params': model.lin1.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
                                  {'params': model.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
                                  {'params': model.prop1.parameters(), 'weight_decay': 0.00, 'lr': args.Agcn_lr}])

    best_epoch, best_val_acc, best_test_acc = -1, 0., 0.

    best_loss = 1000
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []
    bad_counter = 0
    running_epoch = args.epochs
    time_run = []
    for epoch in range(args.epochs):
        t_st = time.time()
        model.train()
        optimizer.zero_grad()
        logits = model(features, edge_index)
        train_loss = F.nll_loss(logits[train_idx], labels[train_idx])
        train_acc = accuracy(logits[train_idx], labels[train_idx])
        train_loss.backward()
        optimizer.step()
        time_epoch = time.time() - t_st
        time_run.append(time_epoch)

        model.eval()
        logits = model(features, edge_index)
        val_loss = F.nll_loss(logits[val_idx], labels[val_idx])
        val_acc = accuracy(logits[val_idx], labels[val_idx])
        test_acc = accuracy(logits[test_idx], labels[test_idx])

        if val_acc >= best_val_acc and test_acc >= best_test_acc:
            best_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), 'checkpoint/{}_{}_best_AGCN'.format(args.dataset,repeat))
        
        if test_acc >= best_test_acc:
            best_test_acc = test_acc

        sys.stdout.flush()
        sys.stdout.write('\r')
        sys.stdout.write(
            "Epoch #{:4d}\tTrain Loss: {:.6f} | Train Acc: {:.4f}".format(epoch, train_loss.item(), train_acc))
        sys.stdout.write(
            " | Val Loss: {:.6f} | Val Acc: {:.4f} | Test Acc: {:.4f} | Best Test Acc: {:.4f} ".format(val_loss.item(),
                                                                                                       val_acc,
                                                                                                       test_acc,
                                                                                                       best_test_acc))
    time_results.append(time_run)
    model.load_state_dict(torch.load('checkpoint/{}_{}_best_AGCN'.format(args.dataset,repeat)))
    model.eval()
    logits = model(features, edge_index)

    madgap = distance(logits)
    test_mad = madgap
    test_acc = accuracy(logits[test_idx], labels[test_idx])
    macro_f1 = f1_score(labels[test_idx].cpu(), logits[test_idx].max(1)[1].cpu().data.numpy(), average='macro')
    nmi = metrics.normalized_mutual_info_score(labels[test_idx].cpu(),
                                               logits[test_idx].max(1)[1].cpu().data.numpy())

    print('\nAGCN best_val_epoch: {}, test_acc: {:.4f}, f1:{:.4f},nmi:{:.4f},mad:{:.4f}'.format(best_epoch, test_acc,
                                                                                                macro_f1,
                                                                                                nmi, test_mad))

    # print(avg)
    # print('\n')
    # print(std)
    print('******************** Repeat {} Done ********************\n'.format(repeat))
    acc.append(round(test_acc.item(), 4))
    f1.append(round(macro_f1.item(), 4))
    NMI.append(round(nmi.item(), 4))
    mad.append(round(test_mad.item(), 4))

# avgmad = sum(mad)/10
avg = sum(acc) / 10
N = sum(NMI) / 10
m1 = np.std(acc)
m2 = np.std(NMI)

print('Avg mad: {:.4f}'.format(sum(mad) / 10))
print('Result: {}'.format(acc))
print('Avg acc: {:.4f}'.format(sum(acc) / 10))
print('acc std: {:.4f}'.format(np.std(acc)))
print('f1 Result: {}'.format(f1))
print('f1 Avg acc: {:.4f}'.format(sum(f1) / 10))
print('acc std: {:.4f}'.format(np.std(f1)))
print('nmi Result: {}'.format(NMI))
print('nmi Avg acc: {:.4f}'.format(sum(NMI) / 10))
print('acc std: {:.4f}'.format(np.std(NMI)))
print('\nAll Done!')

#f = open("./res/{}.txt".format(args.dataset),'a')
#f.write("lr:{},ber_lr:{}, wd:{} ,drop:{},drape:{}, acc_test:{},m1:{},nmi:{},m2:{}".format(args.lr, args.Agcn_lr,args.weight_decay,args.dropout,args.dprate,avg,m1,N,m2))
#f.write("\n")
#f.close()
