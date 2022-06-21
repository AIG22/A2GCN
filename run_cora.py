import sys
import torch
import torch.nn.functional as F
import utils
from args import parameter_parser
from model import AGNN
from utils import load_data, set_seed, accuracy, distance
from sklearn.metrics import f1_score
from sklearn import metrics
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
args = parameter_parser()
set_seed(args.seed)
print(args)

acc = []
f1 = []
mad = []
NMI = []
for repeat in range(10):
    print('-------------------- Repeat {} Start -------------------'.format(repeat))
    # load data
    split_path = utils.root + '/splits/' + args.dataset + '_split_0.6_0.2_%s.npz' % repeat if args.split == 'DEFAULT' else args.split
    # load dataset
    adj, features, labels, train_idx, val_idx, test_idx, feat_dim, class_dim = load_data(
        args.dataset,
        split_path
    )

    # load data
    # adj, features, labels, labels_oneHot, train_idx, val_idx, test_idx = load_data(args.dataset, repeat, args.device, args.self_loop)
    print('Data load init finish')
    print(' Num nodes: {} | Num features: {} | Num classes: {}'.format(
        adj.shape[0], feat_dim, class_dim))

    # init model
    model = AGNN(feat_dim, args.hidden, class_dim, args)
    model.cuda()


    model.load_state_dict(torch.load('checkpoint/{}_{}_best_AGCN'.format(args.dataset,repeat)))
    model.eval()
    logits = model(features, adj)

    madgap = distance(logits)
    test_mad = madgap
    test_acc = accuracy(logits[test_idx], labels[test_idx])
    macro_f1 = f1_score(labels[test_idx].cpu(), logits[test_idx].max(1)[1].cpu().data.numpy(), average='macro')
    nmi = metrics.normalized_mutual_info_score(labels[test_idx].cpu(),
                                               logits[test_idx].max(1)[1].cpu().data.numpy())

    print('test_acc: {:.4f}, f1:{:.4f},nmi:{:.4f},mad:{:.4f}'.format(test_acc,macro_f1,nmi, test_mad))

    # print(avg)
    # print('\n')
    # print(std)
    print('******************** Repeat {} Done ********************\n'.format(repeat))
    acc.append(round(test_acc.item(), 4))
    f1.append(round(macro_f1.item(), 4))
    NMI.append(round(nmi.item(), 4))
    # mad.append(round(test_mad.item(), 4))

# avgmad = sum(mad)/10
avg = sum(acc) / 10
N = sum(NMI) / 10
m1 = np.std(acc)
m2 = np.std(NMI)
# f = open("./res/{}.txt".format(args.dataset),'a')
# f.write("lr:{},ber_lr:{}, wd:{} ,drop:{},drape:{}, acc_test:{},m1:{},nmi:{},m2:{},mad:{:.4f}".format(args.lr, args.Agnn_lr,args.weight_decay,args.dropout,args.dprate,avg,m1,N,m2,avgmad ))
# f.write("\n")
# f.close()
# print('Avg mad: {:.4f}'.format(sum(mad) / 10))
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


