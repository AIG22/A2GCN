import argparse
import torch


def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2108550661, help='seeds for random splits.')
    parser.add_argument('--epochs', type=int, default=1000, help='max epochs.')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay.')
    parser.add_argument('--early_stopping', type=int, default=200, help='early stopping.')
    parser.add_argument('--hidden', type=int, default=64, help='hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout for neural networks.')
    parser.add_argument('--patience', type=int, default=100, help='Patience for early stopping.')
    parser.add_argument('--K', type=int, default=10, help='propagation steps.')
    parser.add_argument('--dprate', type=float, default=0.6, help='dropout for propagation layer.')
    parser.add_argument('--split', type=str, default="DEFAULT", help='data split to use')
    parser.add_argument('--dataset', type=str, choices=['cora','citeseer','pubmed','computers','photo','chameleon','squirrel','actor','texas','cornell','wisconsin','film'],
                        default='chameleon')
    parser.add_argument('--net', type=str, choices=['GCN', 'GAT', 'APPNP', 'ChebNet', 'GPRGNN','AGNN','MLP'], default='AGNN')
    parser.add_argument('--Agnn_lr', type=float, default=0.05, help='learning rate for BernNet propagation layer.')

    args = parser.parse_args()
    return args