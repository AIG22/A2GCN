import re
import argparse
import torch


def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--early_stopping', type=int, default=200)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=2108550661, help='Seed.')
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.1)  # APPNP
    parser.add_argument('--epsilon', type=float, default=0.5)  # FAGCN
    parser.add_argument('--k', type=float, default=2)  # H2GCN
    parser.add_argument('--patience', type=int, default=100, help='Patience for early stopping.')
    parser.add_argument('--split', type=str, default="DEFAULT", help='data split to use')
    parser.add_argument('--dprate', type=float, default=0.0)
    parser.add_argument('--C', type=int)
    parser.add_argument('--Init', type=str,
                        choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'],
                        default='PPR')
    parser.add_argument('--Gamma', default=None)  # GPRGCN
    parser.add_argument('--ppnp', default='GPR_prop',
                        choices=['PPNP', 'GPR_prop'])
    parser.add_argument('--heads', default=8, type=int)
    parser.add_argument('--output_heads', default=1, type=int)

    parser.add_argument('--dataset',choices=['cora', 'citeseer', 'pubmed', 'chameleon', 'squirrel', 'film', 'cornell', 'texas','wisconsin'], default='cornell')
    parser.add_argument('--net', type=str,
                        choices=['GCN', 'GAT','SGC', 'APPNP', 'ChebNet', 'MLP', 'GPRGNN', 'FAGCN', 'Bern', 'H2GCN_Net'],
                        default='Bern')
    parser.add_argument('--no_cuda', action='store_false', default=True,
                    help='Using CUDA or not. Default is True (Using CUDA).')

    parser.add_argument('--Bern_lr', type=float, default=0.001, help='learning rate for BernNet propagation layer.')
    args = parser.parse_args()
    return args
