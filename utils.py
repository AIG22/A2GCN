import os
import random
from args import parameter_parser
import os
import pickle as pkl
import random
import sys
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.utils import to_scipy_sparse_matrix
root = os.path.split(__file__)[0]
def distance(fea):
    #eq 1-2 masked cos similarity
    A = fea.cpu().detach().numpy()
    A2 = np.linalg.norm(A, ord=2, axis=1, keepdims=True)
    A = A / A2
    mat1 = np.expand_dims(A, 1)
    mat2 = np.expand_dims(A, 0)
    D= np.sqrt(np.sum((mat1 - mat2) ** 2, axis=-1))/2
    smvi= (np.sum(D, axis=-1, keepdims=True))/(fea.shape[0]-1)
    dis = (np.sum(smvi))/fea.shape[0]
    return dis
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def to_sparse_tensor(edge_index):
    """Convert edge_index to sparse matrix"""
    sparse_mx = to_scipy_sparse_matrix(edge_index)
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    if not isinstance(sparse_mx, sp.coo_matrix):
        sparse_mx = sp.coo_matrix(sparse_mx)
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.LongTensor(np.array([sparse_mx.row, sparse_mx.col]))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=shape
    )
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index



def get_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def normalize_adj(adj):
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return (d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)).tocoo()

def normalized_laplacian(adj):
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return (sp.eye(adj.shape[0]) - d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)).tocoo()

def gcn(adj):
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return (sp.eye(adj.shape[0]) + d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)).tocoo()

def aug_normalized_adjacency(adj):
   adj = adj + sp.eye(adj.shape[0])
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def II_normalized_adjacency(adj):
   adj_hat = 2*sp.eye(adj.shape[0])-adj
   adj_hat = sp.coo_matrix(adj_hat)
   adj =adj + 2*sp.eye(adj.shape[0])
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj_hat).dot(d_mat_inv_sqrt).tocoo()

def norm_feat(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    row_sum = np.array(features.sum(1))
    row_sum = (row_sum == 0) * 1 + row_sum
    r_inv = np.power(row_sum.astype(np.float), -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def sys_normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1))
    row_sum = (row_sum == 0) * 1 + row_sum
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    return adj.tocoo()

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

'''
def sp_to_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    if not isinstance(sparse_mx, sp.coo_matrix):
        sparse_mx = sp.coo_matrix(sparse_mx)
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.LongTensor([sparse_mx.row, sparse_mx.col])
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=shape
    )
'''
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_citation(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(root + "/data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        root + "/data/ind.{}.test.index".format(dataset_str)
    )
    test_idx_range = np.sort(test_idx_reorder)
    # Fix citeseer dataset (there are some isolated nodes in the graph)
    # Find isolated nodes, add them as zero-vecs into the right position
    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = get_mask(idx_train, labels.shape[0])
    val_mask = get_mask(idx_val, labels.shape[0])
    test_mask = get_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, labels, train_mask, val_mask, test_mask



def load_data(dataset_name, splits_file_path=None):
    nx_graph = nx.DiGraph()
    if dataset_name in {'cora', 'citeseer', 'pubmed'}:
        adj, features, labels, _, _, _ = load_citation(dataset_name)
        labels = np.argmax(labels, axis=-1)
        nclass = len(set(labels.tolist()))
        features = features.todense()
        for i in range(features.shape[0]):
            nx_graph.add_node(i, features=features[i], label=labels[i])
        coo_adj = adj.tocoo()
        indices = [coo_adj.row, coo_adj.col]
        for i in range(len(indices[0])):
            nx_graph.add_edge(indices[0][i], indices[1][i])
    else:
        graph_adjacency_list_file_path = os.path.join(
            root + '/new_data',
            dataset_name,
            'out1_graph_edges.txt'
        )
        graph_node_features_and_labels_file_path = os.path.join(
            root + "/new_data",
            dataset_name,
            'out1_node_feature_label.txt'
        )
        graph_node_features_dict = {}
        graph_labels_dict = {}
        if dataset_name == 'film':
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    feature_blank = np.zeros(932, dtype=np.uint8)
                    feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                    graph_node_features_dict[int(line[0])] = feature_blank
                    graph_labels_dict[int(line[0])] = int(line[2])
        else:
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                    graph_labels_dict[int(line[0])] = int(line[2])
        # add edges to graph
        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                if int(line[0]) not in nx_graph:
                    nx_graph.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                                      label=graph_labels_dict[int(line[0])])
                if int(line[1]) not in nx_graph:
                    nx_graph.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                                      label=graph_labels_dict[int(line[1])])
                nx_graph.add_edge(int(line[0]), int(line[1]))
        adj = nx.adjacency_matrix(nx_graph, sorted(nx_graph.nodes()))
        features = np.array(
            [features for _, features in sorted(nx_graph.nodes(data='features'), key=lambda x: x[0])]
        )
        labels = np.array(
            [label for _, label in sorted(nx_graph.nodes(data='label'), key=lambda x: x[0])]
        )
        nclass = len(set(labels.tolist()))
    features = norm_feat(features)
    with np.load(splits_file_path) as splits_file:
        train_mask = splits_file['train_mask']
        val_mask = splits_file['val_mask']
        test_mask = splits_file['test_mask']

    num_features = features.shape[1]
    num_classes = len(np.unique(labels))
    assert (np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))))

    features = torch.FloatTensor(features).cuda()
    labels = torch.LongTensor(labels).cuda()
    train_mask = torch.BoolTensor(train_mask).cuda()
    val_mask = torch.BoolTensor(val_mask).cuda()
    test_mask = torch.BoolTensor(test_mask).cuda()
    #adj = adj.tocoo()

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = sp.coo_matrix(adj)
    values = adj.data
    indices = np.vstack((adj.row, adj.col))
    edge_index = torch.LongTensor(indices).cuda()

    #

    return edge_index, features, labels, train_mask, val_mask, test_mask, num_features, num_classes
