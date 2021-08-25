#  MIT License
#
#  Copyright (c) 2019 Geom-GCN Authors
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

import os
import re

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch as th
from dgl import DGLGraph
from sklearn.model_selection import ShuffleSplit


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    # np.savetxt("rowsum.txt", rowsum, fmt="%d", delimiter=",")
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    # convert to matrix representation
    r_mat_inv = sp.diags(r_inv)
    # row-normalization
    features = r_mat_inv.dot(features)
    return features


def load_data(dataset_name, splits_file_path=None, train_percentage=None, val_percentage=None, embedding_mode=None,
              embedding_method=None,
              embedding_method_graph=None, embedding_method_space=None, all_edge=None):

    graph_adjacency_list_file_path = os.path.join('new_data', dataset_name, 'out1_graph_edges.txt')
    # place holder
    graph_node_features_and_labels_file_path = os.path.join('new_data', dataset_name,
                                                            f'{dataset_name}.tsv')
    G = nx.DiGraph()
    graph_node_features_dict = {}
    graph_labels_dict = {}

    with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
        for line in graph_node_features_and_labels_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 3)
            assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
            # map: id->feature
            graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.float32)
            # map: id->label
            graph_labels_dict[int(line[0])] = int(line[2])

    with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
        graph_adjacency_list_file.readline()
        for line in graph_adjacency_list_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 2)
            # add node to G with node feature and label (discard isolated nodes)
            if int(line[0]) not in G:
                G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                            label=graph_labels_dict[int(line[0])])
            if int(line[1]) not in G:
                G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                            label=graph_labels_dict[int(line[1])])
            # for only
            if not all_edge:
                G.add_edge(int(line[0]), int(line[1]))

    # for only space neighboorhood
    if not all_edge:
        # adjacency matrix
        adj = nx.adjacency_matrix(G, sorted(G.nodes()))

    features = np.array(
        [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
    labels = np.array(
        [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])
    # row normalization 
    features = preprocess_features(features)

    if not embedding_mode:
        # add self-loop
        g = DGLGraph(adj + sp.eye(adj.shape[0]))
    else:
        embedding_file_path = os.path.join('structural_neighborhood',
                                        f'outf_nodes_space_relation_{dataset_name}_{embedding_method}.txt')
        space_and_relation_type_to_idx_dict = {}

        with open(embedding_file_path) as embedding_file:
            for line in embedding_file:
                if line.rstrip() == 'node1,node2	space	relation_type':
                    continue
                line = re.split(r'[\t,]', line.rstrip())
                # both neighborhood in graph and in latent space have relation type
                assert (len(line) == 4)
                assert (int(line[0]) in G and int(line[1]) in G)
                # for only space neighborhood
                if all_edge:
                    if line[2] == all_edge:
                        print(line[0], line[1], line[2])
                        continue
                if (line[2], int(line[3])) not in space_and_relation_type_to_idx_dict:
                    space_and_relation_type_to_idx_dict[(line[2], int(line[3]))] = len(
                        space_and_relation_type_to_idx_dict)
                # remove original edge and add edge with feature
                if G.has_edge(int(line[0]), int(line[1])):
                    G.remove_edge(int(line[0]), int(line[1]))
                G.add_edge(int(line[0]), int(line[1]), subgraph_idx=space_and_relation_type_to_idx_dict[
                    (line[2], int(line[3]))])

        space_and_relation_type_to_idx_dict['self_loop'] = len(space_and_relation_type_to_idx_dict)

        # add self loop edge with feature
        for node in sorted(G.nodes()):
            if G.has_edge(node, node):
                G.remove_edge(node, node)
            G.add_edge(node, node, subgraph_idx=space_and_relation_type_to_idx_dict['self_loop'])
        # regenerate adjacency matrix A = A+I
        adj = nx.adjacency_matrix(G, sorted(G.nodes()))
        g = DGLGraph(adj)

        # add edge feature
        for u, v, feature in G.edges(data='subgraph_idx'):
            g.edges[g.edge_id(u, v)].data['subgraph_idx'] = th.tensor([feature])


    with np.load(splits_file_path) as splits_file:
        train_mask = splits_file['train_mask']
        val_mask = splits_file['val_mask']
        test_mask = splits_file['test_mask']
    
    # dimension of feature vector of a node
    num_features = features.shape[1]
    num_labels = len(np.unique(labels))
    assert (np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))))

    features = th.FloatTensor(features)
    labels = th.LongTensor(labels)
    train_mask = th.BoolTensor(train_mask)
    val_mask = th.BoolTensor(val_mask)
    test_mask = th.BoolTensor(test_mask)

    # Adapted from https://docs.dgl.ai/tutorials/models/1_gnn/1_gcn.html
    degs = g.in_degrees().float()
    norm = th.pow(degs, -0.5).cuda()
    norm[th.isinf(norm)] = 0
    g.ndata['norm'] = norm.unsqueeze(1)
    return g, features, labels, train_mask, val_mask, test_mask, num_features, num_labels
