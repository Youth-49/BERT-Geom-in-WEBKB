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

import argparse
import json
import os
import time

import dgl.init
import numpy as np
from numpy.core.fromnumeric import sort
import tensorboardX
import torch as th
from torch.nn import parameter
import torch.nn.functional as F

import utils_data
from utils_layers import GATNet
from torch import nn
from torch.utils.data import DataLoader
from utils.dataloader import load_data
from net import train_loop, test_loop
from sklearn.model_selection import train_test_split
from utils.vis import draw_hist_loss, draw_hist_acc
# from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer as BertTokenizer
from transformers import AutoModel as BertModel


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--num_hidden', type=int)
    parser.add_argument('--num_heads_layer_one', type=int)
    parser.add_argument('--num_heads_layer_two', type=int)
    parser.add_argument('--dropout_rate', type=float)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay_layer_one', type=float)
    parser.add_argument('--weight_decay_layer_two', type=float)
    parser.add_argument('--num_epochs_patience', type=int, default=100)
    parser.add_argument('--num_epochs_max', type=int, default=5000)
    parser.add_argument('--run_id', type=str)
    parser.add_argument('--dataset_split', type=str)
    parser.add_argument('--learning_rate_decay_patience', type=int, default=50)
    parser.add_argument('--learning_rate_decay_factor', type=float, default=0.8)
    args = parser.parse_args()
    vars(args)['model'] = 'GAT_TwoLayers'

        
    # Set random seed.
    RANDOM_SEED = 2021
    th.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED) # Sklearn uses numpy's random seed.

    # Determine device.
    device = 'cuda' if th.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    uni_lt = ['cornell', 'texas', 'wisconsin']
    uni_name = uni_lt[0]

    print(f'Dealing with {uni_name} dataset.')

    # Set hyperparameters.
    max_len = 512

    # Load data.
    raw_node_ids, raw_texts, raw_labels = load_data('./datasets', uni_name)
    sorted_data = sorted([(raw_node_ids[i], raw_texts[i], raw_labels[i]) for i in range(len(raw_node_ids))], key = lambda x:int(x[0]))
    node_ids = []
    texts = []
    labels = []
    for idx in range(len(sorted_data)):
        node_ids.append(sorted_data[idx][0])
        texts.append(sorted_data[idx][1])
        labels.append(sorted_data[idx][2])

    # print(labels)
    labels = th.tensor(labels)
    # Tokenization.
    bert_name = 'prajjwal1/bert-tiny'
    tokenizer = BertTokenizer.from_pretrained(bert_name)

    tokens = tokenizer(texts, truncation=True, padding=True, max_length=max_len, return_tensors='pt')

    # features and labels are expired
    t1 = time.time()
    g, _, _, train_mask, val_mask, test_mask, num_features, num_labels = utils_data.load_data(
        args.dataset, args.dataset_split)
    print(time.time() - t1)

    # set none for node and edge without feature
    g.set_n_initializer(dgl.init.zero_initializer)
    g.set_e_initializer(dgl.init.zero_initializer)

    # num_features is depend on Bert model TODO
    num_features = 128

    # Deploy the model.
    bert = BertModel.from_pretrained(bert_name).to(device)
    # Deploy GCN
    net = GATNet(num_input_features=num_features, num_output_classes=num_labels, num_hidden=args.num_hidden,
                 dropout_rate=args.dropout_rate,
                 num_heads_layer_one=args.num_heads_layer_one, num_heads_layer_two=args.num_heads_layer_two)

    # add bert?
    optimizer = th.optim.Adam([{'params': bert.parameters(), 'lr': 6e-5},
                               {'params': net.gat1.parameters(), 'weight_decay': args.weight_decay_layer_one, 'lr': 0.005},
                               {'params': net.gat2.parameters(), 'weight_decay': args.weight_decay_layer_two, 'lr': 0.005}],)
    learning_rate_scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                                      factor=args.learning_rate_decay_factor,
                                                                      patience=args.learning_rate_decay_patience)
    writer = tensorboardX.SummaryWriter(logdir=f'runs/{args.model}_{args.run_id}')


    # labels !!
    net.cuda()
    # features = features.cuda()
    # labels = labels.cuda()
    train_mask = train_mask.cuda()
    val_mask = val_mask.cuda()
    test_mask = test_mask.cuda()
    # To GPU.
    device = next(bert.parameters()).device
    tokens, labels = tokens.to(device), labels.to(device)

    # Adapted from https://github.com/PetarV-/GAT/blob/master/execute_cora.py
    patience = args.num_epochs_patience
    # minial loss
    vlss_mn = np.inf
    # maxial acc 
    vacc_mx = 0.0
    vacc_early_model = None
    vlss_early_model = None
    state_dict_early_model = None
    curr_step = 0

    epoch_list = []
    train_acc_list = []
    train_loss_list = []
    val_acc_list = []
    val_loss_list = []



    # Adapted from https://docs.dgl.ai/tutorials/models/1_gnn/1_gcn.html
    dur = []
    last_features = None
    for epoch in range(args.num_epochs_max):
        t0 = time.time()
         
        bert.train()
        net.train()
        # Compute prediction and loss
        # input_ids, attention_masks = **tokens
        bert_outputs = bert(**tokens)
        features = bert_outputs.last_hidden_state[:, 0]

        # normalization for features TODO
        train_logits = net(g,features)
        train_logp = F.log_softmax(train_logits, 1)
        # loss function: nllloss
        train_loss = F.nll_loss(train_logp[train_mask], labels[train_mask])
        train_pred = train_logp.argmax(dim=1)
        train_acc = th.eq(train_pred[train_mask], labels[train_mask]).float().mean().item()

        optimizer.zero_grad()
        train_loss.backward()
        # update all parameters after bp
        optimizer.step()

        bert.eval()
        net.eval()
        with th.no_grad():
            bert_outputs = bert(**tokens)
            features = bert_outputs.last_hidden_state[:, 0]
            val_logits = net(g, features)
            val_logp = F.log_softmax(val_logits, 1)
            val_loss = F.nll_loss(val_logp[val_mask], labels[val_mask]).item()
            val_pred = val_logp.argmax(dim=1)
            val_acc = th.eq(val_pred[val_mask], labels[val_mask]).float().mean().item()

        learning_rate_scheduler.step(val_loss)

        dur.append(time.time() - t0)

        print(
            "Epoch {:05d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Loss {:.4f} | Val Acc {:.4f} | Time(s) {:.4f}".format(
                epoch, train_loss.item(), train_acc, val_loss, val_acc, sum(dur) / len(dur)))

        #
        epoch_list.append(epoch)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss.item())
        #

        writer.add_scalar('Train Loss', train_loss.item(), epoch)
        writer.add_scalar('Val Loss', val_loss, epoch)
        writer.add_scalar('Train Acc', train_acc, epoch)
        writer.add_scalar('Val Acc', val_acc, epoch)

        # Adapted from https://github.com/PetarV-/GAT/blob/master/execute_cora.py
        if val_acc >= vacc_mx or val_loss <= vlss_mn:
            if val_acc >= vacc_mx and val_loss <= vlss_mn:
                vacc_early_model = val_acc
                vlss_early_model = val_loss
                state_dict_early_model = net.state_dict()
            vacc_mx = np.max((val_acc, vacc_mx))
            vlss_mn = np.min((val_loss, vlss_mn))
            curr_step = 0
        else:
            curr_step += 1
            if curr_step >= patience:
                break


    # Test
    net.load_state_dict(state_dict_early_model)
    bert.eval()
    net.eval()
    with th.no_grad():
        bert_outputs = bert(**tokens)
        features = bert_outputs.last_hidden_state[:, 0]
        test_logits = net(g, features)
        test_logp = F.log_softmax(test_logits, 1)
        test_loss = F.nll_loss(test_logp[test_mask], labels[test_mask]).item()
        test_pred = test_logp.argmax(dim=1)
        test_acc = th.eq(test_pred[test_mask], labels[test_mask]).float().mean().item()
    #     test_hidden_features = net.geomgcn1(features).cpu().numpy()

    #     final_train_pred = test_pred[train_mask].cpu().numpy()
    #     final_val_pred = test_pred[val_mask].cpu().numpy()
    #     final_test_pred = test_pred[test_mask].cpu().numpy()

    results_dict = vars(args)
    results_dict['test_loss'] = test_loss
    results_dict['test_acc'] = test_acc
    results_dict['actual_epochs'] = 1 + epoch
    results_dict['val_acc_max'] = vacc_mx
    results_dict['val_loss_min'] = vlss_mn
    results_dict['total_time'] = sum(dur)
    with open(os.path.join('runs', f'{args.dataset}_results.txt'), 'w') as outfile:
        outfile.write(json.dumps(results_dict) + '\n')
    # np.savez_compressed(os.path.join('runs', f'{args.model}_{args.run_id}_hidden_features.npz'),
    #                     hidden_features=test_hidden_features)
    # np.savez_compressed(os.path.join('runs', f'{args.model}_{args.run_id}_final_train_predictions.npz'),
    #                     final_train_predictions=final_train_pred)
    # np.savez_compressed(os.path.join('runs', f'{args.model}_{args.run_id}_final_val_predictions.npz'),
    #                     final_val_predictions=final_val_pred)
    # np.savez_compressed(os.path.join('runs', f'{args.model}_{args.run_id}_final_test_predictions.npz'),
    #                     final_test_predictions=final_test_pred)

    cf_matrix = confusion_matrix(test_pred[test_mask].cpu().numpy(), labels[test_mask].cpu().numpy())
    print(cf_matrix)
    print('-------',args.dataset,args.dataset_split,'--------')
    print(
        "Total Epoch {:05d} | Test Loss {:.4f} | Test Acc {:.4f} | Val Loss mn {:.4f} | Val Acc mx {:.4f} | Total Time(s) {:.4f}".format(
            epoch, test_loss, test_acc, vlss_mn, vacc_mx, sum(dur)))
    #
    plt.figure(dpi = 200)
    plt.plot(epoch_list, train_acc_list, '-', label='train acc')
    plt.plot(epoch_list, val_acc_list, '-', label='validation acc')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig(f'fig/{args.dataset}_acc.jpg')

    plt.figure(dpi = 200)
    plt.plot(epoch_list,train_loss_list, '-', label='train loss')
    plt.plot(epoch_list, val_loss_list, '-', label='validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(f'fig/{args.dataset}_loss.jpg')

    