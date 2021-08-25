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
from utils_layers import GeomGCNNet, TwoLayers
from torch import nn
from torch.utils.data import DataLoader
from utils.dataloader import load_data
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
    parser.add_argument('--dataset_embedding', type=str)
    parser.add_argument('--bert_model', type=str)
    parser.add_argument('--num_hidden', type=int)
    parser.add_argument('--num_heads_layer_one', type=int)
    parser.add_argument('--num_heads_layer_two', type=int)
    parser.add_argument('--layer_one_ggcn_merge', type=str, default='cat')
    parser.add_argument('--layer_two_ggcn_merge', type=str, default='mean')
    parser.add_argument('--layer_one_channel_merge', type=str, default='cat')
    parser.add_argument('--layer_two_channel_merge', type=str, default='mean')
    parser.add_argument('--dropout_rate', type=float, default=0)
    parser.add_argument('--learning_rate_BERT', type=float)
    parser.add_argument('--learning_rate_FC', type=float)
    parser.add_argument('--weight_decay_linear_layer', type=float)
    parser.add_argument('--num_epochs_patience', type=int, default=100)
    parser.add_argument('--num_epochs_max', type=int, default=5000)
    parser.add_argument('--run_id', type=str)
    parser.add_argument('--dataset_split', type=str)
    parser.add_argument('--learning_rate_decay_patience', type=int, default=50)
    parser.add_argument('--learning_rate_decay_factor', type=float, default=0.8)
    args = parser.parse_args()
    vars(args)['model'] = 'Linear_TwoLayers_bert'

        
    # Set random seed.
    RANDOM_SEED = 2021
    th.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED) # Sklearn uses numpy's random seed.

    # Determine device.
    device = 'cuda' if th.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    uni_name = args.dataset

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
    bert_name = args.bert_model
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
    
    net = TwoLayers(128, 32*9, 5, args.dropout_rate)

    # add bert?
    optimizer = th.optim.Adam([{'params': bert.parameters(), 'lr': args.learning_rate_BERT},
                               {'params': net.parameters(), 'weight_decay': args.weight_decay_linear_layer, 'lr': args.learning_rate_FC}])
    learning_rate_scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                                      factor=args.learning_rate_decay_factor,
                                                                      patience=args.learning_rate_decay_patience)
    writer = tensorboardX.SummaryWriter(logdir=f'runs/{args.model}_{args.run_id}')


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
    for epoch in range(args.num_epochs_max):
    # for epoch in range(30):
        t0 = time.time()
         
        bert.train()
        net.train()
        # Compute prediction and loss
        # input_ids, attention_masks = **tokens
        bert_outputs = bert(**tokens)
        features = bert_outputs.last_hidden_state[:, 0]
        # features on GPU?
        # normalization for features TODO
        train_logits = net(features)
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
            val_logits = net(features)
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
        test_logits = net(features)
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
    with open(os.path.join('runs', f'{args.model}_{args.dataset}_results_{args.dataset_split[-5]}.txt'), 'w') as outfile:
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
    plt.savefig(f'fig/{args.model}_{args.dataset}_acc_{args.dataset_split[-5]}.jpg')

    plt.figure(dpi = 200)
    plt.plot(epoch_list,train_loss_list, '-', label='train loss')
    plt.plot(epoch_list, val_loss_list, '-', label='validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(f'fig/{args.model}_{args.dataset}_loss_{args.dataset_split[-5]}')

    with open(f'list/{args.model}_{args.dataset}_train_loss.txt', 'w') as train_loss_file:
        train_loss_file.write(str(train_loss_list))
    with open(f'list/{args.model}_{args.dataset}_train_acc.txt', 'w') as train_acc_file:
        train_acc_file.write(str(train_acc_list))
    with open(f'list/{args.model}_{args.dataset}_val_loss.txt', 'w') as val_loss_file:
        val_loss_file.write(str(val_loss_list))
    with open(f'list/{args.model}_{args.dataset}_val_acc.txt', 'w') as val_acc_file:
        val_acc_file.write(str(val_acc_list))