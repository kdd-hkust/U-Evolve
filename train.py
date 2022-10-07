# -*- coding: utf-8 -*-
import os
import sys
import argparse
import numpy as np
import torch
import time
import dgl
import torch.nn as nn
import logging
import random
from utils import *
from model import *
from sklearn import metrics
from sklearn.metrics import classification_report

which_gpu = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = which_gpu
parser = argparse.ArgumentParser(description='Train Urban Vibrancy Forecast Model')
parser.add_argument('--enable_cuda', action='store_true', default=True, help='Disables CUDA training.')
parser.add_argument('--dataset', type=str, default='beijing', help='selected city')
parser.add_argument('--batch_size', type=int, default=5, help='Number of batch to train and test.')
parser.add_argument('--seed', type=int, default=33, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--t_in', type=int, default=3, help='Input time step.')
parser.add_argument('--t_out', type=int, default=1, help='Output time step.')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=50, help='Patience')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--hid_dim', type=int, default=32, help='Dim of hidden units.')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
parser.add_argument('--gat_hop', type=int, default=1, help='Number of hop in graph.')
parser.add_argument('--distance_top', type=int, default=5, help='topk connection in distance graph.')
parser.add_argument('--mobility_top', type=int, default=5, help='topk connection in mobility graph.')
parser.add_argument('--semantic_top', type=int, default=5, help='topk connection in semantic graph.')




args = parser.parse_args()


HOME = 'FILE PATH'  
logging.basicConfig(level = logging.INFO,filename=HOME+'logs/train.log',\
                        format = '%(asctime)s - %(process)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.enable_cuda and torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')
print(args)
logger.info(args)

total_train_loss = []
total_val_loss = []
total_test_loss = []

def train_epoch(loader_train, G_mobility, G_semantic, G_distance):

    G_mobility = G_mobility.copy()
    train_loss = []
    global total_train_loss

    for i,(X_batch,M_batch,Y_batch,Z) in enumerate(loader_train):
        net.train()
        X_batch = X_batch.to(device=args.device)
        M_batch = M_batch.to(device=args.device)
        Y_batch = Y_batch.to(device=args.device)
        Z = Z.tolist()
        print(Z)
        G_mobilitys=[]
        G_semantics = []
        for t_num in range(args.t_in):
            G_mobility_tmp=[]
            G_semantic_tmp=[]
            for idx in Z:
                G_mobility_tmp.append(G_mobility[idx][t_num])
                G_semantic_tmp.append(G_semantic[idx][t_num])
            G_mobility_tmp = dgl.batch(G_mobility_tmp)
            G_semantic_tmp = dgl.batch(G_semantic_tmp)
            G_mobilitys.append(G_mobility_tmp)
            G_semantics.append(G_semantic_tmp)

        now_batch = X_batch.shape[0]
        G_dist_copy = make_dgl_graph(G_distance[:now_batch])
        optimizer.zero_grad()
        y_pred = net(X_batch, X_static,M_batch,G_mobilitys,G_semantics,G_dist_copy)
        clf_loss = loss_criterion(y_pred.view(-1, y_pred.size()[-1]), Y_batch.view(-1).long())
        loss = clf_loss
        loss.backward()
        optimizer.step()
        train_loss.append(loss.detach().cpu().numpy())
        
    total_train_loss.append(sum(train_loss)/len(train_loss))
    return total_train_loss[-1]

def test_epoch(loader_val, G_mobility, G_semantic, G_distance,stage):
    global total_val_loss
    global total_test_loss
    val_loss = []
    val_acc = []
    y_labels =[]
    y_preds =[]
    y_pred_probs = []
    
    for i,(X_batch,M_batch,Y_batch,Z) in enumerate(loader_val):
        net.eval()
        X_batch = X_batch.to(device=args.device)
        M_batch = M_batch.to(device=args.device)
        Y_batch = Y_batch.to(device=args.device)
        Z = Z.tolist()
        G_mobilitys=[]
        G_semantics = []
        for t_num in range(args.t_in):
            G_mobility_tmp=[]
            G_semantic_tmp=[]
            for idx in Z:
                G_mobility_tmp.append(G_mobility[idx][t_num])
                G_semantic_tmp.append(G_semantic[idx][t_num])
            G_mobility_tmp = dgl.batch(G_mobility_tmp)
            G_semantic_tmp = dgl.batch(G_semantic_tmp)
            G_mobilitys.append(G_mobility_tmp)
            G_semantics.append(G_semantic_tmp)

        now_batch = X_batch.shape[0]
        G_dist_copy = make_dgl_graph(G_distance[:now_batch])
        y_pred = net(X_batch, X_static, M_batch,G_mobilitys,G_semantics,G_dist_copy) # (B, N,class)

        loss_val_clf = loss_criterion(y_pred.view(-1, y_pred.size()[-1]), Y_batch.view(-1).long())
        
        loss_val = loss_val_clf
        val_loss.append(np.asscalar(loss_val.detach().cpu().numpy()))
                
        y_label_clf = Y_batch.detach().cpu().numpy() # (B,N)
        y_pred_prob = y_pred.detach().cpu().numpy() # (B,N,class)

        y_pred_clf = y_pred_prob.argmax(axis=-1)
        y_labels.append(y_label_clf)
        y_preds.append(y_pred_clf)
        y_pred_probs.append(y_pred_prob)

    
    cur_loss = sum(val_loss)/len(val_loss)
    if stage == 'validation':
        total_val_loss.append(cur_loss)
    elif stage == 'test':
        total_test_loss.append(cur_loss)
    
    class_num = 3
    y_pred = np.concatenate(y_preds, axis=0) # (t*B,N)
    y_label = np.concatenate(y_labels, axis=0) #(t*B,N)
    y_prob = np.concatenate(y_pred_probs, axis=0) # (t*B,N, class)

    y_pred = y_pred.reshape(-1, 1) # (B*N, 1)
    y_label = y_label.reshape(-1,1) # (B*N, 1)
    y_prob = y_prob.reshape(-1, class_num) # (B*N, class)

    y_prob -= np.max(y_prob)
    y_prob = np.exp(y_prob)/(np.sum(np.exp(y_prob), axis=-1).reshape(-1,1))


    back_metrics, str_metrics = get_triple_metric(y_label, y_pred, y_prob, cur_loss, stage)
    
    return cur_loss, back_metrics, str_metrics

def get_triple_metric(y_label, y_pred, y_prob, loss, stage):
    target_names = ['decline', 'stable', 'bloom']

    micro_precision = metrics.precision_score(y_label, y_pred, average='micro')
    macro_precision = metrics.precision_score(y_label, y_pred, average='macro')    
    micro_recall = metrics.recall_score(y_label, y_pred, average='micro')
    macro_recall = metrics.recall_score(y_label, y_pred, average='macro')
    micro_f1 = metrics.f1_score(y_label, y_pred, average='micro')
    macro_f1 = metrics.f1_score(y_label, y_pred, average='macro')   
    clf_report = classification_report(y_label, y_pred, target_names=target_names)
    weighted_f1 = metrics.f1_score(y_label, y_pred, average='weighted')


    micro_precision_str = "micro_precision: {}".format(','+(str(micro_precision)))
    macro_precision_str = "macro_precision: {}".format(','+(str(macro_precision)))
    macro_recall_str = "macro_recall: {}".format(','+(str(macro_recall)))
    macro_f1_str = "macro_f1: {}".format(','+(str(macro_f1)))
    weighted_f1_str = "weighted_f1: {}".format(','+(str(weighted_f1)))
    stage_str = stage + " metrics: auc, micro precision, macro precisoin, micro recall, macro recall,\
                        micro F1 score, macro F1 score, report, loss"
    loss_str = "loss: {}".format(str(loss))

    tmp_metrics = [micro_precision, macro_precision, macro_recall, macro_f1, weighted_f1]
    str_metrics = [stage_str, loss_str, micro_precision_str, macro_precision_str, macro_recall_str, macro_f1_str, weighted_f1_str, clf_report]

    for cur_metric in str_metrics:
        print(cur_metric)
        logger.info(cur_metric)

    return tmp_metrics, str_metrics[1:]

def make_dgl_graph(graph_list): 
    return dgl.batch(graph_list)

if __name__ == '__main__':
    loader_train,loader_val,loader_test, X_static, F_dim, F_static, N, dgl_adj_distance, G_mobility_train, G_semantic_train, G_mobility_val, G_semantic_val, G_mobility_test, G_semantic_test = load_graph_data(
                                                    dataset=args.dataset,
                                                    T_in=args.t_in,
                                                    T_out=args.t_out,
                                                    Batch_Size=args.batch_size,
                                                    device = args.device,
                                                    distance_top=args.distance_top,
                                                    mobility_top=args.mobility_top,
                                                    semantic_top=args.semantic_top)
    
    X_static = X_static.to(device=args.device)
    net = GLBFM(args = args,
            t_in=args.t_in,
            t_out=args.t_out,
            nfeat=F_dim,
            nstatic=F_static,
            task=args.task,
            dropout=args.dropout, 
            alpha=args.alpha, 
            hid_dim=args.hid_dim,
            gat_hop = args.gat_hop,
            device=args.device).to(device=args.device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_criterion = nn.CrossEntropyLoss()

    max_f1 = 0
    best_epoch = 0
    for epoch in range(args.epochs):
        st_time = time.time()
        '''training'''
        print('training......', epoch)
        loss_train = train_epoch(loader_train,G_mobility_train,G_semantic_train,dgl_adj_distance)
        '''validating'''
        with torch.no_grad():
            print('validating......')
            val_loss, val_metrics, val_str_metrics = test_epoch(loader_val,G_mobility_val,G_semantic_val,dgl_adj_distance,stage='validation')
        '''testing'''
        with torch.no_grad():
            print('testing......')
            test_loss, test_metrics, test_str_metrics = test_epoch(loader_test,G_mobility_test,G_semantic_test,dgl_adj_distance,stage='test')
        
        val_f1_tmp = val_metrics[-2]

        if(val_f1_tmp > max_f1):
            max_f1 = val_f1_tmp
            best_epoch = epoch + 1
            best_metrics = test_metrics
            best_str_metrics = test_str_metrics
            best_loss = test_loss
        
        print("Epoch: {}".format(epoch+1))
        logger.info("Epoch: {}".format(epoch+1))
        print("Train loss: {}".format(loss_train))
        logger.info("Train loss: {}".format(loss_train))
        print("Best Epoch: {}".format(best_epoch))
        logger.info("Best Epoch: {}".format(best_epoch))
        for cur_metric in best_str_metrics:
            print(cur_metric)
            logger.info(cur_metric)
        print('time: {:.4f}s'.format(time.time() - st_time))
        logger.info('time: {:.4f}s\n'.format(time.time() - st_time))
        

        if(epoch+1 - best_epoch >= args.patience):
       
            sys.exit(0)
