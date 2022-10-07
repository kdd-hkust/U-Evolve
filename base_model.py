# -*- coding: utf-8 -*-
import torch 
import torch.nn as nn
import torch.nn.functional as F


class SpGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, ret_adj=False, pa_prop=False, activate=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pa_prop = pa_prop
        self.activate = activate
        self.w_key = nn.Linear(in_features, out_features, bias=True)
        self.w_value = nn.Linear(in_features, out_features, bias=True)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.cosinesimilarity = nn.CosineSimilarity(dim=-1, eps=1e-8)
    
    def edge_attention(self, edges):
        # edge UDF
        # dot-product attention
        att_sim = torch.sum(torch.mul(edges.src['h_key'], edges.dst['h_key']),dim=-1) 
        return {'att_sim': att_sim}

    def message_func(self, edges):
        # message UDF
        return {'h_value': edges.src['h_value'], 'att_sim': edges.data['att_sim']}

    def reduce_func(self, nodes):
        # reduce UDF
        alpha = F.softmax(nodes.mailbox['att_sim'], dim=1) # (# of nodes, # of neibors)
        alpha = alpha.unsqueeze(-1)
        h_att = torch.sum(alpha * nodes.mailbox['h_value'], dim=1)
        return {'h_att': h_att}

    def forward(self, X_key, X_value, g):
        """
        :param X_key: X_key data of shape (batch_size(B), num_nodes(N), in_features_1).
        :param X_value: X_value dasta of shape (batch_size, num_nodes(N), in_features_2).
        :param g: sparse graph.
        :return: Output data of shape (batch_size, num_nodes(N), out_features).
        """
        B,N,in_features = X_key.size()
        h_key = self.w_key(X_key)  # (B,N,out_features)
        h_key = h_key.view(B*N,-1) # (B*N,out_features)
        h_value = X_value if(self.pa_prop == True) else self.w_value(X_value)
        h_value = h_value.view(B*N,-1)
        g.ndata['h_key'] = h_key
        g.ndata['h_value']= h_value
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        h_att = g.ndata.pop('h_att').view(B,N,-1) # (B,N,out_features)
        h_conv = h_att if(self.pa_prop == True or self.activate == False) else self.leakyrelu(h_att)
        return h_conv

 
class GAT(nn.Module):
    def __init__(self, in_feat, nhid=32, dropout=0, alpha=0.2, hopnum=2, pa_prop=False,activation=True):
        """sparse GAT."""
        super(GAT, self).__init__()
        self.pa_prop = pa_prop
        self.dropout = nn.Dropout(dropout)
        if(pa_prop == True): hopnum = 1 
        self.gat_stacks = nn.ModuleList()
        for i in range(hopnum):
            if(i > 0): in_feat = nhid 
            att_layer = SpGraphAttentionLayer(in_feat, nhid, dropout=dropout, alpha=alpha, pa_prop=pa_prop, activate=activation)
            self.gat_stacks.append(att_layer)

    def forward(self, X_key, X_value, adj):
        out = X_key
        for att_layer in self.gat_stacks:
            if(self.pa_prop == True):
                out = att_layer(out, X_value, adj)
            else:
                out = att_layer(out, out, adj)
        return out



