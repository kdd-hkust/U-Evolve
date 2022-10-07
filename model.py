# -*- coding: utf-8 -*-
import math
import torch 
import torch.nn as nn
import torch.nn.functional as F
from base_model import GAT

class FeatureEmb(nn.Module):
    def __init__(self, f_in, f_out=6):
        super(FeatureEmb, self).__init__()
        self.emb_layer = nn.Embedding(f_in, f_out)
        nn.init.xavier_uniform_(self.emb_layer.weight.data, gain=math.sqrt(2.0))
    def forward(self, X):
        X = self.emb_layer(X.long()) 
        return X

class GLBFM(nn.Module):
    '''
    dynamic graph + GRU
    '''
    def __init__(self,args,t_in,t_out,nfeat,nstatic,task,dropout=0.5,alpha=0.2,hid_dim=32,\
                 gat_hop=1,device=torch.device('cuda')):
        super(GLBFM, self).__init__()
        self.device = device
        self.nfeat = nfeat + 3
        self.hid_dim = hid_dim
        self.sta_dim = nstatic + 3
        self.t_out = t_out
        self.out_dim = 3 * self.t_out
        self.time_emb = FeatureEmb(f_in=12, f_out=4)
        # FC layers
        self.output_fc = nn.Linear(hid_dim + self.sta_dim + 4*self.t_out, self.out_dim, bias=True)
        self.leakyrelu = nn.LeakyReLU(alpha)
        # GAT block
        self.nhid = hid_dim//2 
        self.MobilityConv = GAT(in_feat=self.nfeat, nhid=self.nhid, dropout=dropout, alpha=alpha, hopnum=gat_hop, pa_prop=False)
        self.SemanticConv = GAT(in_feat=self.nfeat, nhid=self.nhid, dropout=dropout, alpha=alpha, hopnum=gat_hop, pa_prop=False)
        self.DistanceConv = GAT(in_feat=self.nfeat, nhid=self.nhid, dropout=dropout, alpha=alpha, hopnum=gat_hop, pa_prop=False)
        
        # Layer before GRU
        self.dfea_fc = nn.Linear(3*self.nhid+self.nfeat, hid_dim, bias=True)
        # GRU Cell
        self.GRU = nn.GRUCell(hid_dim, hid_dim, bias=True)
        nn.init.xavier_uniform_(self.GRU.weight_ih,gain=math.sqrt(2.0))
        nn.init.xavier_uniform_(self.GRU.weight_hh,gain=math.sqrt(2.0))
        # 
        self.direction_emb = FeatureEmb(f_in=4, f_out=4)

        # Parameter initialization
        for ele in self.modules():
            if isinstance(ele, nn.Linear):
                nn.init.xavier_uniform_(ele.weight,gain=math.sqrt(2.0))
        self.dropout_layer = nn.Dropout(dropout)

    def transform_X_emb(self, X):

        X_T = self.time_emb(X)
        return X_T
    
    def transform_X_static(self, X):
        X_direction = self.direction_emb(X[:,:,-1])
        X_static = torch.cat([X[:,:,:-1], X_direction], dim=-1)
        return X_static

    def forward(self, X, X_static, X_T, G_mobility,G_semantic,G_distance):

        X_T = self.transform_X_emb(X_T)
        X_dynamic_month = self.transform_X_emb(X[:,:,:,-1])
        X_dynamic = X[:,:,:,:-1]
        X_dfeat =  torch.cat([X_dynamic, X_dynamic_month], dim=-1)
        B, N, T, F_feat = X_dfeat.size()
        X_static = X_static.unsqueeze(0).repeat(B,1,1) # (B, N, F_s)
        h_t = torch.zeros(B*N, self.hid_dim).to(device=self.device) # (B*N, tmp_hid)

        for i in range(T):
            adj_mobility = G_mobility[i]
            adj_semantic = G_semantic[i]
            h_mobility = self.dropout_layer(self.MobilityConv(X_dfeat[:,:,i,:],None,adj_mobility)) 
            h_semantic = self.dropout_layer(self.SemanticConv(X_dfeat[:,:,i,:],None,adj_semantic)) 
            h_dst = self.dropout_layer(self.DistanceConv(X_dfeat[:,:,i,:],None,G_distance))
            gru_in = torch.cat([X_dfeat[:,:,i,:], h_mobility, h_semantic, h_dst], dim=-1).view(-1,3*self.nhid+self.nfeat)
            gru_in = self.dfea_fc(gru_in)
            h_t = self.GRU(gru_in, h_t) 

        h_t = h_t.view(B,N,-1)
        X_static = self.transform_X_static(X_static)
        X_T = X_T.view(B, N, -1)
        h_t = torch.cat([h_t, X_static, X_T], dim=-1)
        out = self.output_fc(h_t)
        out = out.view(B, N, self.t_out, 3) 
        return out

