# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
import dgl

HOME = 'File PATH'  




class GetDynamicDataset(Dataset):
    def __init__(self, X, M, Y):

        self.X = X   
        self.M = M
        self.Y = Y   
        self.idx = range(len(self.X))

    def __getitem__(self, index):
        
        # torch.Tensor
        tensor_X = self.X[index]
        tensor_M = self.M[index]
        tensor_Y = self.Y[index]
        list_idx = self.idx[index]
        
        return tensor_X, tensor_M, tensor_Y, list_idx

    def __len__(self):
        return len(self.X)

def make_dataset(rawdata, label, X_time, T_in, T_out, part_num):
    X = [] 
    Y = []
    M = []
    
    for i in range(0, part_num): 
        X.append(rawdata[:,i:i+T_in, :])
        Y.append(label[:,i:i+T_out])
        M.append(X_time[:,i:i+T_out])

    X = torch.from_numpy(np.asarray(X)).float()
    M = torch.from_numpy(np.asarray(M)).float().squeeze()
    Y = torch.from_numpy(np.asarray(Y)).float().squeeze()

    F_dim = X.size()[-1]
    return GetDynamicDataset(X, M, Y), F_dim

def adj_process(adj, device, topk=5, reversed=True, with_pair = False):

    edge_1 = []
    edge_2 = []
    pairs = []
    for i in range(adj.shape[0]):
        adj_row = adj[i]
        adj_row = sorted(enumerate(adj_row), key=lambda x:x[1], reverse=reversed)
        cnt = 0
        for j,sim in adj_row:
            if i != j and cnt < topk:
                edge_1.append(i)
                edge_2.append(j)
                pairs.append((i, j, sim))
                cnt += 1


    edge_adj = np.asarray([edge_1,edge_2],dtype=int)
    edge_adj = torch.from_numpy(edge_adj).long()

    if not with_pair:
        return edge_adj
    else:
        return edge_adj, pairs

def edge_to_g(edge_adj,N):
    g_adj = dgl.DGLGraph()
    g_adj.add_nodes(N) 
    g_adj.add_edges(edge_adj[0],edge_adj[1])
    return g_adj

def expand_static_adj(adj, N, batch_size):
    '''
    input: edge list of static graph (2, E)
    output: the graph in a larger batch length
    '''
    g_adj = dgl.DGLGraph()
    g_adj.add_nodes(N) 
    g_adj.add_edges(adj[0],adj[1])
    return [g_adj]*batch_size


def make_graph_dataset(device, G_mobility_raw, G_semantic_raw, mobility_top, semantic_top, part_num, T_in):
    G_mobility, G_semantic = [], []
    N = G_mobility_raw.shape[1]
    for i in range(part_num):
        tmp_G = [[],[]]
        for idx in range(T_in):
            tmp_pre_adj_mobility = G_mobility_raw[i+idx]
            tmp_adj_mobility = adj_process(tmp_pre_adj_mobility, device,mobility_top)
            tmp_pre_adj_semantic = G_semantic_raw[i+idx]
            tmp_adj_semantic = adj_process(tmp_pre_adj_semantic, device,semantic_top, reversed=False)
            
            tmp_g_mobility = edge_to_g(tmp_adj_mobility,N)
            tmp_G[0].append(tmp_g_mobility)
            tmp_g_semantic = edge_to_g(tmp_adj_semantic,N)
            tmp_G[1].append(tmp_g_semantic)
        G_mobility.append(tmp_G[0])
        G_semantic.append(tmp_G[1])
    
    return  G_mobility, G_semantic



def load_attribute(dataset):
    '''
    load the basic temporal feature; static feature; 
    '''
    X_dynamic = np.load(HOME + 'data/{}-dynamic-feat.npy'.format(dataset)) 
    X_static  = np.load(HOME + 'data/{}-static-feat.npy'.format(dataset))  
    X_time = np.load(HOME + 'data/{}-time-feat.npy'.format(dataset)) 
    X_distance = np.load(HOME + 'data/{}-distance-feat.npy'.format(dataset)) 

    return X_dynamic, X_static, X_time, X_distance


def load_graph(dataset):
    '''
    Three type of graphs: distance graph (N, N); semantic graph (T, N, N); and mobility graph (T, N, N)
    '''
    # make it as the placeholder 
    T, N = 23, 260
    X_distance = np.random.random((N, N))
    X_semantic = np.random.random((T, N, N))
    X_mobility = np.random.random((T, N, N))

    return X_distance, X_semantic, X_mobility




def load_basic_data(dataset, T_in, T_out, Batch_Size, val_num, test_num):

    X_dynamic, X_static, X_time, X_distance = load_attribute(dataset)
    Label = np.load(HOME + 'data/{}-label.npy'.format(dataset))

    X_static = np.concatenate([X_static, X_distance], axis=1)
    F_static = X_static.shape[1]

    X_static = torch.from_numpy(np.asarray(X_static)).float()

    N, T_all = Label.shape

    section_num = T_all - T_in - T_out + 2

    train_num = section_num - val_num - test_num
    val_split = train_num + val_num


    dataset_train, F_dim = make_dataset(X_dynamic[:, :T_in+train_num-1],
                                    Label[:,T_in-1 : T_in-1+train_num+T_out-1],
                                    X_time[:,T_in-1 : T_in-1+train_num+T_out-1],
                                    T_in,T_out,train_num)

    dataset_val, _ = make_dataset(X_dynamic[:,train_num : T_in+val_split-1],
                                Label[:,T_in-1+train_num : T_in-1+val_split+T_out-1],
                                X_time[:,T_in-1+train_num : T_in-1+val_split+T_out-1],
                                T_in,T_out,val_num)

    dataset_test, _ = make_dataset(X_dynamic[:,val_split:],
                                Label[:,T_in-1+val_split:],
                                X_time[:,T_in-1+val_split:],
                                T_in,T_out,test_num)
    loader_train = DataLoader(dataset=dataset_train, batch_size=Batch_Size, shuffle=True, pin_memory=True,num_workers=1)
    loader_val = DataLoader(dataset=dataset_val, batch_size=Batch_Size, shuffle=False, pin_memory=True,num_workers=1)
    loader_test = DataLoader(dataset=dataset_test, batch_size=Batch_Size, shuffle=False, pin_memory=True,num_workers=1)

    return loader_train,loader_val,loader_test, X_static, F_dim, F_static, N

def load_graph_data(dataset, T_in, T_out, Batch_Size, device, distance_top=5, mobility_top=5, semantic_top=5):

    # load graph
    X_distance, X_semantic, X_mobility = load_graph(dataset)
    T_all, N, N = X_semantic.shape

    val_num, test_num = 4, 5
    # load basic data
    loader_train,loader_val,loader_test, X_static, F_dim, F_static, N = load_basic_data(dataset, T_in, T_out, Batch_Size, val_num, test_num)

    # make dgl graph
    adj_distance = adj_process(X_distance, device, topk=distance_top, reversed=False) 
    dgl_adj_distance = expand_static_adj(adj_distance, N, Batch_Size) 

    section_num = T_all - T_in - T_out + 2

    train_num = section_num - val_num - test_num
    val_split = train_num + val_num

    G_mobility_train, G_semantic_train = make_graph_dataset(device,X_mobility[:T_in+train_num-1],X_semantic[:T_in+train_num-1],
                                                        mobility_top,semantic_top, train_num, T_in)
    G_mobility_val, G_semantic_val = make_graph_dataset(device,X_mobility[train_num : T_in+val_split-1], 
                                                    X_semantic[train_num : T_in+val_split-1],
                                                    mobility_top,semantic_top,val_num, T_in)
    G_mobility_test, G_semantic_test = make_graph_dataset(device,X_mobility[val_split:], X_semantic[val_split:],
                                                        mobility_top,semantic_top,test_num, T_in)

    return loader_train,loader_val,loader_test, X_static, F_dim, F_static, N, dgl_adj_distance, G_mobility_train, G_semantic_train, G_mobility_val, G_semantic_val, G_mobility_test, G_semantic_test





