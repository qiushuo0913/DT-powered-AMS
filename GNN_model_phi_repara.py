'''
Author: Qiushuo Hou
Emial: qshou@zju.edu.cn
FilePath: /DT_powered_AMS_code/GNN_model_phi_repara.py
Description: GNN model as the \phi in the paper
'''

import torch
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, BatchNorm1d as BN
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
    
    def forward(self, x, var, if_bias):
        idx_init = 0
        if if_bias:
            gap = 2
        else:
            gap = 1
        idx = idx_init
        while idx < len(var):

            if idx == idx_init:
                if if_bias:
                    w1, b1 = var[idx], var[idx + 1] # weight and bias
                    x = F.linear(x, w1, b1)
                    x = F.relu(x)
                    idx += 2
                else:
                    w1 = var[idx] # weight
                    x = F.linear(x, w1)
                    x = F.relu(x)
                    idx += 1
            elif idx == gap * 1+idx_init:
                if if_bias:
                    w2, b2 = var[idx], var[idx + 1]  # weight and bias
                    x = F.linear(x, w2, b2)
                    x = F.relu(x)
                    idx += 2
                else:
                    w2 = var[idx]  # weight and bias
                    x = F.linear(x, w2)
                    x = F.relu(x)
                    idx += 1 
        return x
      
class IGConv(MessagePassing):
    def __init__(self, mlp1, mlp2, train_config, **kwargs):
        super(IGConv, self).__init__(aggr='max', **kwargs)

        self.mlp1 = mlp1
        self.mlp2 = mlp2
        self.train_config = train_config
        self.reset_parameters()

    # def reset_parameters(self):
    #     reset(self.mlp1)
    #     reset(self.mlp2)
        
    def update(self, aggr_out, x, var2, if_bias):
        tmp = torch.cat([x, aggr_out], dim=1)
        comb = self.mlp2(tmp, var2, if_bias)
        nor = torch.sqrt(torch.sum(torch.mul(comb,comb),axis=1))
        nor = nor.unsqueeze(axis=-1)
        comp1 = torch.ones(comb.size(), device=self.train_config.device)
        comb = torch.div(comb,torch.max(comp1,nor) )
        return torch.cat([comb, x[:,:2*self.train_config.Nt]],dim=1)
        
    def forward(self, x, edge_index, edge_attr, var1, var2, if_bias):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, var1=var1, var2=var2, if_bias=if_bias)

    def message(self, x_i, x_j, edge_attr, var1, if_bias):
        tmp = torch.cat([x_j, edge_attr], dim=1)
        agg = self.mlp1(tmp, var1, if_bias)
        return agg

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.mlp1,self.mlp2)
class IGCNet_repara(torch.nn.Module):
    def __init__(self, train_config):
        super(IGCNet_repara, self).__init__()
        self.train_config = train_config
        # self.mlp1 = MLP([6*self.train_config.Nt, 4])
        # self.mlp2 = MLP([4+4*self.train_config.Nt, 4])
        # self.mlp2 = Seq(*[self.mlp2,Seq(Lin(4, 2*self.train_config.Nt))])
        self.mlp1 = MLP()
        self.mlp2 = MLP()
        self.conv = IGConv(self.mlp1,self.mlp2,self.train_config)

    def forward(self, data, var1, var2, if_bias):
        x0, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index

        x1 = self.conv(x = x0, edge_index = edge_index, edge_attr = edge_attr, var1 = var1, var2 = var2, if_bias=if_bias)
        x2 = self.conv(x = x1, edge_index = edge_index, edge_attr = edge_attr, var1 = var1, var2 = var2, if_bias=if_bias)
        out = self.conv(x = x2, edge_index = edge_index, edge_attr = edge_attr, var1 = var1, var2 = var2, if_bias=if_bias)
        return out
