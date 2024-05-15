'''
Author: Qiushuo Hou
Emial: qshou@zju.edu.cn
FilePath: /DT_powered_AMS_code/GNN_model_phi.py
Description: obtain the size from the GNN model used for power control
'''
import torch
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, BatchNorm1d as BN


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU())#, BN(channels[i])
        for i in range(1, len(channels))
    ])  
      
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
        
    def update(self, aggr_out, x):
        tmp = torch.cat([x, aggr_out], dim=1)
        comb = self.mlp2(tmp)
        nor = torch.sqrt(torch.sum(torch.mul(comb,comb),axis=1))
        nor = nor.unsqueeze(axis=-1)
        comp1 = torch.ones(comb.size(), device=self.train_config.device)
        comb = torch.div(comb,torch.max(comp1,nor) )
        return torch.cat([comb, x[:,:2*self.train_config.Nt]],dim=1)
        
    def forward(self, x, edge_index, edge_attr):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        tmp = torch.cat([x_j, edge_attr], dim=1)
        agg = self.mlp1(tmp)
        return agg

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.mlp1,self.mlp2)
class IGCNet(torch.nn.Module):
    def __init__(self, train_config):
        super(IGCNet, self).__init__()
        self.train_config = train_config
        self.mlp1 = MLP([6*self.train_config.Nt, 4])
        self.mlp2 = MLP([4+4*self.train_config.Nt, 4])
        self.mlp2 = Seq(*[self.mlp2,Seq(Lin(4, 2*self.train_config.Nt))])
        self.conv = IGConv(self.mlp1,self.mlp2,self.train_config)

    def forward(self, data):
        x0, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        x1 = self.conv(x = x0, edge_index = edge_index, edge_attr = edge_attr)
        x2 = self.conv(x = x1, edge_index = edge_index, edge_attr = edge_attr)
        out = self.conv(x = x2, edge_index = edge_index, edge_attr = edge_attr)
        return out
