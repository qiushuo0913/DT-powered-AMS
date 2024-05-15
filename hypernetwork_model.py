'''
Author: Qiushuo Hou
Emial: qshou@zju.edu.cn
FilePath: /DT_powered_AMS_code/hypernetwork_model.py
Description: AMS mapping (MLP)
'''


import torch
import torch.nn as nn


class hyperNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(hyperNetwork, self).__init__()
        hidden_size = 32
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()
        self.fc1_1 = nn.Linear(hidden_size, hidden_size)
        # the size of the output of "nn.Linear" corresponds to the paprameters of MLPs in GNN model (see paper's setup)
        self.fc2_w1 = nn.Linear(hidden_size, 24)
        self.fc2_w2 = nn.Linear(hidden_size, 32)
        self.fc2_w3 = nn.Linear(hidden_size, 8)
        self.fc2_b1 = nn.Linear(hidden_size, 4)
        self.fc2_b2 = nn.Linear(hidden_size, 4)
        self.fc2_b3 = nn.Linear(hidden_size, 2)
        
        self.fc3_w1 = nn.Linear(24, 24)
        self.fc3_w2 = nn.Linear(32, 32)
        self.fc3_w3 = nn.Linear(8, 8)
        self.fc3_b1 = nn.Linear(4, 4)
        self.fc3_b2 = nn.Linear(4, 4)
        self.fc3_b3 = nn.Linear(2, 2)
        
        
    
    def forward(self, x):
        #normalization
        out = self.fc1((x-144/2)/(144/2))
        out = self.elu(out)
        out = self.elu(self.fc1_1(out))
        out_w1 = self.elu(self.fc2_w1(out))
        out_w2 = self.elu(self.fc2_w2(out))
        out_w3 = self.elu(self.fc2_w3(out))
        out_b1 = self.tanh(self.fc2_b1(out))
        out_b2 = self.tanh(self.fc2_b2(out))
        out_b3 = self.tanh(self.fc2_b3(out))
        
        out_w1 = self.fc3_w1(out_w1)
        out_w2 = self.fc3_w2(out_w2)
        out_w3 = self.fc3_w3(out_w3)
        out_b1 = self.fc3_b1(out_b1)
        out_b2 = self.fc3_b2(out_b2)
        out_b3 = self.fc3_b3(out_b3)
        # print('before cat', out_w1.shape, out_b3.shape)
        out = torch.cat([out_w1, out_b1, out_w2, out_b2, out_w3, out_b3], dim=0)
        # print('afater cat', out.shape)
        
        
        out = out/1.2
        return out
    
