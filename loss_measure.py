'''
Author: Qiushuo Hou
Emial: qshou@zju.edu.cn
FilePath: /DT_powered_AMS_code/loss_measure.py
Description: 
'''

#%%
import scipy.io as sio                     # import scipy.io for .mat file I/O 
import numpy as np                         # import numpy
import matplotlib.pyplot as plt            # import matplotlib.pyplot for figure plotting
import torch
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, BatchNorm1d as BN
import wireless_network_generate as wg
# from GNN_model_phi import IGCNet
import time
import utils as ut
from data_generation import *
import torch.nn as nn



# measure loss
def loss_measure_repara(scenario_config, model, data_list, num_data, var1, var2, if_bias):
    model.eval()

    total_loss = 0
    data_loader = DataLoader(data_list, batch_size=num_data, 
                                shuffle=False, num_workers=1)
   
        
    with torch.no_grad():
        # start = time.time()
        for data in data_loader:
            data = data.to(scenario_config.device)
            out = model(data, var1, var2, if_bias)
            # end = time.time()
            # print('testing time:', end-start)
            loss = ut.sr_loss(data,out,scenario_config.train_K, scenario_config.Nt, scenario_config.var, scenario_config)
            total_loss += loss.item() * data.num_graphs
           
    
    return total_loss / num_data

# loss measure for the directly training the GNN
def loss_measure(scenario_config, model, data_list, num_data):
    model.eval()

    total_loss = 0
    data_loader = DataLoader(data_list, batch_size=num_data, 
                                shuffle=False, num_workers=1)

        
    with torch.no_grad():
        # start = time.time()
        for data in data_loader:
            data = data.to(scenario_config.device)
            out = model(data)
            # end = time.time()
            # print('testing time:', end-start)
            loss = ut.sr_loss(data,out,scenario_config.train_K, scenario_config.Nt, scenario_config.var, scenario_config)
            total_loss += loss.item() * data.num_graphs
            #power = out[:,:2*Nt]
            #Y = power.numpy()
            #power_check(Y)
    
    return total_loss / num_data

# put weight and bias to the model
def give_model_parameter(phi, model):
    # model = IGCNet(scenario_config).to(scenario_config.device)
    state_dict = model.state_dict()
    # Extract the weights and bias parameters of the network
    w11 = state_dict['conv.mlp1.0.0.weight']
    b11 = state_dict['conv.mlp1.0.0.bias']
    
    w21 = state_dict['conv.mlp2.0.0.0.weight']
    b21 = state_dict['conv.mlp2.0.0.0.bias']
    
    w22 = state_dict['conv.mlp2.1.0.weight']
    b22 = state_dict['conv.mlp2.1.0.bias']
    
    index1 = w11.shape[0]*w11.shape[1]
    index2 = b11.shape[0]
    index3 = w21.shape[0]*w21.shape[1]
    index4 = b21.shape[0]
    index5 = w22.shape[0]*w22.shape[1]
    index6 = b22.shape[0]
    
    state_dict['conv.mlp1.0.0.weight'] = phi[0:index1].reshape(w11.shape)
    state_dict['conv.mlp1.0.0.bias'] = phi[index1:index1+index2].reshape(b11.shape)
    
    state_dict['conv.mlp2.0.0.0.weight'] = phi[index1+index2:index1+index2+index3].reshape(w21.shape)
    state_dict['conv.mlp2.0.0.0.bias'] = phi[index1+index2+index3:index1+index2+index3+index4].reshape(b21.shape)
    
    state_dict['conv.mlp2.1.0.weight'] = phi[index1+index2+index3+index4:index1+index2+index3+index4+index5].reshape(w22.shape)
    state_dict['conv.mlp2.1.0.bias'] = phi[index1+index2+index3+index4+index5:index1+index2+index3+index4+index5+index6].reshape(b22.shape)
    
    model.load_state_dict(state_dict)
    return model

# obtain the weight and bias from model (i.e., prepare for the AMS output as the parameters of GNN)
def get_var(phi, model):
    # phi - the output of \eta
    # model - GNN without rewriting repara
    state_dict = model.state_dict()
    
    # Change the accuracy of \phi so that it can adapt to the accuracy of parameters of the GNN
    phi = phi.type(torch.float32)
    
    # Extract the weights and bias parameters of the original network
    w11 = state_dict['conv.mlp1.0.0.weight']
    b11 = state_dict['conv.mlp1.0.0.bias']
    
    w21 = state_dict['conv.mlp2.0.0.0.weight']
    b21 = state_dict['conv.mlp2.0.0.0.bias']
    
    w22 = state_dict['conv.mlp2.1.0.weight']
    b22 = state_dict['conv.mlp2.1.0.bias']
    
    index1 = w11.shape[0]*w11.shape[1]
    index2 = b11.shape[0]
    index3 = w21.shape[0]*w21.shape[1]
    index4 = b21.shape[0]
    index5 = w22.shape[0]*w22.shape[1]
    index6 = b22.shape[0]
    
    var1 = [phi[0:index1].reshape(w11.shape), phi[index1:index1+index2].reshape(b11.shape)]
    var2 = [phi[index1+index2:index1+index2+index3].reshape(w21.shape), phi[index1+index2+index3:index1+index2+index3+index4].reshape(b21.shape),
            phi[index1+index2+index3+index4:index1+index2+index3+index4+index5].reshape(w22.shape),
            phi[index1+index2+index3+index4+index5:index1+index2+index3+index4+index5+index6].reshape(b22.shape)]
    
    # var1, var2 are the parameters of mlp1 and mlp2 in GNN
    return var1, var2
    
    
    
