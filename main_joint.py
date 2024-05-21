'''
Author: Qiushuo Hou
Emial: qshou@zju.edu.cn
FilePath: /DT_powered_AMS_code/main_joint.py
Description: conventional continual learning on GNN with online training --- "CL" in the simulation
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
# import wireless_network_generate as wg
from GNN_model_phi import IGCNet
from GNN_model_phi_repara import IGCNet_repara
from hypernetwork_model import hyperNetwork
import time
import utils as ut
import data_generation as dg
import loss_measure as lm
import wmmse
import torch.nn as nn
import math
import argparse

from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser(description='DTsync')
    parser.add_argument('--K', type=int, default=4, help='the number of users')
    parser.add_argument('--S', type=int, default=9, help='the number of context')
    parser.add_argument('--p', type=float, default=0.0, help='edge dropout probability')
    parser.add_argument('--T', type=int, default=2000, help='training steps')
    parser.add_argument('--Nt', type=int, default=10, help='number of PT data per context')
    parser.add_argument('--Nt_tilde', type=int, default=20, help='number of DT data per context')
    parser.add_argument('--field_length', type=int, default=100, help='area length')
    parser.add_argument('--short_length', type=int, default=20, help='d_min')
    parser.add_argument('--long_length', type=int, default=65, help='d_max')
    parser.add_argument('--seed', type=int, default=7, help='simulation seed')
    parser.add_argument('--tesing_number', type=int, default=100, help='number of testing samples per context')
    parser.add_argument('--Epoch', type=int, default=1, help='online training times in each step')
    parser.add_argument('--WMMSE_times', type=int, default=5, help='the times of running WMMSE algorithm')
    parser.add_argument('--if_bias', type=int, default=1, help='GNN has bias')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate of AMS')
    # parser.add_argument('--count', type=int, default=1, help='')
    # parser.add_argument('--count_lambda', type=int, default=3, help='')
    parser.add_argument('--R', type=int, default=8, help='update frequency')
    # parser.add_argument('--Q', type=int, default=100, help='')
    parser.add_argument('--window_length', type=int, default=20, help='window size')
    # parser.add_argument('--alpha', type=float, default=1, help='')
    # parser.add_argument('--lambda_2', type=float, default=0.5, help='')
    parser.add_argument('--K_factor_PT', type=int, default=0, help='K factor in PT')
    parser.add_argument('--K_factor_DT', type=int, default=0, help='K factor in DT')
    # parser.add_argument('--mode', type=str, default='onlyPT', help='')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()
    print('Called with args:')
    print(args)


    writer = SummaryWriter()

    

    scenario_config = ut.init_parameters(args)
    # GNN model
    model = IGCNet(scenario_config).to(scenario_config.device)
    state_dict = model.state_dict()
    
    
    # hyper-parameter network
    input_size = scenario_config.train_K*scenario_config.train_K
    output_size = state_dict['conv.mlp1.0.0.weight'].numel()+state_dict['conv.mlp1.0.0.bias'].numel()+state_dict['conv.mlp2.0.0.0.weight'].numel()+state_dict['conv.mlp2.0.0.0.bias'].numel()+state_dict['conv.mlp2.1.0.weight'].numel()+state_dict['conv.mlp2.1.0.bias'].numel()
    
    hidden_size1 = 128
    hidden_size2 = 256
    # initialization of hyperparameter network
    eta_network = hyperNetwork(input_size, hidden_size1, hidden_size2, output_size).type(torch.float64)
    
    #Ensure the random seeds are the same
    def reset_randomness(random_seed):
        torch.manual_seed(random_seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)


    # WMMSE
    def batch_wmmse(csis, Pini):
        
        Nt = scenario_config.Nt
        K = scenario_config.n_receiver
        n = csis.shape[0]
        Y = np.zeros( (n,K,Nt),dtype=complex)
        # Pini = 1/np.sqrt(Nt)*np.ones((K,Nt),dtype=complex)*4
        for ii in range(n):
            Y[ii,:,:] = wmmse.np_WMMSE_vector(np.copy(Pini), csis[ii,:,:,:], 1, scenario_config.var)
        return Y


    # all P_max
    def batch_benchmark(csis, Pini):
        
        Nt = scenario_config.Nt
        K = scenario_config.n_receiver
        n = csis.shape[0]
        Y = np.zeros((n,K,Nt),dtype=complex)

        for ii in range(n):
            Y[ii,:,:] = Pini
        return Y


   
    # train GNN
    def train_GNN(t, model, data_list, num_data, Epoch):
        # model - initial GNN
        # data_list - PT graph data ([\hat{N}_t])
        # num_data - number of data in one t (\hat{N}_t) 
        
        model.train()
        Total_loss = []
        
        data_loader = DataLoader(data_list, batch_size=num_data, 
                                    shuffle=False, num_workers=1)
        
        # optimizer_test = torch.optim.SGD(model.parameters(), lr=args.lr)
        
        steps1 = 50
        steps2 = 100
        steps3 = 150
        steps4 = 200
        steps5 = 250
        if (t >= steps1 and t < steps2):
            optimizer_test = torch.optim.SGD(model.parameters(), lr=args.lr/2, weight_decay=0.01)
        elif (t >= steps2 and t < steps3):
            optimizer_test = torch.optim.SGD(model.parameters(), lr=args.lr/4, weight_decay=0.01)
        elif (t >= steps3 and t < steps4):
            optimizer_test = torch.optim.SGD(model.parameters(), lr=args.lr/8, weight_decay=0.01)  
        elif (t >= steps4 and t < steps5):
            optimizer_test = torch.optim.SGD(model.parameters(), lr=args.lr/16, weight_decay=0.01)
        elif (t >= steps5):
            optimizer_test = torch.optim.SGD(model.parameters(), lr=args.lr/32, weight_decay=0.01)   
        else:
           optimizer_test = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.01)
        
        for epoch in range(Epoch):
            total_loss = 0
            total_loss_train = torch.zeros(1).to(scenario_config.device)
            for data in data_loader:
                data = data.to(scenario_config.device)
                # optimizer_test.zero_grad()
                out = model(data)
                loss = ut.sr_loss(data,out,scenario_config.train_K, scenario_config.Nt, scenario_config.var, scenario_config)
                # loss.backward()
                total_loss_train += loss * data.num_graphs/len(data_loader)
                total_loss += loss.item() * data.num_graphs
                # optimizer_test.step()
                
            total_loss_train = total_loss_train/num_data
            optimizer_test.zero_grad()
            total_loss_train.backward()
            optimizer_test.step()
            
            total_loss = -total_loss/num_data
            Total_loss.append(total_loss)
            # print('finish!')
        
        # plt.plot(Total_loss)
        # plt.show()
        return model
    
    
    
    #test GNN
    def test_GNN(model, data_list, csis, num_data, t):
        # model - initial GNN
        # data_list - PT graph data ([\hat{N}_t])
        # csis - PT data true world data for calculating the WMMSE ([\hat{N}_t])
        # num_data - number of data in one t (\hat{N}_t) 
        # t - index of time step, WMMSE can be only computed for ones
        
        
        #all Pmax
        Sr = []
        for i in range(1):
            # pamx as the initialization
            Pini = 1/np.sqrt(scenario_config.Nt)*np.ones((scenario_config.train_K,scenario_config.Nt),dtype=complex)
            Y = batch_benchmark(csis.transpose(0,2,1,3), Pini)
            sr = wmmse.IC_sum_rate(csis,Y,scenario_config.var)
            Sr.append(sr)
        sr_max = max(Sr)
        print('all Pmax rate:',sr_max)
        
        loss_test = lm.loss_measure(scenario_config, model, data_list, num_data)
        normization_rate = -loss_test/sr_max
        ## if you wants to get the value of WMMSE, you can run this part.
        # if t == 0:
        #     wmmse_rate = sr_max_true/sr_max
        # else:
        #     wmmse_rate = 1
        ## fake value, just for consistency
        wmmse_rate = 1
        return normization_rate, wmmse_rate
    
        
    def overall_test_GNN(model, data_list, csis, data_list_test, csis_test, num_data_train_PT, num_data_test):
        Sumrate_average = []
       
        
        for t in range (scenario_config.T):
            # multiple update R
            data_list_PT_S, _, _ = dg.generate_S_dataDT_multi_update(scenario_config)

            
            data_list_PT_t = []
            for step in range (args.R):
                data_list_PT_t += data_list_PT_S[step*num_data_train_PT:(step+1)*num_data_train_PT]
           
            Sum = []
            # WMMSE_sum = []
            for i in range(args.tesing_number):
                csis_test_i = csis_test[i,:,:,:,:]
                data_list_test_i = data_list_test[i*num_data_test:(i+1)*num_data_test]
                sumrate, _ = test_GNN(model, data_list_test_i, csis_test_i, num_data_test, t)
                Sum.append(sumrate)
                # WMMSE_sum.append(wmmse_rate)
            Sumrate_average.append(np.average(Sum))
            
            # if t == 0:
            #     WMMSE_average.append(np.average(WMMSE_sum))
            
            #     print('average WMMSE', WMMSE_average)
           
                

            # tags1 = {'NtPT_%d_seed%d_joint_Kfactor%s' % (scenario_config.train_layouts_PT, args.seed, args.K_factor_PT): np.average(Sum)}
            # writer.add_scalars('Epoch=50', tags1, t)
            

            
            model = train_GNN(t, model, data_list_PT_t, num_data_train_PT, Epoch = args.Epoch)

        
    def joint_test_GNN():
        #Generate a fixed testing dataset with 100 different context variables, and each context variable is generated with 50 different CSIs.
        #reset_randomness(args.seed)
        reset_randomness(7)
        scenario_config.T = args.tesing_number
        scenario_config.train_layouts_PT = 50
        scenario_config.p = 0.0
        scenario_config.S = 9
        CSIs_test, data_list_test, _, _, _ = dg.generate_T_dataPTandDT(scenario_config)
        
        
        #recover the online training setting
        scenario_config.T = args.T
        scenario_config.train_layouts_PT = args.Nt
        scenario_config.p = args.p
        scenario_config.S = args.S
        
        #generate training data for GNN
        
        CSIs_PT, data_list_PT, _, _, _ = dg.generate_T_dataPTandDT(scenario_config)
        
        
        
        model = IGCNet(scenario_config).to(scenario_config.device)
       
        reset_randomness(args.seed)
        overall_test_GNN(model, data_list_PT, CSIs_PT, data_list_test, CSIs_test, num_data_train_PT=args.Nt, num_data_test=50)
        
       
        
    
    joint_test_GNN()
     
