'''
Author: Qiushuo Hou
Emial: qshou@zju.edu.cn
FilePath: /DT_powered_AMS_code/main_repara_naive.py
Description: naive AMS (using PT data on par with DT data)--- "N-DT-AMS" in the simulation
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
import torch.optim as optim

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
    parser.add_argument('--R', type=int, default=8, help='update frequency')
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
    
   
    model = IGCNet(scenario_config).to(scenario_config.device)
    model_repara = IGCNet_repara(scenario_config).to(scenario_config.device)
    
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

    
    
    ### naive-eta

    def train_eta_PPI(t, T, model, eta, data_list, context_vector_t, data_list_PT_ct, data_list_DT_ct, num_data_PT, num_data_DT, Epoch):
    # def train_eta_PPI(model, eta, data_list, context_vector_t, data_list_PT_ct, data_list_DT_ct, num_data_PT, num_data_DT, Epoch):
        # model - GNN with rewrite repara
        # eta - hyperparameter network
        # data_list - DT data in time step t with S number of different c_s
        # contxt_vector_t - context in time step t with S number of different c_s
        # data_list_PT_ct - PT graph data at time step t
        # data_list_DT_ct - DT graph data at time step t
        # num_data - number of data in one t (DT side)-for the traing of c_s
        # Epoch - number of training times
        
        #It only extracts the parameter size information of GNN model \phi, 
        # and does not use this model for any forward propagation or backward propagation.
        model_without_repara = IGCNet(scenario_config).to(scenario_config.device)
        
        model.train()
        eta.train()
        Total_loss = []
        
        data_loader = DataLoader(data_list, batch_size=num_data_DT, 
                                    shuffle=False, num_workers=1)
        # multi-update
        data_loader_PT = DataLoader(data_list_PT_ct, batch_size=num_data_PT, 
                                    shuffle=False, num_workers=1)
        data_loader_DT = DataLoader(data_list_DT_ct, batch_size=num_data_DT, 
                                    shuffle=False, num_workers=1)
        
        
        steps1 = 50
        steps2 = 100
        steps3 = 150
        steps4 = 200
        steps5 = 250
        if (t >= steps1 and t < steps2):
            optimizer_eta = torch.optim.SGD(eta.parameters(), lr=args.lr/2, weight_decay=0.01)
        elif (t >= steps2 and t < steps3):
            optimizer_eta = torch.optim.SGD(eta.parameters(), lr=args.lr/4, weight_decay=0.01)
        elif (t >= steps3 and t < steps4):
            optimizer_eta = torch.optim.SGD(eta.parameters(), lr=args.lr/8, weight_decay=0.01)
        elif (t >= steps4 and t < steps5):
            optimizer_eta = torch.optim.SGD(eta.parameters(), lr=args.lr/16, weight_decay=0.01)
        elif (t >= steps5):
            optimizer_eta = torch.optim.SGD(eta.parameters(), lr=args.lr/32, weight_decay=0.01)   
        else:
           optimizer_eta = torch.optim.SGD(eta.parameters(), lr=args.lr, weight_decay=0.01)
        # optimizer_eta = torch.optim.SGD(eta.parameters(), lr=args.lr, weight_decay=0.01)
        
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_eta, steps)
        for epoch in range(Epoch):
            # total_loss = 0
            total_loss_train = torch.zeros(1).to(scenario_config.device)
            for (index, data) in enumerate(data_loader):
                data = data.to(scenario_config.device)
                
                context = context_vector_t[index+args.R]
                context = torch.from_numpy(context)
                mean_phi = eta(context)
                
                noise = 0
                phi = mean_phi + noise
                
                var1, var2 = lm.get_var(phi, model_without_repara)
                out = model(data, var1, var2, if_bias=args.if_bias)
                loss = ut.sr_loss(data,out,scenario_config.train_K, scenario_config.Nt, scenario_config.var, scenario_config)
                
                total_loss_train += loss/len(data_loader)
                
            # for data_PT in data_loader_PT:
            # multi-update
            loss_PT = torch.zeros(1).to(scenario_config.device)
            for (index, data_PT) in enumerate(data_loader_PT):
                data_PT = data_PT.to(scenario_config.device)
                
                context = context_vector_t[index]
                context = torch.from_numpy(context)
                mean_phi = eta(context)
                
                noise = 0
                phi = mean_phi + noise
                
                var1, var2 = lm.get_var(phi, model_without_repara)
                out_PT = model(data_PT, var1, var2, if_bias=args.if_bias)
                #multi-update
                loss_PT += ut.sr_loss(data_PT,out_PT,scenario_config.train_K, scenario_config.Nt, scenario_config.var, scenario_config)/len(data_loader_PT)
            
            # multi-update
            loss_DT = torch.zeros(1).to(scenario_config.device)   
            for (index, data_DT) in enumerate(data_loader_DT):
                data_DT = data_DT.to(scenario_config.device)
                
                context = context_vector_t[index]
                context = torch.from_numpy(context)
                mean_phi = eta(context)
                
                noise = 0
                phi = mean_phi + noise
                
                var1, var2 = lm.get_var(phi, model_without_repara)
                out_DT = model(data_DT, var1, var2, if_bias=args.if_bias)
                #multi-update
                loss_DT += ut.sr_loss(data_DT,out_DT,scenario_config.train_K, scenario_config.Nt, scenario_config.var, scenario_config)/len(data_loader_DT)
            
            ratio_PT = args.Nt/(args.Nt+args.Nt_tilde*(args.S/args.R-1))
            ratio_DT = args.Nt_tilde*(args.S/args.R-1)/(args.Nt+args.Nt_tilde*(args.S/args.R-1))
            total_loss_train = ratio_DT*total_loss_train+ratio_PT*loss_PT
            
            optimizer_eta.zero_grad() 
            total_loss_train.backward()
            optimizer_eta.step()
            # scheduler.step()
            # if (t >0) and (t == steps):
            #     optimizer_eta = torch.optim.SGD(eta.parameters(), lr=args.lr/2, weight_decay=0.01)
            
            # print('finish!')    
        return eta
    
    
    #test eta
    def test_eta(model, eta, context_vector_t, data_list, csis, num_data, t):
        # model - GNN with rewrite repara
        # eta - hyperparameter network
        # context_vector_t - the current context vector as the input of eta
        # data_list - testing data sample with size=num_data
        # csis - CSIs of testing data
        # num_data - number of data in one t for testing
        # t - index of time step, WMMSE can be only computed for ones
        
        #It only extracts the parameter size information of GNN model \phi, 
        # and does not use this model for any forward propagation or backward propagation.
        model_without_repara = IGCNet(scenario_config).to(scenario_config.device)
        
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
        
        
        eta.eval()
        context_vector_t = torch.from_numpy(context_vector_t)
        mean_phi = eta(context_vector_t)
        noise = 0 
        
        phi = mean_phi + noise
        var1, var2 = lm.get_var(phi, model_without_repara)
        
        loss_test = lm.loss_measure_repara(scenario_config, model, data_list, num_data, var1, var2, if_bias=args.if_bias)
        normization_rate = -loss_test/sr_max
        # if t == 0:
        #     wmmse_rate = sr_max_true/sr_max
        # else:
        #     wmmse_rate = 1
        #testing
        # ## fake value, just for consistency
        wmmse_rate = 1
        return normization_rate, wmmse_rate
    
    
       
    ### naive (using both PT data and DT data)
    def overall_test_eta_PPI(model, eta, data_list_test, context_vector_test, csis_test, num_data_train_DT, num_data_train_PT, num_data_test):
        Sumrate_average = []
        Sumrate_best = []
            
        for t in range (scenario_config.T):
        
            data_list_PT_S, data_list_DT_S, context_vector_list_DT_S = dg.generate_S_dataDT_multi_update(scenario_config)
            
            context_vector_t = context_vector_list_DT_S
            
            
            data_list_DT_t = []
            data_list_PT_t = []
            for step in range (args.R):
                data_list_DT_t += data_list_DT_S[step*num_data_train_DT:(step+1)*num_data_train_DT]
                data_list_PT_t += data_list_PT_S[step*num_data_train_PT:(step+1)*num_data_train_PT]
            
    
            data_list_DT_S_multi_update =  data_list_DT_S[args.R*num_data_train_DT:]
            
            
            Sum = []
            for i in range(args.tesing_number):
                csis_test_i = csis_test[i,:,:,:,:]
                data_list_test_i = data_list_test[i*num_data_test:(i+1)*num_data_test]
                context_vector_test_i = context_vector_test[i]
                
                sumrate, _ = test_eta(model, eta, context_vector_test_i, data_list_test_i, csis_test_i, num_data_test, t)
                Sum.append(sumrate)
            Sumrate_average.append(np.average(Sum))
            Sumrate_best.append(np.max(Sum))
            
            # tags1 = {'NtDT_%d_naive_p%f_seed%d_S%d' % (scenario_config.train_layouts_DT, args.p, args.seed, args.S): np.average(Sum)}
            # writer.add_scalars('Epoch=50_repara', tags1, t)
            
           
            eta = train_eta_PPI(t, scenario_config.T, model, eta, data_list_DT_S_multi_update, context_vector_t, data_list_PT_t, data_list_DT_t, num_data_train_PT, num_data_train_DT, Epoch=args.Epoch)
            # eta = train_eta_PPI(model, eta, data_list_DT_S_multi_update, context_vector_t, data_list_PT_t, data_list_DT_t, num_data_train_PT, num_data_train_DT, Epoch=args.Epoch)
       
        
        
    def joint_test_GNN():
        #Generate a fixed testing dataset with 100 different context variables, and each context variable is generated with 50 different CSIs.
        #reset_randomness(args.seed)
        reset_randomness(7)
        scenario_config.T = args.tesing_number
        scenario_config.train_layouts_PT = 50
        scenario_config.p = 0.0
        scenario_config.S = 9
        CSIs_test, data_list_test, _, _, context_vector_test = dg.generate_T_dataPTandDT(scenario_config)
        # print('1', CSIs_test[0,0,:,:,:])
        
        
        #recover the online training setting
        scenario_config.T = args.T
        scenario_config.train_layouts_PT = args.Nt
        scenario_config.p = args.p
        scenario_config.S = args.S
        
        
        # CSIs_PT, data_list_PT, data_list_DT, data_list_DT_compare, context_vector = dg.generate_T_dataPTandDT(scenario_config)
        
        
        
        eta_network = hyperNetwork(input_size, hidden_size1, hidden_size2, output_size).type(torch.float64)
        model_repara = IGCNet_repara(scenario_config).to(scenario_config.device)
        reset_randomness(args.seed)
        overall_test_eta_PPI(model_repara, eta_network, data_list_test, context_vector_test, CSIs_test, num_data_train_DT=args.Nt_tilde, num_data_train_PT=args.Nt, num_data_test=50)
        
    
    joint_test_GNN()
     