'''
Author: Qiushuo Hou
Emial: qshou@zju.edu.cn
FilePath: /DT_powered_AMS_code/main_repara_PPI_lambda_practical.py
Description: A-DT-AMS (using PT data to correct the error caused by DT data, introducing the hyperparameters)--- "A-DT-AMS" in the simulation
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
    parser.add_argument('--count', type=int, default=1, help='the number of performance has not be improved')
    parser.add_argument('--R', type=int, default=8, help='update frequency')
    parser.add_argument('--window_length', type=int, default=20, help='window size to estimate variance and covariance')
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

    #compute average loss (DT,PT,\tilde{DT})
    def compute_average_loss(model, eta, data_list, context_vector_t, data_list_PT_ct, data_list_DT_ct, num_data_PT, num_data_DT):
        # model - GNN with rewrite repara
        # eta - hyperparameter network
        # data_list - DT data in time step t with S number of different c_s
        # contxt_vector_t - context in time step t with S number of different c_s
        # data_list_PT_ct - PT graph data at time step t
        # data_list_DT_ct - DT graph data at time step t
        # num_data_PT - number of data in one t (PT side)-for the traing of c_s
        # num_data_DT - number of data in one t (DT side)-for the traing of c_s
        
        #It only extracts the parameter size information of GNN model \phi, 
        # and does not use this model for any forward propagation or backward propagation.
        model_without_repara = IGCNet(scenario_config).to(scenario_config.device)
        
        model.eval()
        eta.eval()
        
        data_loader = DataLoader(data_list, batch_size=num_data_DT, 
                                    shuffle=False, num_workers=1)
        # multi-update
        data_loader_PT = DataLoader(data_list_PT_ct, batch_size=num_data_PT, 
                                    shuffle=False, num_workers=1)
        data_loader_DT = DataLoader(data_list_DT_ct, batch_size=num_data_DT, 
                                    shuffle=False, num_workers=1)

        
        loss_DT_S = torch.zeros(1).to(scenario_config.device)
        
        for (index, data) in enumerate(data_loader):
            data = data.to(scenario_config.device)
            
            context = context_vector_t[index+args.R]
            context = torch.from_numpy(context)
            mean_phi = eta(context)
            
            noise = 0
            phi = mean_phi + noise
            
            var1, var2 = lm.get_var(phi, model_without_repara)
            out = model(data, var1, var2, if_bias=1)
            loss = ut.sr_loss(data,out,scenario_config.train_K, scenario_config.Nt, scenario_config.var, scenario_config)
    
            loss_DT_S += loss/len(data_loader)
        
        
        
        loss_PT = torch.zeros(1).to(scenario_config.device)
        
        for (index, data_PT) in enumerate(data_loader_PT):
            data_PT = data_PT.to(scenario_config.device)
            
            context = context_vector_t[index]
            context = torch.from_numpy(context)
            mean_phi = eta(context)
            
            noise = 0
            phi = mean_phi + noise
            
            var1, var2 = lm.get_var(phi, model_without_repara)
            out_PT = model(data_PT, var1, var2, if_bias=1)
            #multi-update
            loss_PT_r = ut.sr_loss(data_PT,out_PT,scenario_config.train_K, scenario_config.Nt, scenario_config.var, scenario_config)/len(data_loader_PT)
            loss_PT += loss_PT_r
            
        
        
        loss_DT = torch.zeros(1).to(scenario_config.device)
    
        for (index, data_DT) in enumerate(data_loader_DT):
            data_DT = data_DT.to(scenario_config.device)
            
            context = context_vector_t[index]
            context = torch.from_numpy(context)
            mean_phi = eta(context)
            
            noise = 0
            phi = mean_phi + noise
            
            var1, var2 = lm.get_var(phi, model_without_repara)
            out_DT = model(data_DT, var1, var2, if_bias=1)
            
            
            #multi-update
            loss_DT_r = ut.sr_loss(data_DT,out_DT,scenario_config.train_K, scenario_config.Nt, scenario_config.var, scenario_config)/len(data_loader_DT)
            loss_DT += loss_DT_r
        
        return loss_DT_S.item(), loss_DT.item(), loss_PT.item()
    
    ### compute \lambda and \mu based on the explicit solution
    def compute_lambda1_and_lambda2(loss_DT_S_list, loss_DT_list, loss_PT_list, gamma, window_length):
        # loss_DT_S_list - loss_DT_tilde set
        # loss_DT_list - loss_DT set
        # loss_PT_list - loss_PT set
        # gamma - weight of var in MSE, first from 1 and decrease to 0.5
        # window_length - the average length of loss (0 - the mean of all before)
        # t - time step
        
        
        print('Start calculating lambda1 and lambda2 !!!')
        if window_length == 0 or window_length > len(loss_DT_S_list):
            loss_DT_S_mean = np.mean(loss_DT_S_list)
            loss_DT_mean = np.mean(loss_DT_list)
            loss_PT_mean = np.mean(loss_PT_list)
            Q = len(loss_DT_S_list)
        else:
            loss_DT_S_mean = np.mean(loss_DT_S_list[-window_length:])
            loss_DT_mean = np.mean(loss_DT_list[-window_length:])
            loss_PT_mean = np.mean(loss_PT_list[-window_length:])
            Q = window_length
            loss_DT_S_list =  loss_DT_S_list[-window_length:]
            loss_DT_list = loss_DT_list[-window_length:]
            loss_PT_list = loss_PT_list[-window_length:]
        # if t == 0:
        constant = -2.5
        
        #Calculate covariance and variance, and average loss
        C_PT_DT = 0
        V_DT = 0
        V_DT_S = 0
        #Adjust the value of lossDT so that it obeys that its mean value is 0
        Loss_DT = loss_DT_mean-constant
        
        for i in range (len(loss_PT_list)):
            C_PT_DT += 1/(Q-1)*(loss_PT_list[i]-loss_PT_mean)*(loss_DT_list[i]-loss_DT_mean)
            V_DT += 1/(Q-1)*(loss_DT_list[i]-loss_DT_mean)*(loss_DT_list[i]-loss_DT_mean)
            V_DT_S += 1/(Q-1)*(loss_DT_S_list[i]-loss_DT_S_mean)*(loss_DT_S_list[i]-loss_DT_S_mean)
            
        
        
        
        V_DT_S = 0
        
        lambda_1_upper = (2*gamma-1)*Loss_DT**4+gamma*(1-gamma)*Loss_DT**2*C_PT_DT+gamma*(2*gamma-1)*V_DT*Loss_DT**2
        
        lambda_1_down = (2*gamma-1)*Loss_DT**4+gamma**2*(Loss_DT**2*(V_DT+V_DT_S)+V_DT*V_DT_S)
        lambda_1 = lambda_1_upper/lambda_1_down
        
        
        lambda_2_upper = gamma*(Loss_DT**2+C_PT_DT)+(1-gamma)*(lambda_1-1)*Loss_DT**2
        lambda_2_down = gamma*(Loss_DT**2+V_DT)
        lambda_2 = lambda_2_upper/lambda_2_down
        
        if lambda_1 > 1:
            lambda_1 = 1
        if lambda_1 < 0:
            lambda_1 = 0
        
        if lambda_2 > 1:
            lambda_2 = 1
        if lambda_2 < 0:
            lambda_2 = 0
               
       
        print('lossDT', loss_DT_mean)
        # print('lossDT list', loss_DT_list)
        print('lossPT', loss_PT_mean)
        print('C_PT_DT', C_PT_DT)
        print('V_DT', V_DT)
        print('V_DT_S', V_DT_S)

        print('Finish calculating lambda1 and lambda2 !!!')
        
        return lambda_1, lambda_2
        
            
                 
    ### train eta
    def train_eta_PPI(t, T, lambda_1, lambda_2, model, eta, data_list, context_vector_t, data_list_PT_ct, data_list_DT_ct, num_data_PT, num_data_DT, Epoch):
    # def train_eta_PPI(model, eta, data_list, context_vector_t, data_list_PT_ct, data_list_DT_ct, num_data_PT, num_data_DT, Epoch):
        # model - GNN with rewrite repara
        # eta - hyperparameter network
        # data_list - DT data in time step t with S number of different c_s
        # contxt_vector_t - context in time step t with S number of different c_s
        # data_list_PT_ct - PT graph data at time step t
        # data_list_DT_ct - DT graph data at time step t
        # num_data - number of data in one t (DT side)-for the traing of c_s
        # Epoch - number of training times
        # lambda_1 - weight of loss DT with S number c_s
        # lambda_2 - weight of loss DT 1 number c_s
        
        #It only extracts the parameter size information of GNN model \phi, 
        # and does not use this model for any forward propagation or backward propagation.
        model_without_repara = IGCNet(scenario_config).to(scenario_config.device)
        
        model.train()
        eta.train()
        Total_loss = []
        
        data_loader = DataLoader(data_list, batch_size=num_data_DT, 
                                    shuffle=False, num_workers=1)
        
        data_loader_PT = DataLoader(data_list_PT_ct, batch_size=num_data_PT, 
                                    shuffle=False, num_workers=1)
        data_loader_DT = DataLoader(data_list_DT_ct, batch_size=num_data_DT, 
                                    shuffle=False, num_workers=1)
        
        # optimizer_eta = torch.optim.Adam(eta.parameters(), lr=args.lr)
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
           
            total_loss_train = loss_PT + lambda_1*total_loss_train - lambda_2*loss_DT
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
   
       
    ### A-DT-AMS
    def overall_test_eta_PPI(model, eta, data_list_test, context_vector_test, csis_test, num_data_train_DT, num_data_train_PT, num_data_test):
        Sumrate_average = []
        Sumrate_best = []
        
        best_sum_rate = 0
        # change gamma (get the expression of MSE)
        gamma = 0.5
        
        loss_DT_S_list = []
        loss_DT_list = []
        loss_PT_list = []
        
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
            
            
            
            
            if Sumrate_average[t] > best_sum_rate:
                best_sum_rate = Sumrate_average[t]
                count = 0
            else:
                count += 1
            # # adaptive adjust the value of gamma (if needed)
            if (count > args.count): 
                
                gamma *= 1
                
                # adjust window length (from 40 to 5)
                args.window_length *= np.exp((np.log(5)-np.log(40))/10)
                args.window_length  = int(np.floor(args.window_length))
                
            
            
            if args.window_length < 5:
                args.window_length = 5
                
            loss_DT_S, loss_DT, loss_PT = compute_average_loss(model, eta, data_list_DT_S_multi_update, context_vector_t, data_list_PT_t, data_list_DT_t, num_data_train_PT, num_data_train_DT)
            
            loss_DT_S_list.append(loss_DT_S)
            loss_DT_list.append(loss_DT)
            loss_PT_list.append(loss_PT)
            
            if t <= args.window_length:
                # initial value of \lambda and \mu
                lambda_1 = 1
                lambda_2 = 0.5
            else:
                lambda_1, lambda_2 = compute_lambda1_and_lambda2(loss_DT_S_list, loss_DT_list, loss_PT_list, gamma, window_length=args.window_length)
            print('lambda1:', lambda_1)
            print('lambda2:', lambda_2)
            print('gamma:', gamma)
            
            # tags1 = {'PPI_lambda_p%f_seed%d_S%d_pratical_Kfactor5' % (args.p, args.seed, args.S): np.average(Sum)}
            # writer.add_scalars('Epoch=50_repara', tags1, t)
            
            # tags2 = {'lamda1_p%f_seed%d_S%d_pratical' % (args.p, args.seed, args.S): lambda_1}
            # writer.add_scalars('Lambda_1', tags2, t)
            
            # tags3 = {'lamda2_p%f_seed%d_S%d_pratical' % (args.p, args.seed, args.S): lambda_2}
            # writer.add_scalars('Lambda_2', tags3, t)
            
            # tags4 = {'gamma_p%f_seed%d_S%d_pratical' % (args.p, args.seed, args.S): gamma}
            # writer.add_scalars('gamma', tags4, t)

            eta = train_eta_PPI(t, scenario_config.T, lambda_1, lambda_2, model, eta, data_list_DT_S_multi_update, context_vector_t, data_list_PT_t, data_list_DT_t, num_data_train_PT, num_data_train_DT, Epoch=args.Epoch)
            # if t == 65:
            #     torch.save(eta.state_dict(), '/Data1/qiushuo/DT_synchronization_repara/results/PPI_lambda/p=0.6/optimal/eta_initialization.pth')
        
        
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
     
    
    
