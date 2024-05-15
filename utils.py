'''
Author: Qiushuo Hou
Emial: qshou@zju.edu.cn
FilePath: /DT_powered_AMS_code/utils.py
Description: 
'''

import scipy.io as sio                     # import scipy.io for .mat file I/O 
import numpy as np                         # import numpy
import matplotlib.pyplot as plt            # import matplotlib.pyplot for figure plotting
import torch
from torch_geometric.data import Data
import wireless_network_generate_DTandPT as wg
import time

class init_parameters():
    def __init__(self, args):
        # wireless network settings
        self.n_links = args.K
        self.n_receiver = args.K
        self.train_K = args.K #In order to reduce subsequent changes, the first three variables have the same meaning.
        self.field_length = args.field_length
        self.shortest_directLink_length = args.short_length
        self.longest_directLink_length = args.long_length
        self.shortest_crossLink_length = 1
        self.bandwidth = 5e6
        self.carrier_f = 2.4e9
        self.tx_height = 1.5
        self.rx_height = 1.5
        self.antenna_gain_decibel = 2.5
        self.tx_power_milli_decibel = 40
        self.tx_power = 1#np.power(10, (self.tx_power_milli_decibel-30)/10)
        self.noise_density_milli_decibel = -169
        self.input_noise_power = np.power(10, ((self.noise_density_milli_decibel-30)/10)) * self.bandwidth
        self.output_noise_power = 1#self.input_noise_power
        self.SNR_gap_dB = 6
        self.SNR_gap = 1 #np.power(10, self.SNR_gap_dB/10)
        self.setting_str = "{}_links_{}X{}_{}_{}_length".format(self.n_links, self.field_length, self.field_length, self.shortest_directLink_length, self.longest_directLink_length)
        # 2D occupancy grid setting
        self.cell_length = 5
        self.Nt = 1
        self.maxrx = 2
        self.minrx = 1
        self.n_grids = np.round(self.field_length/self.cell_length).astype(int)
        self.device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        self.var = 1 # noise power
        # K-factor
        self.K_factor_PT = args.K_factor_PT
        self.K_factor_DT = args.K_factor_DT
        
        
        #数据集设置
        self.S = args.S # number of context variables in one time step
        self.T = args.T # total number of time steps
        self.train_layouts_PT = args.Nt 
        self.train_layouts_DT = args.Nt_tilde 
        #DT dropout node probability
        self.p = args.p
        
        


#Normalize data
def normalize_data(train_data, train_layouts, scenario_config):
    
    tmp_mask = np.expand_dims(np.eye(scenario_config.train_K),axis=-1)
    tmp_mask = [tmp_mask for i in range(scenario_config.Nt)]
    mask = np.concatenate(tmp_mask,axis=-1)
    mask = np.expand_dims(mask,axis=0)
    
    train_copy = np.copy(train_data)
    diag_H = np.multiply(mask,train_copy)
    diag_mean = np.sum(diag_H/scenario_config.Nt)/train_layouts/scenario_config.train_K
    diag_var = np.sqrt(np.sum(np.square(diag_H))/train_layouts/scenario_config.train_K/scenario_config.Nt)
    tmp_diag = (diag_H - diag_mean)/diag_var

    off_diag = train_copy - diag_H
    off_diag_mean = np.sum(off_diag/scenario_config.Nt)/train_layouts/scenario_config.train_K/(scenario_config.train_K-1)
    off_diag_var = np.sqrt(np.sum(np.square(off_diag))/scenario_config.Nt/train_layouts/scenario_config.train_K/(scenario_config.train_K-1))
    tmp_off = (off_diag - off_diag_mean)/off_diag_var
    tmp_off_diag = tmp_off - np.multiply(tmp_off,mask)
    
    norm_train = np.multiply(tmp_diag,mask) + tmp_off_diag
    
    return norm_train


def build_graph(CSI, dist, norm_csi_real, norm_csi_imag, K, p, ifDT, threshold, scenario_config):
    
    context_vector = np.zeros(K-1)
    n = CSI.shape[0]
    Nt = CSI.shape[2]
    x1 = np.array([CSI[ii,ii,:] for ii in range(K)])
    x2 = np.imag(x1)
    x1 = np.real(x1)
    x3 = 1/np.sqrt(Nt)*np.ones((n,2*Nt))
    
    x = np.concatenate((x3,x1,x2),axis=1)
    x = torch.tensor(x, dtype=torch.float)
    
    
    dist2 = np.copy(dist)
    mask = np.eye(K)
    diag_dist = np.multiply(mask,dist2)
    
    dist2 = dist2 + scenario_config.field_length*10 * diag_dist 
    dist2[dist2 > scenario_config.field_length*10] = 0 # ensure no self-circle in graph
    
    
    context_vector = dist.flatten()
    
    #Determine whether it is on the DT side, i.e., whether to use the DT side graph that discards edges with p probability.
    if ifDT:
        flat_dist2 = dist2.flatten()
        sort_index = np.argsort(-flat_dist2)
        
        # Set the first p*K*(K-1) elements in the sorted one-dimensional array to 0 (descending order)
        P = p*K*(K-1)
        P = np.floor(P).astype(int)
        sort_index = sort_index[:P]
        flat_dist2[sort_index] = 0

        # Resize the one-dimensional array to the original matrix size
        dist2 = flat_dist2.reshape(dist2.shape)
    
    
    
                
    attr_ind = np.nonzero(dist2)
    
    edge_attr_real = norm_csi_real[attr_ind]
    edge_attr_imag = norm_csi_imag[attr_ind]
    
    edge_attr = np.concatenate((edge_attr_real,edge_attr_imag), axis=1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    attr_ind = np.array(attr_ind)
    adj = np.zeros(attr_ind.shape)
    adj[0,:] = attr_ind[1,:]
    adj[1,:] = attr_ind[0,:]
    edge_index = torch.tensor(adj, dtype=torch.long)
    
    H1 = np.expand_dims(np.real(CSI),axis=-1)
    H2 = np.expand_dims(np.imag(CSI),axis=-1)
    HH = np.concatenate((H1,H2),axis=-1)
    y = torch.tensor(np.expand_dims(HH,axis=0), dtype=torch.float)
    data = Data(x=x, edge_index=edge_index.contiguous(),edge_attr = edge_attr, y = y)
    #the return data is either the DT or PT data at time c_t
    return data, context_vector

#construct the graph data
def proc_data(HH, dists, norm_csi_real, norm_csi_imag, K, ifDT, scenario_config):
    
    n = HH.shape[0]
    data_list = []
    context_vector_list = []
   
   # fake value, for consistency 
    threshold = 100
    
    if ifDT:
        number_H = scenario_config.train_layouts_DT
    else:
        number_H = scenario_config.train_layouts_PT
    
    HH = np.reshape(HH, (-1, K, K, scenario_config.Nt))
    norm_csi_real = np.reshape(norm_csi_real, (-1, K, K, scenario_config.Nt))
    norm_csi_imag = np.reshape(norm_csi_imag, (-1, K, K, scenario_config.Nt))
    
    for i in range(n):
        for j in range(number_H):
            csi_index = j+i*number_H
            data, context_vector = build_graph(HH[csi_index,:,:,:],dists[i,:,:], 
                                               norm_csi_real[csi_index,:,:,:], norm_csi_imag[csi_index,:,:,:], 
                                               K, scenario_config.p, ifDT, threshold, scenario_config)
            data_list.append(data)
        context_vector_list.append(context_vector)
        
    return data_list, context_vector_list

def power_check(p):
    n = p.shape[0]
    pp = np.sum(np.square(p),axis=1)
    print(np.sum(pp>1.1))


def sr_loss(data,p,K,N,var, scenario_config):
    # H1 K*K*N
    # p1 K*N
    H1 = data.y[:,:,:,:,0]
    H2 = data.y[:,:,:,:,1]
    p1 = p[:,:N]
    p2 = p[:,N:2*N]
    p1 = torch.reshape(p1,(-1,K,1,N))
    p2 = torch.reshape(p2,(-1,K,1,N))
    
    rx_power1 = torch.mul(H1, p1)
    rx_power1 = torch.sum(rx_power1,axis=-1)

    rx_power2 = torch.mul(H2, p2)
    rx_power2 = torch.sum(rx_power2,axis=-1)

    rx_power3 = torch.mul(H1, p2)
    rx_power3 = torch.sum(rx_power3,axis=-1)

    rx_power4 = torch.mul(H2, p1)
    rx_power4 = torch.sum(rx_power4,axis=-1)

    rx_power = torch.mul(rx_power1 - rx_power2,rx_power1 - rx_power2) + torch.mul(rx_power3 + rx_power4,rx_power3 + rx_power4)
    mask = torch.eye(K, device = scenario_config.device)
    valid_rx_power = torch.sum(torch.mul(rx_power, mask), axis=1)
    interference = torch.sum(torch.mul(rx_power, 1 - mask), axis=1) + var
    rate = torch.log2(1 + torch.div(valid_rx_power, interference))
    sum_rate = torch.mean(torch.sum(rate, axis=1))
    loss = torch.neg(sum_rate)
    return loss

