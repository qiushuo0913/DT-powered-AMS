'''
Author: Qiushuo Hou
Emial: qshou@zju.edu.cn
FilePath: /DT_powered_AMS_code/data_generation.py
Description: generating simulated data at DT and data at PT 
'''

#%%
import scipy.io as sio                     # import scipy.io for .mat file I/O 
import numpy as np                         # import numpy
import wireless_network_generate_DTandPT as wg

import utils as ut


#generate data
def generate_T_dataPTandDT(scenario_config):
    # generate number T c_t, their corresponding DT and PT data, and context variable
    dists, CSIs_PT, CSIs_DT, CSIs_DT_compare = wg.sample_generate(scenario_config, scenario_config.T)

    CSIs_PT_real, CSIs_PT_imag = np.real(CSIs_PT), np.imag(CSIs_PT)
    CSIs_DT_real, CSIs_DT_imag = np.real(CSIs_DT), np.imag(CSIs_DT)
    CSIs_DT_compare_real, CSIs_DT_compare_imag = np.real(CSIs_DT_compare), np.imag(CSIs_DT_compare)

    norm_CSIs_PT_real = ut.normalize_data(CSIs_PT_real,scenario_config.T*scenario_config.train_layouts_PT, scenario_config)
    norm_CSIs_PT_imag = ut.normalize_data(CSIs_PT_imag,scenario_config.T*scenario_config.train_layouts_PT, scenario_config)
    norm_CSIs_DT_real = ut.normalize_data(CSIs_DT_real,scenario_config.T*scenario_config.train_layouts_DT, scenario_config)
    norm_CSIs_DT_imag = ut.normalize_data(CSIs_DT_imag,scenario_config.T*scenario_config.train_layouts_DT, scenario_config)
    norm_CSIs_DT_compare_real = ut.normalize_data(CSIs_DT_compare_real,scenario_config.T*scenario_config.train_layouts_DT*(scenario_config.S+1), scenario_config)
    norm_CSIs_DT_compare_imag = ut.normalize_data(CSIs_DT_compare_imag,scenario_config.T*scenario_config.train_layouts_DT*(scenario_config.S+1), scenario_config)
    

    data_list_PT, context_vector_list = ut.proc_data(CSIs_PT, dists, norm_CSIs_PT_real, norm_CSIs_PT_imag, scenario_config.train_K, 0, scenario_config)
    data_list_DT, _ = ut.proc_data(CSIs_DT, dists, norm_CSIs_DT_real, norm_CSIs_DT_imag, scenario_config.train_K, 1, scenario_config)
    data_list_DT_compare, _ = ut.proc_data(CSIs_DT_compare, dists, norm_CSIs_DT_compare_real, norm_CSIs_DT_compare_imag, scenario_config.train_K, 1, scenario_config)
    
    return CSIs_PT, data_list_PT, data_list_DT, data_list_DT_compare, context_vector_list



def generate_S_dataDT_multi_update(scenario_config):
    # generate number S c_s and their corresponding simulated data at DT
    dists, CSIs_PT, CSIs_DT, _ = wg.sample_generate(scenario_config, scenario_config.S)

    
    CSIs_DT_real, CSIs_DT_imag = np.real(CSIs_DT), np.imag(CSIs_DT)
    CSIs_PT_real, CSIs_PT_imag = np.real(CSIs_PT), np.imag(CSIs_PT)

    
    norm_CSIs_DT_real = ut.normalize_data(CSIs_DT_real,scenario_config.S*scenario_config.train_layouts_DT, scenario_config)
    norm_CSIs_DT_imag = ut.normalize_data(CSIs_DT_imag,scenario_config.S*scenario_config.train_layouts_DT, scenario_config)

    norm_CSIs_PT_real = ut.normalize_data(CSIs_PT_real,scenario_config.S*scenario_config.train_layouts_PT, scenario_config)
    norm_CSIs_PT_imag = ut.normalize_data(CSIs_PT_imag,scenario_config.S*scenario_config.train_layouts_PT, scenario_config)
    
    
    data_list_DT, context_vector_list = ut.proc_data(CSIs_DT, dists, norm_CSIs_DT_real, norm_CSIs_DT_imag, scenario_config.train_K, 1, scenario_config)
    data_list_PT, _ = ut.proc_data(CSIs_PT, dists, norm_CSIs_PT_real, norm_CSIs_PT_imag, scenario_config.train_K, 0, scenario_config)
    return  data_list_PT, data_list_DT, context_vector_list

