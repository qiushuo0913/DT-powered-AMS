'''
Author: Qiushuo Hou
Emial: qshou@zju.edu.cn
FilePath: /DT_powered_AMS_code/wireless_network_generate_DTandPT.py
Description: generate the small-scale fading and large-scale fading at each time step
'''


import numpy as np
# from main import parse_args
# args = parse_args()

# np.random.seed(args.seed)

def layout_generate(general_para):
    N = general_para.n_links
    # first, generate transmitters' coordinates
    tx_xs = np.random.uniform(low=0, high=general_para.field_length, size=[N,1])
    tx_ys = np.random.uniform(low=0, high=general_para.field_length, size=[N,1])
    
   

    layout_rx = []
    # generate rx one by one rather than N together to ensure checking validity one by one
    rx_xs = []; rx_ys = []
    tot_links = 0
    n_re = general_para.n_receiver
    for i in range(N):
        n_links = i
        rx_i = []

        num_rx = np.random.randint(general_para.minrx, general_para.maxrx)
        num_rx = min(num_rx,  n_re - tot_links)
        tot_links += num_rx
        for j in range(num_rx): 
            got_valid_rx = False
            while(not got_valid_rx):
                pair_dist = np.random.uniform(low=general_para.shortest_directLink_length, high=general_para.longest_directLink_length)
                pair_angles = np.random.uniform(low=0, high=np.pi*2)
                rx_x = tx_xs[i] + pair_dist * np.cos(pair_angles)
                rx_y = tx_ys[i] + pair_dist * np.sin(pair_angles)
                if(0<=rx_x<=general_para.field_length and 0<=rx_y<=general_para.field_length):
                    got_valid_rx = True
            rx_i.append([rx_x[0], rx_y[0]])
        layout_rx.append(rx_i)
        if(tot_links >= n_re):
            break

    # For now, assuming equal weights and equal power, so not generating them
    layout_tx = np.concatenate((tx_xs, tx_ys), axis=1)
    
    return layout_tx, layout_rx


def distance_generate(general_para,layout_tx,layout_rx):
    distances = np.zeros((general_para.n_receiver,general_para.n_receiver))
    N = len(layout_rx)
    sum_tx = 0
    for tx_index in range(N):
        num_loops = len(layout_rx[tx_index])
        tx_coor = layout_tx[tx_index]
        for tx_i in range(num_loops):
            sum_rx = 0
            for rx_index in range(N):
                for rx_i in layout_rx[rx_index]:
                    rx_coor = rx_i
                    distances[sum_rx][sum_tx] = np.linalg.norm(tx_coor - rx_coor)
                    sum_rx += 1
            sum_tx += 1
    return distances


# Generate dataset at PT
def CSI_PT_generate(general_para, train_layouts):
    
    print("<<<<<<<<<<<<<{} PT layouts: {}>>>>>>>>>>>>".format(
        train_layouts, general_para.setting_str))
    
    
    layout_tx, layout_rx = layout_generate(general_para)
    dis = distance_generate(general_para,layout_tx,layout_rx)
    
    
    #Generate number N_t small-scale channels + large-scale channels
    small_scale_CSIs = []
    Nt = general_para.Nt
    L = general_para.n_receiver
    dists = np.expand_dims(dis,axis=-1)
    shadowing = np.random.randn(L,L,Nt)
    large_scale_CSI = 4.4*10**5/((dists**1.88)*(10**(shadowing*6.3/20)))
    for i in range(train_layouts):
        # small_scale_CSI = 1/np.sqrt(2)*(np.random.randn(L,L,Nt)+1j*np.random.randn(L,L,Nt))*np.sqrt(large_scale_CSI)
        
        # Rician channel
        rayleigh_channel = 1/np.sqrt(2)*(np.random.randn(L,L,Nt)+1j*np.random.randn(L,L,Nt))
        if general_para.K_factor_PT == 0:
            complex_los = 0
        else:
            angle_for_loss = np.random.uniform(low=0, high=np.pi*2)
            complex_los = np.cos(angle_for_loss) + 1j*np.sin(angle_for_loss)
            
        small_scale_CSI = np.sqrt(general_para.K_factor_PT / (general_para.K_factor_PT + 1))*complex_los+np.sqrt(1 / (general_para.K_factor_PT + 1))*rayleigh_channel
        small_scale_CSI = small_scale_CSI*np.sqrt(large_scale_CSI)
        
        
        small_scale_CSIs.append(small_scale_CSI)
    return dis, small_scale_CSIs


# Generate dataset at DT
def CSI_DT_generate(general_para, dis, train_layouts):
    
    print("<<<<<<<<<<<<<{} DT layouts: {}>>>>>>>>>>>>".format(
        train_layouts, general_para.setting_str))
    
    
    
    small_scale_CSIs = []
    Nt = general_para.Nt
    L = general_para.n_receiver
    dists = np.expand_dims(dis,axis=-1)
    shadowing = np.random.randn(L,L,Nt)
    large_scale_CSI = 4.4*10**5/((dists**1.88)*(10**(shadowing*6.3/20)))
    for i in range(train_layouts):
        # small_scale_CSI = 1/np.sqrt(2)*(np.random.randn(L,L,Nt)+1j*np.random.randn(L,L,Nt))*np.sqrt(large_scale_CSI)
        
        
        rayleigh_channel = 1/np.sqrt(2)*(np.random.randn(L,L,Nt)+1j*np.random.randn(L,L,Nt))
        if general_para.K_factor_DT == 0:
            complex_los = 0
        else:
            angle_for_loss = np.random.uniform(low=0, high=np.pi*2)
            complex_los = np.cos(angle_for_loss) + 1j*np.sin(angle_for_loss)
            
        small_scale_CSI = np.sqrt(general_para.K_factor_DT / (general_para.K_factor_DT + 1))*complex_los+np.sqrt(1 / (general_para.K_factor_DT + 1))*rayleigh_channel
        small_scale_CSI = small_scale_CSI*np.sqrt(large_scale_CSI)
        
        
        small_scale_CSIs.append(small_scale_CSI)
    return small_scale_CSIs
    


def sample_generate(general_para, number_of_layouts):
    

    dists = []
    CSIs_DT = []
    CSIs_DT_compare = []
    CSIs_PT = []
    
    train_layouts_PT = general_para.train_layouts_PT
    train_layouts_DT = general_para.train_layouts_DT
    
    
    for i in range(number_of_layouts):
        
        dis, csis_PT = CSI_PT_generate(general_para, train_layouts_PT) 
        csis_DT = CSI_DT_generate(general_para, dis, train_layouts_DT)
        csis_DT_compare = CSI_DT_generate(general_para, dis, train_layouts_DT*(general_para.S+1))
        
        #data collection
        dists.append(dis)
        CSIs_PT.append(csis_PT)
        CSIs_DT.append(csis_DT)
        CSIs_DT_compare.append(csis_DT_compare)
            
    dists = np.array(dists)
    CSIs_PT = np.array(CSIs_PT)
    CSIs_DT = np.array(CSIs_DT)
    CSIs_DT_compare = np.array(CSIs_DT_compare)
    return dists, CSIs_PT, CSIs_DT, CSIs_DT_compare
