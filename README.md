# Automatic AI Model Selection for Wireless Systems: Online Learning via Digital Twinning
This repository contains code for "Automatic AI Model Selection for Wireless Systems: Online Learning via Digital Twinning" -- Qiushuo Hou, Matteo Zecchin, Sangwoo Park, Yunlong Cai, Guanding Yu, Kaushik Chowdhury, and Osvaldo Simeone


![O-RAN](https://github.com/qiushuo0913/DT-powered-AMS/blob/main/image/summary1.png)  
*Figure 1: As a use case of the proposed methodology, this figure illustrates the  O-RAN architecture, in which a base station (BS), known as gNB, is disaggregated into a central unit (CU), a distributed unit (DU), and a radio unit (RU), where both CU and DU are deployed in the O-Cloud, and the RU is deployed at the BS. The near-real time (RT) radio intelligent controller (RIC)  deploys AI-based xApps to carry out functionalities at different layers of the protocol stack. As shown in part (a), in the considered setting, an automatic model selection (AMS) mapping produces the parameters $\phi$ of an AI model  based on  context information from the BS. The AI model may be implemented at the near-RT RIC for an O-RAN system. The focus of this work is on the online optimization of the AMS mapping, which is carried out during a preliminary calibration phase. Specifically, as shown in part (b) for the O-RAN architecture, this work proposes to speed up online calibration via the use of a digital twin, which can produce synthetic data for new contexts.  In the proposed scheme, for each real-world context $c$, the digital twin produces synthetic data from multiple simulated contexts, and the real-world data is leveraged to “rectify” the errors made by the simulator.*
## Dependencies
Python 3.9.18  
Pytorch 1.12.1  
Pytorch Geometric 2.5.0 (conda install pyg -c pyg)  
Scipy 1.11.4  
Tensorboard 2.15.2 (used for visualization, optional)  
## How to use
**main_joint.py** --- for the conventional continual learning scheme that directly trains the GNN model-- *python main_joint.py*  
**main_repara.py** --- for the AMS scheme that only uses PT data-- *python main_repara.py*  
**main_repara_naive.py** --- for the N-DT-AMS scheme that treats the DT data on par with PT data-- *python main_repara_naive.py*  
**main_repara_PPI.py** --- for the DT-AMS scheme that uses the PT data to correct the error caused by the discrepancy between DT and PT-- *python main_repara_PPI.py*   
**main_repara_PPI_lambda_practical.py** --- for the A-DT-AMS scheme that improve the DT-AMS by controlling the trade-off betwwen bias and variance-- *python main_repara_PPI_lambda_practical.py* 
