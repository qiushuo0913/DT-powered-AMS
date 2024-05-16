# DT-powered-AMS
This repository contains code for "Automatic AI Model Selection for Wireless Systems: Online Learning via Digital Twinning"  
## Dependencies
Python 3.9.18  
Pytorch 1.12.1  
Pytorch Geometric 2.5.0 (conda install pyg -c pyg)  
Scipy 1.11.4  
Tensorboard 2.15.2 (used for visualization, optional)  
## How to use
**mian_joint.py** --- for the conventional scheme that directly trains the GNN model-- *python main_joint.py*  
**mian_repara.py** --- for the AMS scheme that only uses PT data-- *python main_repara.py*  
**mian_repara_naive.py** --- for the N-DT-AMS scheme that treats the DT data on par with PT data-- *python main_repara_naive.py*  
**mian_repara_PPI.py** --- for the DT-AMS scheme that uses the PT data to correct the error caused by the discrepancy between DT and PT-- *python main_repara_PPI.py*   
**mian_repara_PPI_lambda_practical.py** --- for the A-DT-AMS scheme that improve the DT-AMS by controlling the trade-off betwwen bias and variance-- *python main_repara_PPI_lambda_practical.py* 
