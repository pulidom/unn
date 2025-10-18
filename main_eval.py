#!/usr/bin/python3
"""
Lee la red neuronal entrenada para L63
"""
import sys; sys.path.insert(0, '../')
import numpy as np
import matplotlib.pyplot as plt
import torch
import datagen as dat
import net
from train import train
# conf para el modelo dinamico
import conf_l63 as conf

# Generate training dataset
[loader_test] = dat.create_dataloaders(conf.dat,['test'])
net_file = conf.exp_dir + '/model_best.ckpt'

Net =torch.load(net_file,map_location=torch.device(conf.device))

for i_batch, (input_dat,target_dat) in enumerate(loader_test):
    print(input_dat.shape)
    prediction = Net.pred(input_dat)
    
