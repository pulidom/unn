#!/usr/bin/python3
"""
Entrenamiento de una red neuronal tipo DeepAR para el L63
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
[loader_train, loader_val] = dat.create_dataloaders(conf.dat)
# Neural net
Net = net.Net(conf.net)
# Training
best_net,loss_t,loss_v = train( Net,loader_train,loader_val,conf.train )


plt.figure(figsize=(6,4))
plt.plot(range(len(loss_t)),loss_t,color='C0')
plt.plot(range(len(loss_v)),loss_v,color='C1')
plt.xlabel('Epoch');
plt.ylabel('Loss');
plt.savefig(conf.exp_dir+f'/loss.png')
