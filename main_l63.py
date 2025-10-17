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
# conf para el modelo dinamico
import conf_l63 as conf

# Generate training dataset
loader_train, loader_val = dat.create_dataloaders(conf)
# Neural net
Net = net.Net(conf.net)
# Training
best_net,loss_t,loss_v = train( Net,train_loader,val_loader,conf.train )

