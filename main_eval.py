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
import utils
# conf para el modelo dinamico
import conf_l63 as conf

# Generate training dataset
[loader_test] = dat.create_dataloaders(conf.dat,['test'])
net_file = conf.exp_dir + '/model_best.ckpt'
Net =torch.load(net_file,map_location=torch.device(conf.device))

print('Making inferences')
for i_batch, (input_dat,target_dat) in enumerate(loader_test):
    input_dat=input_dat.transpose(0,1)
    target_dat=target_dat.transpose(0,1)

    mu,sigma = Net.pred(input_dat, deterministic=True)
    prob_mu,prob_sigma,sample = Net.pred(input_dat, deterministic=False)

rmse_t=utils.leadtime_rmse(target_dat,mu)
cove_t=coverage_gaussian(target_dat, mu, sigma)
crps_t=crps_gaussian(target_dat, mu, sigma)

Mdl = conf.DynMdl()
t=mu.shape[0]*Mdl.dtcy

fig, ax = plt.subplots(3,1,figsize=(9,3))
ax[0].plot(t,rmse_t,color='C0')
ax[0].set(xlabel='lead time',ylabel='RMSE')

ax[1].plot(t,cove_t,color='C0')
ax[1].set(xlabel='lead time',ylabel='Coverage probability')

ax[2].plot(t,crps_t,color='C0')
ax[2].set(xlabel='lead time',ylabel='CRPS')

plt.savefig(conf.exp_dir+f'/rmse-lead-time.png')
fig.tight_layout()
