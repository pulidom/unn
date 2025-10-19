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
net_file = conf.exp_dir + f'/model_{conf.sexp}_best.ckpt'
Net =torch.load(net_file,map_location=torch.device(conf.device))

net_file2 = conf.exp_dir + f'/model_{conf.sexp2}_best.ckpt'
Net2 =torch.load(net_file,map_location=torch.device(conf.device))

print('Making inferences')
for i_batch, (input_dat,target_dat) in enumerate(loader_test):
    input_dat=input_dat.transpose(0,1)
    target_dat=target_dat.transpose(0,1)

    mu,sigma = Net.pred(input_dat, deterministic=True)
    prob_mu,prob_sigma,sample = Net.pred(input_dat, deterministic=False)

    mu2,sigma2 = Net2.pred(input_dat, deterministic=True)
    prob_mu2,prob_sigma2,sample2 = Net2.pred(input_dat, deterministic=False)

rmse_t=utils.leadtime_rmse(target_dat,mu)
cove_t=utils.coverage_gaussian(target_dat, mu, sigma)
crps_t=utils.crps_gaussian(target_dat, mu, sigma)

rmse_t=rmse_t.cpu().detach().numpy()
cove_t=cove_t.cpu().detach().numpy()
crps_t=crps_t.cpu().detach().numpy()

rmse_t2=utils.leadtime_rmse(target_dat,mu2)
cove_t2=utils.coverage_gaussian(target_dat, mu2, sigma2)
crps_t2=utils.crps_gaussian(target_dat, mu2, sigma2)

rmse_t2=rmse_t2.cpu().detach().numpy()
cove_t2=cove_t2.cpu().detach().numpy()
crps_t2=crps_t2.cpu().detach().numpy()

Mdl = conf.dat.DynMdl()
t=np.arange(mu.shape[0])*Mdl.dtcy
print(t.shape)
print(rmse_t.shape)

fig, ax = plt.subplots(3,1,figsize=(7,8))
ax[0].plot(t,rmse_t,color='C0')
ax[0].plot(t,rmse_t2+1,color='C1')
ax[0].set(xlabel='lead time',ylabel='RMSE')

ax[1].plot(t,cove_t,color='C0')
ax[1].plot(t,cove_t2,color='C1')
ax[1].set(xlabel='lead time',ylabel='Coverage probability')

ax[2].plot(t,crps_t,color='C0')
ax[2].plot(t,crps_t2,color='C1')
ax[2].set(xlabel='lead time',ylabel='CRPS')

fig.tight_layout()
plt.savefig(conf.exp_dir+f'/rmse-lead-time.png')
