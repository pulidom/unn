#import torch
#import numpy as np

from dyn import L63 as Dyn_Mdl
import lossfn

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device='cpu'

seed=2
exp_dir= './dat/l63'

# Define parametros de los datos
class dat: 
    n_train=20000
    n_val = 501

    nt_jump=100   # salto de los datos
    nt_window=200  # total times
    nt_start=100 # warm-up times

    var_in=[0,2]
    var_out=[0]
    exp_dir=exp_dir

# Define parametros de la optimizacion
class train: 
    loss = lossfn.loss_fn #nn.MSELoss() #nn.logGaussLoss #SupervLoss # GaussLoss
    batch_size=256
    n_epochs = 30
    learning_rate = 1e-3
    exp_dir = exp_dir
    lval = True 
    patience = 10
    
# Define parametros de la red
class net:
    hidden=40
    layers=3
    dropout=0.1
    device=device



    
