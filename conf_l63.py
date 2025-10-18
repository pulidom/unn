#import torch
#import numpy as np

from dyn import L63
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
    nt_start=100 # warm-up times/ input_times

    var_in=[0,2]
    var_out=[0]
    exp_dir=exp_dir
    DynMdl=L63
    batch_size=256
    

    
# Define parametros de la optimizacion
class train: 
    loss = lossfn.loss_fn #nn.MSELoss() #nn.logGaussLoss #SupervLoss # GaussLoss
    batch_size=dat.batch_size
    n_epochs = 200
    learning_rate = 1e-3
    exp_dir = exp_dir
    lvalidation = True 
    patience = 10
    
# Define parametros de la red
class net:
    hidden_dim=40
    layers=3
    input_dim=len(dat.var_in)
    output_dim=len(dat.var_out)
    dropout=0.1
    device=device



