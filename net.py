'''Defines the recursive neural network
    Version preliminar tomada de deepar
     Se la piensa para sistemas dinamicos. 
     Se la hace mas eficiente.
     Se maneja teacher forcing y prediccion desde la propia clase Net
'''

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Net(nn.Module):
    
    def __init__(self, conf):
        '''
        We define a recurrent network that predicts the future values 
        of a time-dependent variable based on
        past inputs and covariates.
        '''
        super().__init__()
        self.conf = conf
        # LSTM transforma: input_dim -> hidden_dim
        self.lstm = nn.LSTM(input_size=conf.input_dim,
                            hidden_size=conf.hidden_dim,
                            num_layers=conf.layers,
                            bias=True,
                            batch_first=False,
                            dropout=conf.dropout)
        
        self.initialize_forget_gate() 
        self.return_state =True #conf.return_state
        self.output_dim = conf.output_dim

        # Capas finales transforman: hidden_dim -> output_dim 
        self.distribution_mu = nn.Linear(conf.hidden_dim * conf.layers, conf.output_dim)
        self.distribution_sigma = nn.Sequential(
            nn.Linear(conf.hidden_dim * conf.layers, conf.output_dim),
            nn.Softplus(),) # softplus to make sure standard deviation is positive

    def forward(self, input, hx=None):
        '''
        Predict mu and sigma of the distribution for z_t.
        Args:
             x: ([1, batch_size, input_dim]): z_{t-1} + x_t, note that z_0 = 0
            hidden ([layers, batch_size, hidden_dim]): LSTM h from time step t-1
            cell ([layers, batch_size, hidden_dim]): LSTM c from time step t-1
        Returns:
            mu ([batch_size]): estimated mean of z_t
            sigma ([batch_size]): estimated standard deviation of z_t
            hidden ([layers, batch_size, hidden_dim]): LSTM h from time step t
            cell ([layers, batch_size, hidden_dim]): LSTM c from time step t
        '''
        output, (hidden, cell) = self.lstm(input, hx)

        # use h from all three layers to calculate mu and sigma
        hidden_permute = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)
        mu = self.distribution_mu(hidden_permute)
        sigma = self.distribution_sigma(hidden_permute)
        outputs = mu, sigma # torch.squeeze(mu), torch.squeeze(sigma)

        #mu and sigma shape [batch_size, conf.output_dim]
        if self.return_state:
            return outputs, (hidden, cell)
        else:
            return outputs

    def init_hidden(self, input_size):
        return torch.zeros(self.conf.layers, input_size, self.conf.hidden_dim, device=self.conf.device)

    def init_cell(self, input_size):
        return torch.zeros(self.conf.layers, input_size, self.conf.hidden_dim, device=self.conf.device)
    
    def initialize_forget_gate(self):
        ''' initialize LSTM forget gate bias to be 1 as recommended
        by http://proceedings.mlr.press/v37/jozefowicz15.pdf
        '''
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)  # Forget gate = 1

    def train_step(self, train_batch):
        '''
        Para el entrenamiento forward pass in the training time window 
        with teacher forcing.
        
        '''
        batch_size = train_batch.shape[1]
        seq_len = train_batch.shape[0]
        hidden = self.init_hidden(batch_size)
        cell = self.init_cell(batch_size)

        self.return_state = True
        
        mu_t = torch.zeros(seq_len, batch_size, self.output_dim, device=self.conf.device)
        sigma_t = torch.zeros(seq_len, batch_size, self.output_dim, device=self.conf.device)

        for t in range(seq_len):
            
            (mu_t[t], sigma_t[t]), (hidden, cell) = self.forward(
                train_batch[t].unsqueeze_(0).clone(), 
                 (hidden, cell)
            )

        return mu_t, sigma_t
    
    def pred(self, x, deterministic=True):
        ''' 
         Prediccion usando warmup time y luego si autoregresivo
           option deterministica o estocastica
         x = [seq,batch,output_dim]
        '''
        batch_size = x.shape[1]
        nt_start = self.conf.nt_start
        nt_window = self.conf.nt_window
        nt_pred = nt_window - self.conf.nt_start
        n_samples = self.conf.n_samples
        hidden = self.init_hidden(batch_size)
        cell = self.init_cell(batch_size)

        mu = torch.zeros((nt_window, batch_size, self.output_dim), device=self.conf.device)
        sigma = torch.zeros((nt_window, batch_size, self.output_dim), device=self.conf.device)
        # warming up period
        for t in range(nt_start):
            (mu[t], sigma[t]), (hidden, cell) = self.forward(
                x[t].unsqueeze_(0).clone(), 
                 (hidden, cell)     )

        # prediction period: 
        if deterministic:
            for t in range(nt_start,self.conf.nt_window):
                xaug=torch.cat([mu[t-1],x[t-1,...,self.output_dim:]],dim=-1)
                (mu[t], sigma[t]), (hidden, cell) = self.forward(
                    xaug.unsqueeze_(0).clone(), 
                    (hidden, cell)     )
            return mu, sigma
        else: # stochastic
            samples = torch.zeros(
                nt_pred, n_samples,batch_size, 
                self.output_dim,
                device=self.conf.device
            )
            # replico para todas las muestras
            hidden = hidden.repeat_interleave(n_samples, dim=1) 
            cell = cell.repeat_interleave(n_samples, dim=1)

            # Inicializo 
            mu_rep = mu[nt_start-1].unsqueeze(0).repeat(n_samples, 1, 1)  # [n_samples, batch_size, output_dim]
            sigma_rep = sigma[nt_start-1].unsqueeze(0).repeat(n_samples, 1, 1) 

            gaussian = torch.distributions.normal.Normal(mu_rep, sigma_rep)
            sampler = gaussian.sample().view(n_samples * batch_size, self.output_dim)
            
            # Prediction with Autoregression 
            for t in range(nt_start, nt_window):
                covariates = x[t, :, self.output_dim:].repeat_interleave(n_samples, dim=0)
                xaug = torch.cat([sampler, covariates], dim=-1)

                (mu_samples, sigma_samples), (hidden, cell) = self.forward(
                    xaug.unsqueeze(0).clone(),
                    (hidden, cell)
                )

                gaussian = torch.distributions.normal.Normal(mu_samples.squeeze(0), sigma_samples.squeeze(0))
                sampler = gaussian.sample()

                samples[t-nt_start] = sampler.view(n_samples, batch_size, self.output_dim)
                mu[t] = torch.median(samples[t - nt_start], dim=0)[0]
                sigma[t] = samples[t - nt_start].std(dim=0)
            # samples [n_sample, nt_window-nt_start,batch_size,output_dim]
            return mu, sigma, samples
            

