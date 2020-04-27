#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 09:54:59 2020

@author: petrapoklukar
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'..')
import pickle

class FullyConnecteDecoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(FullyConnecteDecoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, output_size)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

    
def load_generative_model(ld, path_to_model, device):
    """
    Loads the generative model.
    
    ld: latent dimension 
    path_to_model: e.g. 'models/InfoGAN_yumi_l15_s{ld}_SnetS/RL_infogan_checkpoint999.pt'
    """
    
    model_dict = torch.load(path_to_model, map_location=device)
    gen_model = FullyConnecteDecoder(ld, 7*79)
    try:
        gen_model.load_state_dict(model_dict)
        gen_model.eval()
        print(' *- Loaded: ', path_to_model)
        return gen_model
    except: 
        raise NotImplementedError(
                'Generative model {0} not recognized'.format(path_to_model))
        

def sample_fixed_noise(ntype, n_samples, noise_dim, var_range=1, device='cpu'):
        """Samples one type of noise only"""
        if ntype == 'uniform':
            return torch.empty((n_samples, noise_dim), 
                               device=device).uniform_(-var_range, var_range)
        elif ntype == 'normal':
            return torch.empty((n_samples, noise_dim), device=device).normal_()
        # Equidistant on interval [-var_range, var_range]
        elif ntype == 'equidistant':
            noise = (np.arange(n_samples + 1) / n_samples) * 2*var_range - var_range
            return torch.from_numpy(noise)
        else:
            raise ValueError('Noise type {0} not recognised.'.format(ntype))
            
            
def sample_latent_codes(ld, n_samples, dgm_type, ntype='equidistant', device='cpu'):
    """
    """
    n_equidistant_pnts = 4
    n_repeats = n_samples / (n_equidistant_pnts + 1)
    latent_codes_dict = {}
    noise_type = 'normal' if dgm_type == 'vae' else 'uniform'
    
    for dim in range(ld):
        codes = sample_fixed_noise(noise_type, n_samples, noise_dim=ld)
        fixed_n_range = sample_fixed_noise(ntype, n_equidistant_pnts, 
                                           noise_dim=None, var_range=1.5)
        fixed_n = np.repeat(fixed_n_range, n_repeats)
        codes[:, dim] = fixed_n
        latent_codes_dict[str(dim)] = codes
    return latent_codes_dict
        

def load_simulation_state_dict(model_name, fignum=1):
    """
    path_to_data: e.g. 'dataset/simulation_states/gan2/'
    """
    path_to_data = 'dataset/simulation_states/{0}/{0}.pkl'.format(model_name)
    with open(path_to_data, 'rb') as f:
        states_dict = pickle.load(f)
    
    ld = states_dict['0'].shape[1]
    state_names = ['x', 'y', 'theta']
    
    for fixed_fac in list(states_dict.keys()):
        plt.figure(fixed_fac, figsize=(10, 10))
        plt.clf()
        plt.suptitle(path_to_data.split('/')[-2] + ' with fixed {0} dim'.format(fixed_fac))
        for i in range(ld):
            plt.subplot(ld, 1, i + 1)
            plt.hist(states_dict[fixed_fac][:, i], bins=50, 
                     label='Std: ' + str(round(np.std(states_dict[fixed_fac][:, i]), 2)))
            plt.legend()
            plt.title(state_names[i])
        plt.subplots_adjust(hspace=0.5)
        plt.show()
    
    
def compute_mmd(sample1, sample2, alpha):
    """
    Computes MMD for samples of the same size bs x n_features using Gaussian
    kernel.
    
    See Equation (3) in http://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf
    for the exact formula.
    """
    # Get number of samples (assuming m == n)
    m = sample1.shape[0]
    
    # Calculate pairwise products for each sample (each row). This yields
    # 2-norms |xi|^2 on the diagonal and <xi, xj> on non diagonal
    xx = torch.mm(sample1, sample1.t())
    yy = torch.mm(sample2, sample2.t())
    zz = torch.mm(sample1, sample2.t())
    
    # Expand the norms of samples into the original size
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    # Remove the diagonal elements, the non diagonal are exactly |xi - xj|^2
    # = <xi, xi> + <xj, xj> - 2<xi, xj> = |xi|^2 + |xj|^2 - 2<xi, xj>
    kernel_samples1 = torch.exp(- alpha * (rx.t() + rx - 2*xx))
    kernel_samples2 = torch.exp(- alpha * (ry.t() + ry - 2*yy))
    kernel_samples12 = torch.exp(- alpha * (rx.t() + ry - 2*zz))
    
    # Normalisations
    n_same = (1./(m * (m-1)))
    n_mixed = (2./(m * m)) 
    
    term1 = n_same * torch.sum(kernel_samples1)
    term2 = n_same * torch.sum(kernel_samples2)
    term3 = n_mixed * torch.sum(kernel_samples12)
    return term1 + term2 - term3
    

def load_simulation_states(path_to_data, fignum=1):
    """
    path_to_data: e.g. 'dataset/simulation_states/gan2/'
    """
    # actions = np.load(path_to_data + 'actions.npy')
    states = np.load(path_to_data + 'states.npy')
    ld = states.shape[1]
    state_names = ['x', 'y', 'theta']
    
    plt.figure(fignum, figsize=(10, 10))
    plt.clf()
    plt.suptitle(path_to_data.split('/')[-2])
    for i in range(ld):
        plt.subplot(ld, 1, i + 1)
        plt.hist(states[:, i], bins=50, label='Std: ' + str(round(np.std(states[:, i]), 2)))
        plt.legend()
        plt.title(state_names[i])
    plt.subplots_adjust(hspace=0.5)
    plt.show()
    

if __name__ == '__main___':
    model_names = ['dataset/simulation_states/gan2/', 
                   'dataset/simulation_states/gan3/', 
                   'dataset/simulation_states/vae5/',
                   'dataset/simulation_states/vae8/']
    
    for i in range(len(model_names)):
        load_simulation_states(model_names[i], fignum=i)
