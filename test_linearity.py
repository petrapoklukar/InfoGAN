#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 11:49:12 2020

@author: petrapoklukar
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import torch
import pickle

def sample_fixed_noise(ntype, n_samples, noise_dim, var_range=1, device='cpu'):
    """Samples one type of noise only."""
    if ntype == 'uniform':
        return torch.empty((n_samples, noise_dim), 
                           device=device).uniform_(-var_range, var_range)
    elif ntype == 'normal':
        return torch.empty((n_samples, noise_dim), device=device).normal_()
    elif ntype == 'equidistant':
        noise = (np.arange(n_samples + 1) / n_samples) * 2*var_range - var_range
        return torch.from_numpy(noise)
    else:
        raise ValueError('Noise type {0} not recognised.'.format(ntype))
     
        
def sample_neighbourhoods(ld, n_samples, n_local_samples, dgm_type, scale=0.05, 
                        device='cpu'):
    """
    Samples n_local_samples in a ld-dimensional ball with radius scale centered
    around randomly sampled points in the latent space.
    """
    noise_type = 'normal' if dgm_type == 'vae' else 'uniform'
    linearity_dict = {}
    for i in range(n_samples):
        local_sample = sample_fixed_noise(noise_type, 1, ld).numpy()
        neigh_samples = random_ball(local_sample, scale, n_local_samples, ld)
        linearity_dict[i] = neigh_samples
    return linearity_dict


def random_ball(loc, scale, num_points, ld):
    """
    Samples num_points on a ld-dimensional sphere centered in loc and
    with radius scale.
    """
    random_directions = np.random.normal(size=(ld, num_points))
    random_directions /= np.linalg.norm(random_directions, axis=0)   
    random_radii = np.random.random(num_points) ** (1/ld)
    return scale * (random_directions * random_radii).T + loc


def test_linearity(Z, S):
    """
    Fits a linear transformation S = AZ + B and calulcates its score.
    """
    reg = LinearRegression().fit(Z, S)
    S_pred = reg.predict(Z)
#    print(mean_squared_error(S, S_pred))
#    print(r2_score(S, S_pred))
    print(round(reg.score(Z, S), 2))
    return round(reg.score(Z, S), 2)


models = ['gan' + str(i) for i in range(1, 10)] + ['vae' + str(i) for i in range(1, 10)]
for model in models:
    with open('dataset/linearity_states/linearity_test_{0}.pkl'.format(model), 
              'rb') as f:
        data = pickle.load(f)

    Z = data['latent00']
    S = data['state00'][:, (0, 1, -1)]
    print(model)
    test_linearity(Z, S)
    print('\n')