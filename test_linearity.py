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
import matplotlib
matplotlib.use('Qt5Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

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


def test_linearity(Z, S, split=True):
    """
    Fits a linear transformation S = AZ + B and calulcates its score.
    """
    if split:
#        ld = Z.shape[-1]
#        min_params = 3 * ld + 3
#        splitratio = min_params + min_params % 10
        splitratio = int(len(Z) * 0.20)
        Z_train, Z_test = Z[:splitratio, :], Z[splitratio:, :]
        S_train, S_test = S[:splitratio, :], S[splitratio:, :]
#        print(Z_train.shape, Z_test.shape)
        reg = LinearRegression().fit(Z_train, S_train)
        print(reg.coef_.shape)
        S_pred = reg.predict(Z_test)
        mse = round(mean_squared_error(S_test, S_pred), 3)
        reg_score = round(reg.score(Z_test, S_test), 3)
    else:
        reg = LinearRegression().fit(Z, S)
        S_pred = reg.predict(Z)
        mse = round(mean_squared_error(S, S), 3)
        reg_score = round(reg.score(Z, S), 3)
    return mse, reg_score

models = ['gan' + str(i) for i in range(1, 10)] + ['vae' + str(i) for i in range(1, 10)]

if False:
    lin_scores = {model: {'reg_score': 0, 'mse': 0} for model in models}
    
    for model in models:
        with open('dataset/linearity_states/linearity_test_{0}.pkl'.format(model), 
                  'rb') as f:
            data = pickle.load(f)
        
        print(model)
        for sample in range(data['num_points']):
            key = '{:02d}'.format(sample)
            Z = data['latent' + key]
            S = data['state' + key]#[:, (0, 1, -1)]
            mse, reg_score = test_linearity(Z, S, split=True)
            lin_scores[model]['mse'] += mse 
            lin_scores[model]['reg_score'] += reg_score 
            print(lin_scores[model]['reg_score'])
        lin_scores[model]['mse'] /= data['num_points']
        lin_scores[model]['reg_score'] /= data['num_points']

if True:
    vae_group1 = ['vae' + str(i) for i in range(1, 6)]
    vae_group2 = ['vae' + str(i) for i in range(6, 10)]
    gan_group1 = ['gan' + str(i) for i in range(1, 6)]
    gan_group2 = ['gan' + str(i) for i in range(6, 10)]
    
    key = 'reg_score'
    ymin = min(lin_scores[model][key] for model in lin_scores.keys())
    ylim = (ymin - ymin % 0.01, 1.0)
    plt.figure(1, figsize=(10, 10))
    plt.clf()
    plt.subplot(2, 2, 1)
    for model in lin_scores.keys():
        if model in vae_group1:
            i = vae_group1.index(model)
            plt.bar(i, lin_scores[model][key], label=model)
    plt.legend(loc='lower right', framealpha=1)
    plt.ylim((0.99, 1.0))
        
    
    plt.subplot(2, 2, 2)
    for model in lin_scores.keys():
        if model in vae_group2:
            i = vae_group2.index(model)
            plt.bar(i, lin_scores[model][key], label=model)
    plt.legend(loc='lower right', framealpha=1)
    plt.ylim((0.99, 1.0))
    
    plt.subplot(2, 2, 3)
    for model in lin_scores.keys():
        if model in gan_group1:
            i = gan_group1.index(model)
            plt.bar(i, lin_scores[model][key], label=model)
    plt.legend(loc='lower right', framealpha=1)
    plt.ylim(ylim)
    
    plt.subplot(2, 2, 4)
    for model in lin_scores.keys():
        if model in gan_group2:
            i = gan_group2.index(model)
            plt.bar(i, lin_scores[model][key], label=model)
    plt.legend(loc='lower right', framealpha=1)
    plt.ylim(ylim)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.show()