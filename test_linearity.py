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
from scipy.stats import rankdata


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
        splitratio = int(len(Z) * 0.70)
        Z_train, Z_test = Z[:splitratio, :], Z[splitratio:, :]
        S_train, S_test = S[:splitratio, :], S[splitratio:, :]
        reg = LinearRegression().fit(Z_train, S_train)
        S_pred = reg.predict(Z_test)
        mse = round(mean_squared_error(S_test, S_pred), 3)
        reg_score = round(reg.score(Z_test, S_test), 3)
    else:
        reg = LinearRegression().fit(Z, S)
        S_pred = reg.predict(Z)
        mse = round(mean_squared_error(S, S), 3)
        reg_score = round(reg.score(Z, S), 3)
    print(mse, reg_score)
    return mse, reg_score

def compute_rank():    
    key = 'reg_score'
    total = []
    for model in sorted(lin_scores.keys()):
        score = lin_scores[model][key]
        total.append(score)
        
    total_rank_avg = rankdata(total, method='average')
    total_rank_min = rankdata(total, method='min')

    with open('linearity_test/linearity_npoints{0}_var{1}_ranks.pkl'.format(
            str(20), str(0.05)), 'wb') as f:
        pickle.dump({'total_rank_avg': total_rank_avg, 
                     'total_rank_min': total_rank_min, 
                     'total': total}, f)
        
    return total_rank_min

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
#            print(lin_scores[model]['reg_score'])
        lin_scores[model]['mse'] /= data['num_points']
        lin_scores[model]['reg_score'] /= data['num_points']

if False:
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
    
    fig2 = plt.figure(9, figsize=(10, 10))
    plt.clf()
    xmin = min(lin_scores[model][key] for model in lin_scores.keys())
    xlim = (xmin - xmin % 0.01, 1.0)
    gs = fig2.add_gridspec(2, 2)#, width_ratios=[1, 2])
    
    vae_sgroup1 = sorted(vae_group1, key=lambda x: lin_scores[x][key], reverse=True)
    f2_ax1 = fig2.add_subplot(gs[0, 0])
    f2_ax1.set_yticks(np.arange(len(vae_sgroup1)) + 1)
    f2_ax1.set_yticklabels(vae_sgroup1[::-1])
    f2_ax1.set_xlim((0.99, 1.0))
#    f2_ax1.set_xlim(xlim)
    
    vae_sgroup2 = sorted(vae_group2, key=lambda x: lin_scores[x][key], reverse=True)
    f2_ax2 = fig2.add_subplot(gs[0, 1])
    f2_ax2.set_yticks(np.arange(len(vae_sgroup2)) + 1)
    f2_ax2.set_yticklabels(vae_sgroup2[::-1])
    f2_ax2.set_xlim((0.99, 1.0))
#    f2_ax2.set_xlim(xlim)
 
    gan_sgroup1 = sorted(gan_group1, key=lambda x: lin_scores[x][key], reverse=True)
    f2_ax3 = fig2.add_subplot(gs[1, 0])
    f2_ax3.set_yticks(np.arange(len(gan_sgroup1)) + 1)
    f2_ax3.set_yticklabels(gan_sgroup1[::-1])
    f2_ax3.set_xlabel('R2 linearity score')
    f2_ax3.set_xlim(xlim)
    
    gan_sgroup2 = sorted(gan_group2, key=lambda x: lin_scores[x][key], reverse=True)
    f2_ax4 = fig2.add_subplot(gs[1, 1])
    f2_ax4.set_yticks(np.arange(len(gan_sgroup2)) + 1)
    f2_ax4.set_yticklabels(gan_sgroup2[::-1])
    f2_ax4.set_xlabel('MMD aggregated score')
    f2_ax4.set_xlim(xlim)
    
    for model_name in lin_scores.keys():
        res = lin_scores[model_name][key]
        if model_name in vae_group1:
            agg_names = vae_sgroup1
            ax = f2_ax1
        elif model_name in vae_group2:
            agg_names = vae_sgroup2
            ax = f2_ax2
        elif model_name in gan_group1:
            agg_names = gan_sgroup1
            ax = f2_ax3
        else:
            agg_names = gan_sgroup2
            ax = f2_ax4
        
        i = len(agg_names) - agg_names.index(model_name)
        
        ax.barh(i, res, align='center', label=model_name, 
                    height=0.5)
#        ax.legend(loc='upper left', framealpha=1)
        
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.show()
    
if False:
    with open('linearity_test/linearity_npoints{0}_var{1}_ranks.pkl'.format(
            str(20), str(0.05)), 'rb') as f:
        lin_rank_dict = pickle.load(f)
        
    with open('test_pr/ipr_nhood{0}_s{1}_ranks.pkl'.format('20', '10'), 'rb') as f:
        ipr_rank_dict = pickle.load(f)
        
    with open('disentanglement_test/disentanglement_MMDalpha{0}_pval{1}_ranks.pkl'.format(
            '15', '0.001'), 'rb') as f:
        dis_rank_dict = pickle.load(f)
        
    model_names = ['gan{0}'.format(str(i)) for i in range(1, 10)] + \
        ['vae{0}'.format(str(i)) for i in range(1, 10)] 
        
    # Average ranking
    key = '_avg'
    score_sep_avg = np.round(
            (lin_rank_dict['total_rank' + key] + \
             ipr_rank_dict['prec_rank' + key] + \
             ipr_rank_dict['rec_rank' + key] + \
             dis_rank_dict['prec_rank' + key] + \
             dis_rank_dict['rec_rank' + key]) / 5, 3)
    
    score_tot_avg = np.round(
            (lin_rank_dict['total_rank' + key] + \
             ipr_rank_dict['total_rank' + key] + \
             dis_rank_dict['total_rank' + key]) / 3, 3)
    
    score_sep = np.round(np.mean(
            [lin_rank_dict['total'], ipr_rank_dict['prec'],
             ipr_rank_dict['rec'], dis_rank_dict['prec'],
             dis_rank_dict['rec']], axis=0), 3)
    
    score_sep = np.round(np.sum(
            [lin_rank_dict['total'], ipr_rank_dict['prec'],
             ipr_rank_dict['rec'], dis_rank_dict['prec'],
             dis_rank_dict['rec']], axis=0), 3)
    
    ratio = 1
    score_sep2 = np.round(np.mean(
            [lin_rank_dict['total'], 
             ratio * np.array(ipr_rank_dict['rec']) + (1- 1) * np.array(ipr_rank_dict['prec']),
#             0.5 * np.array(dis_rank_dict['prec']) + (1- 0.5) * np.array(dis_rank_dict['rec'])
             dis_rank_dict['total']], axis=0), 3)
    
    score_tot = np.round(np.mean(
            [lin_rank_dict['total'], ipr_rank_dict['total'], 
             dis_rank_dict['total']], axis=0), 3)
        
    # Minimum ranking
    key = '_min'
    score_sep_min = np.round(
            (lin_rank_dict['total_rank' + key] + \
             ipr_rank_dict['prec_rank' + key] + \
             ipr_rank_dict['rec_rank' + key] + \
             dis_rank_dict['prec_rank' + key] + \
             dis_rank_dict['rec_rank' + key]) / 5, 3)
    
    score_tot_min = np.round(
            (lin_rank_dict['total_rank' + key] + \
             ipr_rank_dict['total_rank' + key] + \
             dis_rank_dict['total_rank' + key]) / 3, 3)
        
    scores = [score_sep_avg, score_tot_avg, score_sep_min, score_tot_min]
    for score in [score_tot]:
        print(sorted(list(zip(model_names, score)), key=lambda x:x[1]))
        