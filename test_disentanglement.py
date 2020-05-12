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
from scipy import stats
import heapq
import csv


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
    Performs the latent interventions where one dimension is fixed and the rest
    are sampled either normally (in case DGM is a VAE) or uniformally on [-1, 1]
    (in case DGM is an InfoGAN).
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
        # fixed_val = sample_fixed_noise(noise_type, 1, noise_dim=1)
        # fixed_n = np.repeat(fixed_val, n_samples)
        codes[:, dim] = fixed_n
        latent_codes_dict[str(dim)] = codes
    #latent_codes_dict['-1'] = sample_fixed_noise(noise_type, n_samples, noise_dim=ld)
    latent_codes_dict['info'] = [('n_equidistant_pnts', n_equidistant_pnts), 
                                 ('n_repeats', n_repeats)]
    return latent_codes_dict


def load_simulation_state_dict(file_name, fignum=1, plot=True):
    """
    path_to_data: e.g. 'dataset/simulation_states/gan2/'
    """
    path_to_data = 'dataset/simulation_states/{0}.pkl'.format(file_name)
    with open(path_to_data, 'rb') as f:
        states_dict = pickle.load(f)
    
    # ld = states_dict['0'].shape[1]
    state_list = [0, 1, -1]
    state_names = ['x', 'y', 'theta']
    keys_list = list(states_dict.keys())
    keys_list.remove('info')
    if plot:
        for fixed_fac in keys_list:
            plt.figure(fixed_fac, figsize=(10, 10))
            plt.clf()
            plt.suptitle(path_to_data.split('/')[-2] + ' with fixed {0} dim'.format(fixed_fac))
            for i in range(len(state_list)):
                plt.subplot(len(state_list), 1, i + 1)
                plt.hist(states_dict[fixed_fac][:, state_list[i]], bins=50, 
                         label='Std: ' + str(round(np.std(states_dict[fixed_fac][:, i]), 2)))
                plt.legend()
                plt.title(state_names[i])
            plt.subplots_adjust(hspace=0.5)
            plt.show()
    return states_dict
        
        
    
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
    

def compute_ks_test(model_data, gts_data, n_sub_samples=2000, p_value=0.01,
                    n_inter=4, n_inter_samples=2000, n_resampling=10):
    """
    Computes the Kolmogorov-Smirnov two sample test with a given pvalue.
    
    Compares the distribution of each factor F in (X, Y, Theta) obtained from 
    each latent intervention to the grount truth distribution of F obtained 
    from the training data.
    
    Critical values are obtained from Wikipedia 
    https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test.
    """    
    critical_val_dict = {0.01: 1.628*np.sqrt(2/n_sub_samples), 
                         0.05: 1.358*np.sqrt(2/n_sub_samples), 
                         0.005: 1.731*np.sqrt(2/n_sub_samples), 
                         0.001: 1.949*np.sqrt(2/n_sub_samples)}
    c_val = critical_val_dict[p_value]
    assert(n_sub_samples <= n_inter_samples)
    
    n_factors = 3
    res_dict = {key: {factor: [] for factor in range(n_factors)} for key in model_data.keys()}
    for e in range(n_inter + 1):
        for key in [k for k in model_data.keys() if k !='info']:
            # Result factors corresponding to each intervention in the latent space
            sample1 = torch.from_numpy(model_data[key])[e*n_inter_samples:(e+1)*n_inter_samples, (0, 1, -1)]
            
            # Sample the same amount of ground truth data
            sample2 = torch.from_numpy(gts_data)

            for p in range(n_resampling):
                rand_rows1 = torch.randperm(sample1.size(0))[:n_sub_samples]
                rand_rows2 = torch.randperm(sample2.size(0))[:n_sub_samples]
                
                # factor is either X, Y or theta
                for factor in range(sample1.size(-1)):
                    assert(sample1[rand_rows1, factor].shape == sample2[rand_rows2, factor].shape )
                    
                    # KS test if the generated factor distribution is the same as gt one
                    ks = stats.ks_2samp(sample1[rand_rows1, factor], sample2[rand_rows2, factor])
                    
                    # If small enough p-value as well as big enough statistics, reject null hypothesis
                    if ks.pvalue < p_value and ks.statistic > c_val:
                        # Distributions are different
                        res_dict[key][factor].append(round(ks.statistic - c_val, 2))
                    
    res_agg_dict = {}
    res_agg2_dict = {}
    for key in res_dict.keys():
        res_agg_dict[key] = []
        res_agg2_dict[key] = []
        for f in res_dict[key]:
            # Mean value of succesfull KS tests per pair (latent_intervention, factor)
            # The higher the better!
            avg_mmd_score = np.mean(res_dict[key][f])
            if not np.isnan(avg_mmd_score):
                res_agg_dict[key].append((avg_mmd_score, f))
                res_agg2_dict[key].append((avg_mmd_score, f))
        # Select the factor where ks was the highest and compare it with the rest to get
        # the confidence level
        if len(res_agg_dict[key]) > 1:
            key_ks_max = max(res_agg_dict[key], key=lambda x: x[0])
            key_ks_max_factor = key_ks_max[1]
            inter_ks_list = list(map(lambda x: (key_ks_max[0]-x[0])/key_ks_max[0], res_agg_dict[key]))
            inter_ks_list.remove(0.0)
            key_score = round(np.mean(inter_ks_list), 2)
        elif len(res_agg_dict[key]) == 1:
            key_ks_max = res_agg_dict[key][0]
            key_ks_max_factor = key_ks_max[1]
            key_score = 1.0
        else: 
            # res_agg_dict[key] = 'not significant'
            continue
        res_agg_dict[key] = (key_ks_max_factor, round(key_ks_max[0], 5), key_score)
        
    return res_dict, res_agg_dict, res_agg2_dict


def compute_mmd_test(model_data, gts_data, n_sub_samples=100, p_value=0.005, 
                     n_inter=4, n_inter_samples=2000, n_resampling=10):
    """
    Computes the Maximum Mean Discrepancy using exponential kernel with 
    hyperparameter alpha. 
    
    Compares the distribution of each factor F in (X, Y, Theta) obtained from 
    each latent intervention to the grount truth distribution of F obtained 
    from the training data.
    
    Critical value is obtained by performing a permutation test 100x and 
    setting it to (1 - p_value)-quantile.
    """    
    assert(n_sub_samples <= n_inter_samples)
    n_factors = 3
    res_dict = {key: {factor: [] for factor in range(n_factors)} for key in model_data.keys()}
    for e in range(n_inter + 1):
        for key in [k for k in model_data.keys() if k !='info']:
            
            # Result factors corresponding to each intervention in the latent space
            sample1 = torch.from_numpy(model_data[key])[e*n_inter_samples:(e+1)*n_inter_samples, (0, 1, -1)]
            
            # Sample the same amount of ground truth data
            sample2 = torch.from_numpy(gts_data)
            
            # exponential kernel parameters per factor
            alpha_list = [10, 10, 10]
            
            for p in range(n_resampling):
                rand_rows1 = torch.randperm(sample1.size(0))[:n_sub_samples]
                rand_rows2 = torch.randperm(sample2.size(0))[:n_sub_samples]
                
                # factor is either X, Y or theta
                for factor in range(sample1.size(-1)):
                    s1 = sample1[rand_rows1, factor].reshape(-1, 1)
                    s2 = sample2[rand_rows2, factor].reshape(-1, 1)
                    
                    # Get the critical value from the permutation test by choosing
                    # the 1 - pvalue quantile of the distribution of MMD estimators
                    s1s2 = torch.cat([s1, s2])
                    mmd_perm_test = []
                    for j in range(100):
                        np.random.shuffle(s1s2)
                        s1_sh, s2_sh = s1s2[:n_sub_samples], s1s2[n_sub_samples:]
                        mmd = compute_mmd(s1_sh, s2_sh, alpha_list[factor])
                        mmd_perm_test.append(mmd.item())
                    c_val = np.quantile(mmd_perm_test, 1 - p_value)
                    
                
                    mmd = compute_mmd(s1, s2, alpha=alpha_list[factor]).item()
                    if mmd > c_val:
                        res_dict[key][factor].append(round(mmd - c_val, 2))
    
    res_agg_dict = {}
    res_agg2_dict = {}
    for key in res_dict.keys():
        res_agg_dict[key] = []
        res_agg2_dict[key] = []
        for f in res_dict[key]:
            # Mean value of succesfull KS tests per pair (latent_intervention, factor)
            # The higher the better!
            avg_mmd_score = np.mean(res_dict[key][f])
            if not np.isnan(avg_mmd_score):
                res_agg_dict[key].append((avg_mmd_score, f))
                res_agg2_dict[key].append((avg_mmd_score, f))
        # Select the factor where ks was the highest and compare it with the rest to get
        # the confidence level
        if len(res_agg_dict[key]) > 1:
            key_mmd_max = max(res_agg_dict[key], key=lambda x: x[0])
            key_mmd_max_factor = key_mmd_max[1]
            inter_mmd_list = list(map(lambda x: (key_mmd_max[0]-x[0])/key_mmd_max[0], res_agg_dict[key]))
            inter_mmd_list.remove(0.0)
            key_score = round(np.mean(inter_mmd_list), 2)
        elif len(res_agg_dict[key]) == 1:
            key_mmd_max = res_agg_dict[key][0]
            key_mmd_max_factor = key_mmd_max[1]
            key_score = 1.0
        else: 
            # res_agg_dict[key] = 'not significant'
            del res_agg_dict[key] 
            continue

        res_agg_dict[key] = (key_mmd_max_factor, round(key_mmd_max[0], 5), key_score)       
        
    return res_dict, res_agg_dict, res_agg2_dict


def select_top3_factors(res_dict):
    """
    Selelcts three latent interventions with the highest hypothesis test 
    score.
    """
    n_factors = 3
    score_list = [[key, res_dict[key][1]] for key in res_dict.keys()]
    top3_keys = heapq.nlargest(3, score_list, key=lambda x: x[1])
    dis_precision = round(np.sum(list(map(lambda x: x[1], top3_keys))), 3)
    cover_list = [res_dict[key][0] for key in map(lambda x: x[0], top3_keys)]
    unique_cover_list = np.unique(cover_list)
    dis_recall = round(len(unique_cover_list) / n_factors, 3)
    top3_results = [i + [res_dict[i[0]][0]] for i in top3_keys]
    return dis_precision, dis_recall, top3_results
    

if __name__ == '__main__':
    model_names = [
        'data/ppo_gan1_t1.pkl', 
        'data/ppo_gan2_t1.pkl', 
        'data/ppo_gan3_t1.pkl', 
        'data/ppo_vae1_t1.pkl', 
        'data/ppo_vae2_t1.pkl', 
        'data/ppo_vae3_t1.pkl', 
        'data/ppo_vae4_t1.pkl', 
        'data/ppo_vae5_t1.pkl', 
        'data/ppo_vae6_t1.pkl', 
        'data/ppo_vae7_t1.pkl', 
        'data/ppo_vae8_t1.pkl', 
        'data/ppo_vae9_t1.pkl'
        ] 
    
    gt_data = np.load('dataset/simulation_states/yumi_states.npy')
    gts_data = gt_data[:, (0, 1, -1)]

    end_results = {}
    results_filename = 'results/disentanglement_scores.csv'
    with open(results_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
         
        for i in range(len(model_names)-6):
            model_name = model_names[i].split('.')[0].split('/')[-1]
            print(model_name)
            
            end_results[model_name] = {}
            
            
            model_data = load_simulation_state_dict(model_names[i], plot=False)
            indices = list(map(lambda x: 0 if x[0] == 'n_equidistant_pnts' else 1, model_data['info']))
            n_inter = model_data['info'][indices[0]][1]
            n_inter_samples = int(model_data['info'][indices[1]][1])
            del model_data['info']
            mmdd, mmdd_temp, mmdd_temp1 = compute_mmd_test(model_data, gts_data, n_sub_samples=20, 
                                            p_value=0.001, n_inter=n_inter,
                                            n_inter_samples=n_inter_samples)
            dp_mmd, dr_mmd, t3_mmd = select_top3_factors(mmdd_temp)
            end_results[model_name]['mmd'] = (dp_mmd, dr_mmd)
            t3_mmd = sorted(t3_mmd, key=lambda x: x[0])
            print(' *- MMD', t3_mmd)
            
            d, d_temp, _ = compute_ks_test(model_data, gts_data, n_sub_samples=2000, 
                                           p_value=0.001, n_inter=n_inter,
                                            n_inter_samples=n_inter_samples)
            dp_ks, dr_ks, t3_ks = select_top3_factors(d_temp)
            end_results[model_name]['ks'] = (dp_ks, dr_ks)
            t3_ks = sorted(t3_ks, key=lambda x: x[0])
            print(' *- KS', t3_ks)
            
            writer.writerow([model_name, 'MMD', dp_mmd, dr_mmd, t3_mmd, 'KS', dp_mmd, dr_mmd, t3_mmd])
        
    
    with open('results/disentanglement_scores_dict.pkl', 'wb') as f:
        pickle.dump(end_results, f)
 
    # plot the results
    plt.figure(10)
    plt.clf()
    plt.subplot(1, 2, 1)
    for model in end_results.keys():
        model_name = model.split('_')[1]
        x, y = end_results[model]['mmd']
        plt.scatter(x, y, alpha=0.7, label=model_name, marker='D')
    plt.legend()
    plt.title('Disentanglement using MMD')
    plt.xlabel('disentangling precision')
    plt.ylabel('disentangling recall')
    plt.ylim((0.28, 1.02))
    plt.yticks(ticks=[0.333, 0.666, 1.0], labels=['1/3', '2/3', '3/3'])
    # plt.show()

    # plt.figure(11)
    # plt.clf()
    plt.subplot(1, 2, 2)
    for model in end_results.keys():
        model_name = model.split('_')[1]
        x, y = end_results[model]['ks']
        plt.scatter(x, y, alpha=0.7, label=model_name, marker='D')
    plt.legend()
    plt.title('Disentanglement using KS')  
    plt.xlabel('disentangling precision')
    # plt.ylabel('disentangling recall')
    plt.ylim((0.28, 1.02))
    plt.yticks(ticks=[0.333, 0.666, 1.0], labels=['1/3', '2/3', '3/3'])
    plt.show()
    