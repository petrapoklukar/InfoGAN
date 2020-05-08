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


def load_simulation_state_dict(model_name, fignum=1, plot=True):
    """
    path_to_data: e.g. 'dataset/simulation_states/gan2/'
    """
    path_to_data = 'dataset/simulation_states/{0}/{0}.pkl'.format(model_name)
    with open(path_to_data, 'rb') as f:
        states_dict = pickle.load(f)
    
    # ld = states_dict['0'].shape[1]
    state_list = [0, 1, -1]
    state_names = ['x', 'y', 'theta']
    
    if plot:
        for fixed_fac in list(states_dict.keys()):
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
    

def compute_ks_test(model_data, gts_data, n_samples=2000, p_value=0.01):
    """
    Computes the Kolmogorov-Smirnov two sample test with a given pvalue.
    
    Compares the distribution of each factor F in (X, Y, Theta) obtained from 
    each latent intervention to the grount truth distribution of F obtained 
    from the training data.
    
    Critical values are obtained from Wikipedia 
    https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test.
    """    
    critical_val_dict = {0.01: 1.628*np.sqrt(2/n_samples), 
                         0.05: 1.358*np.sqrt(2/n_samples), 
                         0.005: 1.731*np.sqrt(2/n_samples), 
                         0.001: 1.949*np.sqrt(2/n_samples)}
    c_val = critical_val_dict[p_value]
    
    n_factors = 3
    res_dict = {key: {factor: [] for factor in range(n_factors)} for key in model_data.keys()}
    for e in range(5):
        for key in model_data.keys():
            # Result factors corresponding to each intervention in the latent space
            sample1 = torch.from_numpy(model_data[key])[e*n_samples:(e+1)*n_samples, (0, 1, -1)]
            
            # Sample the same amount of ground truth data
            sample2 = torch.from_numpy(gts_data)
            rand_rows2 = torch.randperm(sample2.size(0))[:n_samples]
            
            # factor is either X, Y or theta
            for factor in range(sample1.size(-1)):
                assert(sample1[:, factor].shape == sample2[rand_rows2, factor].shape )
                
                # KS test if the generated factor distribution is the same as gt one
                ks = stats.ks_2samp(sample1[:, factor], sample2[rand_rows2, factor])
                
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
            inter_ks_list.remove(0)
            key_score = round(np.mean(inter_ks_list), 2)
        elif len(res_agg_dict[key]) == 1:
            key_ks_max = res_agg_dict[key][0]
            key_ks_max_factor = key_ks_max[1]
            key_score = 1.0
        else: 
            res_agg_dict[key] = 'not significant'
            continue
        res_agg_dict[key] = (key_ks_max_factor, round(key_ks_max[0], 2), key_score)
        
    return res_dict, res_agg_dict, res_agg2_dict


def compute_mmd_test(model_data, gts_data, n_samples=100, p_value=0.005):
    """
    Computes the Maximum Mean Discrepancy using exponential kernel with 
    hyperparameter alpha. 
    
    Compares the distribution of each factor F in (X, Y, Theta) obtained from 
    each latent intervention to the grount truth distribution of F obtained 
    from the training data.
    
    Critical value is obtained by performing a permutation test 100x and 
    setting it to (1 - p_value)-quantile.
    """    
    n_factors = 3
    res_dict = {key: {factor: [] for factor in range(n_factors)} for key in model_data.keys()}
    for e in range(5):
        for key in model_data.keys():
            # Result factors corresponding to each intervention in the latent space
            sample1 = torch.from_numpy(model_data[key])[e*2000:(e+1)*2000, (0, 1, -1)]
            rand_rows1 = torch.randperm(sample1.size(0))[:n_samples]
            
            # Sample the same amount of ground truth data
            sample2 = torch.from_numpy(gts_data)
            rand_rows2 = torch.randperm(sample2.size(0))[:n_samples]
            
            # exponential kernel parameters per factor
            alpha_list = [10,10,10]
            
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
                    s1_sh, s2_sh = s1s2[:n_samples], s1s2[n_samples:]
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
            key_ks_max = max(res_agg_dict[key], key=lambda x: x[0])
            key_ks_max_factor = key_ks_max[1]
            inter_ks_list = list(map(lambda x: (key_ks_max[0]-x[0])/key_ks_max[0], res_agg_dict[key]))
            inter_ks_list.remove(0)
            key_score = round(np.mean(inter_ks_list), 2)
        elif len(res_agg_dict[key]) == 1:
            key_ks_max = res_agg_dict[key][0]
            key_ks_max_factor = key_ks_max[1]
            key_score = 1.0
        else: 
            res_agg_dict[key] = 'not significant'
            continue

        res_agg_dict[key] = (key_ks_max_factor, round(key_ks_max[0], 2), key_score)       
    return res_dict, res_agg_dict, res_agg2_dict



if __name__ == '__main___':
    model_names = ['gan2', 'gan3', 'vae5','vae8']
    
    gt_data = np.load('dataset/simulation_states/yumi_states.npy')
    gts_data = gt_data[:, (0, 1, -1)]

    for i in range(len(model_names)):
        print(model_names[i])
        model_data = load_simulation_state_dict(model_names[i], plot=False)
        d, d_temp, _ = compute_mmd_test(model_data, gts_data, n_samples=200, 
                                        p_value=0.001)
        print(' *- MMD', d_temp)
        d, d_temp, _ = compute_ks_test(model_data, gts_data, n_samples=2000, 
                                       p_value=0.001)
        print(' *- KS', d_temp)
        
    model_data = load_simulation_state_dict('vae8', plot=True)
    
    # Plot GT data    
    plt.figure(50)
    # plt.clf()
    # for i in range(6):
    #     plt.subplot(6, 1, i+1)
    #     plt.hist(gt_data[:, i], bins=100)
    # plt.subplots_adjust(hspace=0.5)
    # plt.show()
    
    plt.clf()
    plt.subplot(3, 1, 1)
    plt.hist(gt_data[:, 0])#, bins=50)
    plt.title('X')
    
    plt.subplot(3, 1, 2)
    plt.hist(gt_data[:, 1])#, bins=50)
    plt.title('Y')
    
    plt.subplot(3, 1, 3)
    plt.hist(gt_data[:, -1])#, bins=50)
    plt.title('Theta')
    
    plt.show()
    
    
    # # Manual correction
    # model_data_corr = model_data.copy()
    # pairs = [('2', 1), ('0', 2), ('1', 0)]
    # for p in pairs:
    #     temp3 = model_data[p[0]][:5000, p[1]]
    #     temp1 = model_data[p[0]][torch.randperm(10000)[:5000], :]
    #     temp1[:, p[1]] = temp3
    #     model_data_corr[p[0]] = temp1

    # ld = 3
    # for fixed_fac in list(model_data_corr.keys()):
    #     plt.figure(fixed_fac, figsize=(10, 10))
    #     plt.clf()
    #     for i in range(ld):
    #         plt.subplot(ld, 1, i + 1)
    #         plt.hist(model_data_corr[fixed_fac][:, i], bins=50)
    #         plt.legend()
            
    #     plt.subplots_adjust(hspace=0.5)
    #     plt.show()
    from scipy.stats import wasserstein_distance
    from scipy import stats
    for key in model_data.keys():
        print('Fixed latent dim ', key)
        sample1 = torch.from_numpy(model_data[key])[:, (0, 1, -1)]
        sample2 = torch.from_numpy(gts_data)#.reshape(-1, 1))
        rand_rows1 = torch.randperm(sample1.size(0))[:10000]
        rand_rows2 = torch.randperm(sample2.size(0))[:10000]
        
        inter_mmd_list = []
        for i in range(sample1.size(-1)):
            # mmd = compute_mmd(sample1[:, i].reshape(-1, 1), 
            #                   sample2[rand_rows2, i].reshape(-1, 1), alpha=2)
            mmd = stats.ks_2samp(sample1[rand_rows1, i], 
                              sample2[rand_rows2, i])#, equal_var=False)
            if mmd.pvalue < 0.05:
                inter_mmd_list.append((round(mmd.statistic, 2), i))
                print(key, i, round(mmd.statistic, 2))
        
        if inter_mmd_list:
            key_max_mmd = max(inter_mmd_list, key=lambda x: x[0])
            # np.max((inter_mmd_list - key_max_mmd)/key_max_mmd)
            print((key, key_max_mmd), '\n', )
        
        
    stat_alpha = 0.05
    n_samples = 500
    for key in model_data.keys():
        print('Fixed latent dim ', key)
        sample1 = torch.from_numpy(model_data[key])[:, (0, 1, -1)]
        sample2 = torch.from_numpy(gts_data)#.reshape(-1, 1))
        

        rand_rows1 = torch.randperm(sample1.size(0))[:n_samples]
        rand_rows2 = torch.randperm(sample2.size(0))[:n_samples]
        
        alpha_list = [10,10,10]
        inter_mmd_list = []
        for i in range(sample1.size(-1)):
            s1 = sample1[rand_rows1, i].reshape(-1, 1)
            s2 = sample2[rand_rows2, i].reshape(-1, 1)
            
            s1s2 = torch.cat([s1, s2])
            # Get the 1 - alpha quantile of MMD estimators
            mmd_perm_test = []
            for j in range(100):
                rand_rows1 = torch.randperm(sample1.size(0))[:n_samples]
                rand_rows2 = torch.randperm(sample2.size(0))[:n_samples]
                s1 = sample1[rand_rows1, i].reshape(-1, 1)
                s2 = sample2[rand_rows2, i].reshape(-1, 1)
                
                s1s2 = torch.cat([s1, s2])
                np.random.shuffle(s1s2)
                s1_sh, s2_sh = s1s2[:n_samples], s1s2[n_samples:]
                mmd = compute_mmd(s1_sh, s2_sh, alpha_list[i])
                mmd_perm_test.append(mmd.item())
            rej_threshold = np.quantile(mmd_perm_test, 1 - stat_alpha)
            
            # mmd = compute_mmd(sample1[:, i].reshape(-1, 1), 
            #                   sample2[rand_rows2, i].reshape(-1, 1), alpha=2)
            mmd = compute_mmd(s1, s2, alpha=alpha_list[i]).item()
            if mmd > rej_threshold:
                print('   {0} and {1} not same: {2}'.format(key, str(i), str(mmd)))
            else:
                print('   {0} and {1}: {2}'.format(key, str(i), mmd))
            inter_mmd_list.append((mmd, mmd > rej_threshold))
        
        key_max_mmd = max(inter_mmd_list, key=lambda x: x[0])
        key_max_mmd_dim = inter_mmd_list.index(key_max_mmd)
        inter_mmd_list = list(map(lambda x: (key_max_mmd[0]-x[0])/key_max_mmd[0], inter_mmd_list))
        inter_mmd_list.remove(0)
        key_score = np.mean(inter_mmd_list)
        # np.max((inter_mmd_list - key_max_mmd)/key_max_mmd)
        print((key, key_score, key_max_mmd_dim, key_max_mmd[1]), '\n', )
 
    
    
    
    stat_alpha = 0.05
    n_samples = 500
    for key in model_data.keys():
        print('Fixed latent dim ', key)
        sample1 = torch.from_numpy(model_data[key])[:, (0, 1, -1)]
        sample2 = torch.from_numpy(gts_data)#.reshape(-1, 1))
        for k in range(3):
            print(' k = ', str(k))
            rand_rows1 = torch.randperm(sample1.size(0))[:n_samples]
            rand_rows2 = torch.randperm(sample2.size(0))[:n_samples]
            
            alpha_list = [10,10,10]
            inter_mmd_list = []
            for i in range(sample1.size(-1)):
                s1 = sample1[rand_rows1, i].reshape(-1, 1)
                s2 = sample2[rand_rows2, i].reshape(-1, 1)
                
                s1s2 = torch.cat([s1, s2])
                # Get the 1 - alpha quantile of MMD estimators
                mmd_perm_test = []
                for j in range(100):
                    np.random.shuffle(s1s2)
                    s1_sh, s2_sh = s1s2[:n_samples], s1s2[n_samples:]
                    mmd = compute_mmd(s1_sh, s2_sh, alpha_list[i])
                    mmd_perm_test.append(mmd.item())
                rej_threshold = np.quantile(mmd_perm_test, 1 - stat_alpha)
                
                # mmd = compute_mmd(sample1[:, i].reshape(-1, 1), 
                #                   sample2[rand_rows2, i].reshape(-1, 1), alpha=2)
                mmd = compute_mmd(s1, s2, alpha=alpha_list[i]).item()
                if mmd > rej_threshold:
                    print('   {0} and {1} not same: {2}'.format(key, str(i), str(mmd)))
                else:
                    print('   {0} and {1}: {2}'.format(key, str(i), mmd))
                inter_mmd_list.append((mmd, mmd > rej_threshold))
        
        key_max_mmd = max(inter_mmd_list, key=lambda x: x[0])
        key_max_mmd_dim = inter_mmd_list.index(key_max_mmd)
        inter_mmd_list = list(map(lambda x: (key_max_mmd[0]-x[0])/key_max_mmd[0], inter_mmd_list))
        inter_mmd_list.remove(0)
        key_score = np.mean(inter_mmd_list)
        # np.max((inter_mmd_list - key_max_mmd)/key_max_mmd)
        print((key, key_score, key_max_mmd_dim, key_max_mmd[1]), '\n', )


    for key in model_data.keys():
        print('Fixed latent dim ', key)
        sample1 = torch.from_numpy(model_data[key])[:, (0, 1, -1)]
        sample2 = torch.from_numpy(gts_data)#.reshape(-1, 1))
        rand_rows1 = torch.randperm(sample1.size(0))[:5000]
        rand_rows2 = torch.randperm(sample2.size(0))[:5000]
        
        alpha_list = [10,10,10]
        inter_mmd_list = []
        for i in range(sample1.size(-1)):
            s1 = sample1[rand_rows1, i].reshape(-1, 1)
            s2 = sample2[rand_rows2, i].reshape(-1, 1)
            
            s1s2 = torch.cat([s1, s2])
            # Get the 1 - alpha quantile of MMD estimators
            mmd_perm_test = []
            for j in range(100):
                np.random.shuffle(s1s2)
                s1_sh, s2_sh = s1s2[:5000], s1s2[5000:]
                mmd = compute_mmd(s1_sh, s2_sh, alpha_list[i])
                mmd_perm_test.append(mmd)
            
            # mmd = compute_mmd(sample1[:, i].reshape(-1, 1), 
            #                   sample2[rand_rows2, i].reshape(-1, 1), alpha=2)
            mmd = compute_mmd(sample1[rand_rows1, i].reshape(-1, 1), 
                              sample2[rand_rows2, i].reshape(-1, 1), alpha=alpha_list[i])
            inter_mmd_list.append(mmd.numpy())
            print(mmd)
        
        key_max_mmd = max(inter_mmd_list)
        key_max_mmd_dim = inter_mmd_list.index(key_max_mmd)
        inter_mmd_list = list(map(lambda x: (key_max_mmd-x)/key_max_mmd, inter_mmd_list))
        inter_mmd_list.remove(0)
        key_score = np.mean(inter_mmd_list)
        # np.max((inter_mmd_list - key_max_mmd)/key_max_mmd)
        print((key, key_score, key_max_mmd_dim), '\n', )
            
