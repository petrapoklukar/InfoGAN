#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:55:49 2020

@author: petrapoklukar
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from importlib.machinery import SourceFileLoader
import os
import sys
sys.path.insert(0,'..')
import prd_score as prd
from itertools import groupby
import InfoGAN_yumi as infogan
import InfoGAN_models as models
import matplotlib
matplotlib.use('Qt5Agg') # Must be before importing matplotlib.pyplot or pylab!
#matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import pickle
import iprd_score as iprd 
import tensorflow as tf
from scipy.stats import rankdata


class TrajDataset(Dataset):
    def __init__(self, data_filename, device=None, scaled=True):
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.np_data = np.load(data_filename)
        self.min = np.min(self.np_data, axis=0)
        self.max = np.max(self.np_data, axis=0)
        unit_scaled = (self.np_data - self.min) / (self.max - self.min)
        self.data_scaled = 2 * unit_scaled - 1
        if scaled:
            self.data = torch.from_numpy(self.data_scaled).float().to(self.device)
        else: 
            self.data = torch.from_numpy(self.np_data).float().to(self.device)
        self.num_samples = self.data.shape[0]                
    
    def get_subset(self, max_ind, n_points, fixed_indices=None, reshape=True):
        if fixed_indices is None:
            self.prd_indices = np.random.choice(max_ind, n_points, replace=False)
        else:
            self.prd_indices = fixed_indices
        subset_list = [self.data[i].cpu().numpy() for i in self.prd_indices]
        if reshape: 
            return np.array(subset_list).reshape(n_points, -1)
        else: 
            return np.array(subset_list)
    
    def scale(self, data):
        unit_scaled = (data.cpu().numpy().reshape(-1, 7, 79) - self.min) / (self.max - self.min)
        data_scaled = 2 * unit_scaled - 1
        return data_scaled
        
    def descale(self, data):
        unit_scaled = (data.cpu().numpy() + 1)/2
        descaled = (unit_scaled.reshape(-1, 7, 79) * (self.max - self.min)) + self.min
        return descaled
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    
def get_ref_samples(baseline_indices):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_prd_samples = len(baseline_indices)
    path_to_data = 'dataset/robot_trajectories/yumi_joint_pose.npy'
    test_dataset = TrajDataset(path_to_data, device, scaled=False)
    ref_np = test_dataset.get_subset(
        len(test_dataset), n_prd_samples, fixed_indices=baseline_indices, 
        reshape=False)
    return ref_np
    

def get_vae_samples(config_name, ld, n_prd_samples):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the trained model
    filename = 'models/models/{}.mdl'.format(config_name)
    model = models.FullyConnecteDecoder(ld, 7*79)
    model_dict = {}
    for k, v in torch.load(filename, map_location=device).items():
        if 'decoder' in k:
            k_new = '.'.join(k.split('.')[1:])
            model_dict[k_new] = v
    model.load_state_dict(model_dict)
    model = model.to(device)

    # Get the sampled np array
    with torch.no_grad():
        z_noise = torch.empty((n_prd_samples, ld), device=device).normal_()
        eval_np = model(z_noise).detach().cpu().numpy().reshape(-1, 7, 79)
    return eval_np
    

def get_infogan_samples(config_name, ld, n_prd_samples, chpnt=''):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device is: {}'.format(device))

    # Load the trained model
    model = models.FullyConnecteDecoder(ld, 7*79)
    filename = 'models/models/{0}.pt'.format(config_name)
    model_dict = torch.load(filename, map_location=device)
    model.load_state_dict(model_dict)
    model = model.to(device)
    
    # Get the ground truth np array from the test split
    path_to_data = 'dataset/robot_trajectories/yumi_joint_pose.npy'
    test_dataset = TrajDataset(path_to_data, device, scaled=False)

    # Get the sampled np array
    with torch.no_grad():
        z_noise = torch.empty((n_prd_samples, ld), device=device).uniform_(-1, 1)
        eval_data = model(z_noise).detach()
        eval_np = test_dataset.descale(eval_data).reshape(-1, 7, 79)
    return eval_np


def moving_average(x, n):
    axis = 2
    y_padded = np.pad(x, ((0, 0), (0, 0), (n//2, n-1-n//2)), mode='edge')
    y_smooth = np.apply_along_axis(
            lambda ax: np.convolve(ax, np.ones((n,))/n, mode='valid'), 
            axis, y_padded)
    return y_smooth 


def index_to_ld(i):
    if i <= 3:
        return 2
    elif 4 <= i and i <= 6:
        return 3
    else:
        return 6


if __name__ == '__main__':    
      
    gan_configs = [
            'infogan1_l01_s2_4RL_model',
            'infogan2_l15_s2_4RL_model',
            'infogan3_l35_s2_4RL_model',
            'infogan4_l01_s3_4RL_model',
            'infogan5_l15_s3_4RL_model',
            'infogan6_l35_s3_4RL_model',
            'infogan7_l01_s6_4RL_model',
            'infogan8_l15_s6_4RL_model',
            'infogan9_l35_s6_4RL_model']
    yumi_gan_models = {model: index_to_ld(int(model.split('_')[0][-1])) \
        for model in gan_configs}
    yumi_vae_models = {'vae' + str(i): index_to_ld(i) for i in range(1, 10)}
    evaluate = True
    analyse = False
    
    if analyse:
        n_samples = 2000
        with open('test_pr/ipr_results_nhoodFinal_{0}samples.pkl'.format(n_samples), 'rb') as f:
            data_o = pickle.load(f)
        
        data = {}    
        for key, value in data_o.items():
            if 'infogan' in key:
                new_key = key.split('_')[0][4:]
                data[new_key] = value
            else:
                data[key] = value
    
        # plot the results
        vae_group1 = ['vae' + str(i) for i in range(1, 6)]
        vae_group2 = ['vae' + str(i) for i in range(6, 10)]
        gan_group1 = ['gan' + str(i) for i in range(1, 6)]
        gan_group2 = ['gan' + str(i) for i in range(6, 10)]
        
        color_dict = {'vae1': '#bedef4', 'vae2': '#ffd6b3', 'vae3': '#c3eec3',
                      'vae4': '#f3bebf', 'vae5': '#d9cae8',
                      
                      'vae6': '#67b1e5', 'vae7': '#ffa04d', 'vae8': '#73d873',
                      'vae9': '#e36869',
                      
                      'gan1': '#1f77b4', 'gan2': '#ff7f0e', 'gan3': '#2ca02c', 
                      'gan4': '#d62728', 'gan5': '#7545a0', 
                      
                      'gan6': '#13486d', 'gan7': '#803c00', 'gan8': '#6c1414', 
                      'gan9': '#412759'                      
                }
        
        blue = ['#bedef4', '#67b1e5', '#1f77b4', '#13486d' ]
        orange = ['#ffd6b3', '#ffa04d', '#ff7f0e', '#803c00']
        green = ['#c3eec3', '#73d873', '#2ca02c', '#113c11']
        red = ['#f3bebf', '#e36869', '#d62728', '#6c1414']
        purple = ['#d9cae8', '#a783c9', '#7545a0', '#412759']
                  
        
        def compute_rank(nhood, s):
            vae_res = 'res0_vae'
            gan_res = 'res{0}_gan'.format(str(s))
            ni = [20, 25, 30, 50].index(int(nhood))
            
            prec_data, rec_data = [], []
            total = []
            for model in sorted(data.keys()):
                key = gan_res if 'gan' in model else vae_res
                prec = data[model][key]['precision'][ni]
                rec = data[model][key]['recall'][ni]
                prec_data.append(prec)
                rec_data.append(rec)
                total.append(prec + rec)
                
            prec_rank_min = rankdata(prec_data, method='min')
            prec_rank_avg = rankdata(prec_data, method='average')
            rec_rank_min = rankdata(rec_data, method='min')
            rec_rank_avg = rankdata(rec_data, method='average')
            total_rank_min = rankdata(total, method='min')
            total_rank_avg = rankdata(total, method='average')
            
            with open('test_pr/ipr_nhood{0}_s{1}_ranks.pkl'.format(nhood, s), 'wb') as f:
                pickle.dump({
                        'prec_rank_min': prec_rank_min, 'prec_rank_avg': prec_rank_avg, 
                         'rec_rank_min': rec_rank_min, 'rec_rank_avg': rec_rank_avg, 
                         'total_rank_min': total_rank_min, 
                         'total_rank_avg': total_rank_avg,
                         'total': total, 'prec': prec_data, 'rec': rec_data}, f)
            return prec_rank_min, rec_rank_min, total_rank_min
            
            
                  
        def plot_gan_vae_ipr(nhoods, smooth):
#            nhoods = ['5']#['', '5', '12', '20']
#            smooth = ['0', '10']
#            nhood = 20 #[20,25,30,50]
            ni = [20, 25, 30, 50].index(int(nhoods[0]))
            
            plt.figure(12, figsize=(10, 10))
            plt.clf()
#            plt.suptitle('Improved PR scores')
            for nhood in nhoods:
                for s in smooth:
#                    vae_res = 'res{0}_vae'.format(nhood)
#                    gan_res = 'res{0}_gan{1}'.format(nhood, s)
                    vae_res = 'res0_vae'
                    gan_res = 'res10_gan'
                    s = 10

                    prec_list = list(map(lambda k: 
                        data[k][vae_res]['precision'][ni] if 'vae' in k else \
                        data[k][gan_res]['precision'][ni], data.keys()))
                    rec_list = list(map(lambda k: 
                        data[k][vae_res]['recall'][ni] if 'vae' in k else \
                        data[k][gan_res]['recall'][ni], data.keys()))    
            
                    prec_min = np.round(min(prec_list) - 0.5 * 10**(-2), 2)
                    prec_max = np.round(max(prec_list) + 0.5 * 10**(-2), 2)
                    rec_min = np.round(min(rec_list) - 0.5 * 10**(-2), 2)
                    rec_max = np.round(max(rec_list) + 0.5 * 10**(-2), 2)
                    
                    # ------------- Plot IPR results
        #            gan_smooth = 'res_gan20'
                    xlim = (prec_min, prec_max)
                    ylim = (rec_min, rec_max)
                    limit_axis = True
                    plt.subplot(2, 2, 1)
                    for model_name in data.keys():
                        label = '{2}_n{0}_s{1}'.format(nhood, s, model_name)
                        if model_name in vae_group1:
                            x = data[model_name][vae_res]['precision'][ni]
                            y = data[model_name][vae_res]['recall'][ni]
                            plt.scatter(x, y, alpha=0.7, label=model_name, marker='D')
#                    if show_legend:
                    plt.legend(loc='upper left')
                    if limit_axis:
                        plt.xlim((0.95, 1.001))
                        plt.ylim((0.4, 0.44))
#                    pyplot.locator_params(axis='y', nbins=5)
                    plt.yticks(np.arange(0.4, 0.45, 0.01))
                    
                    plt.ylabel('recall')
                    
                    plt.subplot(2, 2, 2)
                    for model_name in data.keys():
                        if model_name in vae_group2:
                            label = '{2}_n{0}_s{1}'.format(nhood, s, model_name)
                            x = data[model_name][vae_res]['precision'][ni]
                            y = data[model_name][vae_res]['recall'][ni]
                            plt.scatter(x, y, alpha=0.7, label=model_name, marker='D')
#                    if show_legend:
                    plt.legend(loc='upper left')
                    if limit_axis:
                        plt.xlim((0.95, 1.001))
                        plt.ylim((0.4, 0.44))
                    plt.yticks(np.arange(0.4, 0.45, 0.01))
                    
                    plt.subplot(2, 2, 3)
                    for model_name in data.keys():
                        if model_name in gan_group1:
#                            label = '{2}_n{0}_s{1}'.format(nhood, s, model_name)
                            x = data[model_name][gan_res]['precision'][ni]
                            y = data[model_name][gan_res]['recall'][ni]
                            plt.scatter(x, y, alpha=0.7, label=model_name, marker='D')
#                    if show_legend:
                    plt.legend(loc='lower left')
                    if limit_axis:
                        plt.xlim((0.95-0.001, 1.001))
                        plt.ylim((0.4-0.006, 1.001))
#                    plt.xticks(np.arange(0.95, 1.0, 0.025))
                    plt.xlabel('precision')
                    plt.ylabel('recall')
                    
                    plt.subplot(2, 2, 4)
                    for model_name in data.keys():
                        if model_name in gan_group2:
                            label = '{2}_n{0}_s{1}'.format(nhood, s, model_name)
                            x = data[model_name][gan_res]['precision'][ni]
                            y = data[model_name][gan_res]['recall'][ni]
                            plt.scatter(x, y, alpha=0.7, label=model_name, marker='D')
#                    if show_legend:
                    plt.legend(loc='lower left')
                    if limit_axis:
                        plt.xlim((0.95-0.001, 1.001))
                        plt.ylim((0.4-0.006, 1.001))
                    plt.xlabel('precision')
                    
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.3, wspace=0.3)
            plt.savefig('test_pr/IPR_nhood{0}_s{1}_n_samples{2}'.format(
                    nhood, s, n_samples))
            plt.show()
            
            
        def plot_agg_gan_vae_ipr(nhood, s):
#            nhoods = ['5']#['', '5', '12', '20']
#            smooth = ['0', '10']
            
#            vae_res = 'res{0}_vae'.format(nhood)
#            gan_res = 'res{0}_gan{1}'.format(nhood, s)
            
            ni = [20, 25, 30, 50].index(int(nhood))
            vae_res = 'res0_vae'
            gan_res = 'res10_gan'
            s = 10
    #        agg_fn = lambda x, y: np.sqrt(x**2 + y**2)
            agg_fn = lambda x, y: x + y
            fig2 = plt.figure(9, figsize=(10, 10))
            plt.clf()
            gs = fig2.add_gridspec(2, 2)#, width_ratios=[1, 2])
            
#            fig2.suptitle('Aggregated IPR scores, nhood = {0}, gan_smoothening = {1}, n_samples = {2}'.format(
#                    nhood, s, n_samples)) #fontsize=16)
            
            vae_agg_results_g1 = []
            vae_agg_results_g2 = []
            gan_agg_results_g1 = []
            gan_agg_results_g2 = []
            for model_name in data.keys():
                if 'vae' in model_name:
                    x = data[model_name][vae_res]['precision'][ni]
                    y = data[model_name][vae_res]['recall'][ni]
                   
                    if model_name in vae_group1:
                        vae_agg_results_g1.append((model_name, agg_fn(x, y)))
                    else:
                        vae_agg_results_g2.append((model_name, agg_fn(x, y)))
                else:
                    
                    x = data[model_name][gan_res]['precision'][ni]
                    y = data[model_name][gan_res]['recall'][ni]

                    if model_name in gan_group1:
                        gan_agg_results_g1.append((model_name, agg_fn(x, y)))
                    else:
                        gan_agg_results_g2.append((model_name, agg_fn(x, y)))

            vae_agg_results_g1 = sorted(vae_agg_results_g1, key=lambda x: x[1], reverse=True)
            vae_agg_results_g2 = sorted(vae_agg_results_g2, key=lambda x: x[1], reverse=True)
            vae_agg_g1_names, vae_agg_g1_res = zip(*vae_agg_results_g1)
            vae_agg_g2_names, vae_agg_g2_res = zip(*vae_agg_results_g2)
            
            gan_agg_results_g1 = sorted(gan_agg_results_g1, key=lambda x: x[1], reverse=True)
            gan_agg_results_g2 = sorted(gan_agg_results_g2, key=lambda x: x[1], reverse=True)
            gan_agg_g1_names, gan_agg_g1_res = zip(*gan_agg_results_g1)
            gan_agg_g2_names, gan_agg_g2_res = zip(*gan_agg_results_g2)
            
            xlim = (0, max(list(vae_agg_g1_res) + list(vae_agg_g2_res) +
                           list(gan_agg_g1_res) + list(gan_agg_g2_res)) + 0.1)
            f2_ax1 = fig2.add_subplot(gs[0, 0])
            f2_ax1.set_yticks(np.arange(len(vae_agg_g1_names)) + 1)
            f2_ax1.set_yticklabels(vae_agg_g1_names[::-1])
            f2_ax1.set_xlim(xlim)
            
            f2_ax2 = fig2.add_subplot(gs[0, 1])
            f2_ax2.set_yticks(np.arange(len(vae_agg_g2_names)) + 1)
            f2_ax2.set_yticklabels(vae_agg_g2_names[::-1])
            f2_ax2.set_xlim(xlim)
     
            f2_ax3 = fig2.add_subplot(gs[1, 0])
            f2_ax3.set_yticks(np.arange(len(gan_agg_g1_names)) + 1)
            f2_ax3.set_yticklabels(gan_agg_g1_names[::-1])
            f2_ax3.set_xlim(xlim)
            f2_ax3.set_xlabel('Aggregated PR score')
            
            f2_ax4 = fig2.add_subplot(gs[1, 1])
            f2_ax4.set_yticks(np.arange(len(gan_agg_g2_names)) + 1)
            f2_ax4.set_yticklabels(gan_agg_g2_names[::-1])
            f2_ax4.set_xlim(xlim)
            f2_ax4.set_xlabel('Aggregated PR score')
            
            for model_name in data.keys():
                if model_name in vae_group1:
                    agg_names = vae_agg_g1_names
                    agg_res = vae_agg_g1_res
                    ax = f2_ax1
                elif model_name in vae_group2:
                    agg_names = vae_agg_g2_names
                    agg_res = vae_agg_g2_res
                    ax = f2_ax2
                elif model_name in gan_group1:
                    agg_names = gan_agg_g1_names
                    agg_res = gan_agg_g1_res
                    ax = f2_ax3
                else:
                    agg_names = gan_agg_g2_names
                    agg_res = gan_agg_g2_res
                    ax = f2_ax4
                    
                i = len(agg_names) - agg_names.index(model_name)
                res = agg_res[agg_names.index(model_name)]
                
                ax.barh(i, res, align='center', label=model_name, 
                            height=0.5)
                ax.legend(loc='upper left', framealpha=1)
                    
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.3, wspace=0.3)
            plt.savefig('test_pr/AggIPR_nhood{0}_s{1}_n_samples{2}'.format(
                    nhood, s, n_samples))
            plt.show()
            
                         
    
    if evaluate:
        max_ind = 15750
        
        for n_points in [2000, 5000, 10000, 15000]:
            print('Chosen n_points: ', n_points)
            nhood_sizes = [3, 20, 25, 30, 50, 100]
            base = np.random.choice(max_ind, n_points, replace=False)
            ref_np = get_ref_samples(base)
            
            final_dict = {}
            for model, ld in yumi_gan_models.items():
                print('InfoGAN model with ld: ', model, ld)
                infogan_eval_np = get_infogan_samples(model, ld, n_points)

                infogan_eval_np_avg5 = moving_average(infogan_eval_np, n=5)                
                infogan_eval_np_avg10 = moving_average(infogan_eval_np, n=10)
                infogan_eval_np_avg15 = moving_average(infogan_eval_np, n=15)
                
                print('Starting to calculate InfoGAN PR....')
                sess = tf.Session()
                with sess.as_default():
                    res0_gan = iprd.knn_precision_recall_features(
                                ref_np.reshape(-1, 7*79), 
                                infogan_eval_np.reshape(-1, 7*79), 
                                nhood_sizes=nhood_sizes,
                                row_batch_size=500, col_batch_size=100, num_gpus=1)
                    
                    
                    res5_gan = iprd.knn_precision_recall_features(
                                ref_np.reshape(-1, 7*79), 
                                infogan_eval_np_avg5.reshape(-1, 7*79), 
                                nhood_sizes=nhood_sizes,
                                row_batch_size=500, col_batch_size=100, num_gpus=1)
                    
                    res10_gan = iprd.knn_precision_recall_features(
                                ref_np.reshape(-1, 7*79), 
                                infogan_eval_np_avg10.reshape(-1, 7*79), 
                                nhood_sizes=nhood_sizes,
                                row_batch_size=500, col_batch_size=100, num_gpus=1)
                    
                    res15_gan = iprd.knn_precision_recall_features(
                                ref_np.reshape(-1, 7*79), 
                                infogan_eval_np_avg15.reshape(-1, 7*79), 
                                nhood_sizes=nhood_sizes,
                                row_batch_size=500, col_batch_size=100, num_gpus=1)
                    
                    
                final_dict[model] = {
                        'res0_gan': res0_gan,
                        'res5_gan': res5_gan,
                        'res10_gan': res10_gan,
                        'res15_gan': res15_gan,
                        'nhoods': nhood_sizes,
                        'npoints': n_points
                        }
                    
            #_nhood file: res15 & res keys [5, 10, 25, 50]
            for model, ld in yumi_vae_models.items():
                print('VAE model with ld: ', model, ld)
                vae_eval_np = get_vae_samples(model, ld, n_points)
    
                print('Starting to calculate InfoGAN PR....')
                sess = tf.Session()
                with sess.as_default():               
                    res0_vae = iprd.knn_precision_recall_features(
                                ref_np.reshape(-1, 7*79), 
                                vae_eval_np.reshape(-1, 7*79), 
                                nhood_sizes=nhood_sizes,
                                row_batch_size=500, col_batch_size=100, num_gpus=1)
            
                final_dict[model] = {
                    'res0_vae': res0_vae,
                    'nhoods': nhood_sizes,
                    'npoints': n_points}
                
            print('Results ready ', final_dict)
            with open('test_pr/ipr_results_nhood_{0}samples.pkl'.format(n_points), 'wb') as f:
                pickle.dump(final_dict, f)