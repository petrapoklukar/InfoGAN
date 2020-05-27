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
import matplotlib.pyplot as plt
import pickle
import iprd_score as iprd 
import tensorflow as tf


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
        descaled = (unit_scaled.reshape(-1, 7, 79)) * (self.max - self.min) + self.min
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
        print(z_noise.shape)
        print(z_noise.dtype)
        print(type(z_noise))
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
    evaluate = False
    analyse = True
    
    if analyse:
        with open('test_pr/ipr_results_7500samples.pkl', 'rb') as f:
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
    
        def plot_ipr():
            # ------------- Plot IPR results
            plt.figure(12)
            plt.clf()
            plt.suptitle('Improved PR scores')
            gan_smooth = 'res_gan20'
            vae_res = 'res_vae'
            xlim = (0.5, 1)
            ylim = (0.2, 0.7)
            limit_axis = False
            plt.subplot(2, 2, 1)
            for model_name in data.keys():
                if model_name in vae_group1:
                    x = data[model_name][vae_res]['precision']
                    y = data[model_name][vae_res]['recall']
                    plt.scatter(x, y, alpha=0.7, label=model_name, marker='D')
            plt.legend()
            if limit_axis:
                plt.xlim(xlim)
                plt.ylim(ylim)
            plt.ylabel('disentangling recall')
            
            plt.subplot(2, 2, 2)
            for model_name in data.keys():
                if model_name in vae_group2:
                    x = data[model_name][vae_res]['precision']
                    y = data[model_name][vae_res]['recall']
                    plt.scatter(x, y, alpha=0.7, label=model_name, marker='D')
            plt.legend()
            if limit_axis:
                plt.xlim(xlim)
                plt.ylim(ylim)
            
            plt.subplot(2, 2, 3)
            for model_name in data.keys():
                if model_name in gan_group1:
                    x = data[model_name][gan_smooth]['precision']
                    y = data[model_name][gan_smooth]['recall']
                    plt.scatter(x, y, alpha=0.7, label=model_name, marker='D')
            plt.legend()
            if limit_axis:
                plt.xlim(xlim)
                plt.ylim(ylim)
            plt.xlabel('disentangling precision')
            plt.ylabel('disentangling recall')
            
            plt.subplot(2, 2, 4)
            for model_name in data.keys():
                if model_name in gan_group2:
                    x = data[model_name][gan_smooth]['precision']
                    y = data[model_name][gan_smooth]['recall']
                    plt.scatter(x, y, alpha=0.7, label=model_name, marker='D')
            plt.legend()
            if limit_axis:
                plt.xlim(xlim)
                plt.ylim(ylim)
            plt.ylabel('disentangling recall')
            
            plt.subplots_adjust(hspace=0.5)
            plt.show()
    
    
    
    if evaluate:
        max_ind = 10000
        
        for n_points in [750, 1000, 5000, 7500]:
            print('Chosen n_points: ', n_points)
            base = np.random.choice(max_ind, n_points, replace=False)
            ref_np = get_ref_samples(base)
            
            final_dict = {}
            for model, ld in yumi_gan_models.items():
                print('InfoGAN model with ld: ', model, ld)
                infogan_eval_np = get_infogan_samples(model, ld, n_points)
                
                infogan_eval_np_avg20 = moving_average(infogan_eval_np, n=20)
                infogan_eval_np_avg15 = moving_average(infogan_eval_np, n=15)
                infogan_eval_np_avg10 = moving_average(infogan_eval_np, n=10)
                infogan_eval_np_avg5 = moving_average(infogan_eval_np, n=5)
                
                print('Starting to calculate InfoGAN PR....')
                sess = tf.Session()
                with sess.as_default():
                    res_gan = iprd.knn_precision_recall_features(
                                ref_np.reshape(-1, 7*79), 
                                infogan_eval_np.reshape(-1, 7*79), nhood_sizes=[3],
                                row_batch_size=500, col_batch_size=100, num_gpus=1)
                    
                    res_gan5 = iprd.knn_precision_recall_features(
                                ref_np.reshape(-1, 7*79), 
                                infogan_eval_np_avg5.reshape(-1, 7*79), nhood_sizes=[3],
                                row_batch_size=500, col_batch_size=100, num_gpus=1)
                    
                    res_gan10 = iprd.knn_precision_recall_features(
                                ref_np.reshape(-1, 7*79), 
                                infogan_eval_np_avg10.reshape(-1, 7*79), nhood_sizes=[3],
                                row_batch_size=500, col_batch_size=100, num_gpus=1)
                    
                    res_gan15 = iprd.knn_precision_recall_features(
                                ref_np.reshape(-1, 7*79), 
                                infogan_eval_np_avg15.reshape(-1, 7*79), nhood_sizes=[3],
                                row_batch_size=500, col_batch_size=100, num_gpus=1)
                    
                    res_gan20 = iprd.knn_precision_recall_features(
                                ref_np.reshape(-1, 7*79), 
                                infogan_eval_np_avg20.reshape(-1, 7*79), nhood_sizes=[3],
                                row_batch_size=500, col_batch_size=100, num_gpus=1)
                    
                    res5_gan15 = iprd.knn_precision_recall_features(
                                ref_np.reshape(-1, 7*79), 
                                infogan_eval_np_avg15.reshape(-1, 7*79), nhood_sizes=[5],
                                row_batch_size=500, col_batch_size=100, num_gpus=1)
                    
                    res5_gan20 = iprd.knn_precision_recall_features(
                                ref_np.reshape(-1, 7*79), 
                                infogan_eval_np_avg20.reshape(-1, 7*79), nhood_sizes=[5],
                                row_batch_size=500, col_batch_size=100, num_gpus=1)
                    
                final_dict[model] = {
                        'res_gan': res_gan,
                        'res_gan5': res_gan5,
                        'res_gan10': res_gan10,
                        'res_gan15': res_gan15,
                        'res_gan20': res_gan20,
                        'res5_gan15': res5_gan15,
                        'res5_gan20': res5_gan20,
                        }
                    
            
            for model, ld in yumi_vae_models.items():
                print('VAE model with ld: ', model, ld)
                vae_eval_np = get_vae_samples('vae1', 2, n_points)
    
                print('Starting to calculate InfoGAN PR....')
                sess = tf.Session()
                with sess.as_default():     
                    res_vae = iprd.knn_precision_recall_features(
                                    ref_np.reshape(-1, 7*79), 
                                    vae_eval_np.reshape(-1, 7*79), nhood_sizes=[3],
                                    row_batch_size=500, col_batch_size=100, num_gpus=1)
            
                final_dict[model] = {'res_vae': res_vae}
                
            print('Results ready ', final_dict)
            with open('test_pr/ipr_results_{0}samples.pkl'.format(n_points), 'wb') as f:
                pickle.dump(final_dict, f)