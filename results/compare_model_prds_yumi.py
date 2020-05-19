#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:42:14 2020

@author: petrapoklukar
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from importlib.machinery import SourceFileLoader
import os
import argparse
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
            self.data = torch.from_numpy(self.data_scaled).float()
        else: 
            self.data = torch.from_numpy(self.np_data)
        self.num_samples = self.data.shape[0]                
    
    def get_subset(self, max_ind, n_points, fixed_indices=None, reshape=True):
        if fixed_indices is None:
            self.prd_indices = np.random.choice(max_ind, n_points, replace=False)
        else:
            self.prd_indices = fixed_indices
        subset_list = [self.data[i].numpy() for i in self.prd_indices]
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
    

def get_model_names(filename='prd_models.txt'):
    with open('prd_models.txt') as f:
        lines = f.read().splitlines()
        all_models = [list(group) for k, group in groupby(lines, lambda x: x == "") \
                      if not k]
        yumi_models = all_models[-1]
    return yumi_models


def compare_gan_prd(configs, baseline_indices=None, random_seed=1201, 
                    chpnt_list=[], color_list=[]):
    """Calculates PRD for a given list of models."""
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    prd_dict = {config_name: {
        'prd_data': [], 'F8_data': [], 'prd_names': []} for config_name in configs}
    parent_dir = '../'
    for config_name in configs:
        prd_scores = []
        prd_models = []   
        # Load model config
        config_file = os.path.join(parent_dir, 'configs', config_name + '.py')
        export_directory = os.path.join(parent_dir, 'models', config_name)
    
        print(' *- Config name: {0}'.format(config_name))
        
        config_file = SourceFileLoader(config_name, config_file).load_module().config 
        config_file['train_config']['exp_name'] = config_name
        config_file['train_config']['exp_dir'] = export_directory # the place where logs, models, and other stuff will be stored
    
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config_file['train_config']['device'] = device
        print('Device is: {}'.format(device))
    
        # Load the trained model
        model = infogan.InfoGAN(config_file)
        eval_config = config_file['eval_config']
        eval_config['filepath'] = parent_dir + eval_config['filepath'].format(config_name)
        model.load_model(eval_config)
        print(eval_config)
    
        # Number of samples to calculate PR on
        n_prd_samples = eval_config['n_prd_samples']
        
        # Get the ground truth np array from the test split
        path_to_data = parent_dir + config_file['data_config']['path_to_data']
        test_dataset = TrajDataset(path_to_data, device, scaled=True)
        ref_np = test_dataset.get_subset(len(test_dataset), n_prd_samples,
                                                  fixed_indices=baseline_indices)
    
        # Get the sampled np array
        with torch.no_grad():
            z_noise, con_noise = model.ginput_noise(n_prd_samples)
            eval_data = model.g_forward(z_noise, con_noise)
            eval_np = eval_data.cpu().numpy().reshape(n_prd_samples, -1)
            

        # Load the chpt
        chpr_prd_list = []
        if chpnt_list != []:
            for c in chpnt_list:
                model_chpt = infogan.InfoGAN(config_file)
                model_chpt.load_checkpoint(
                    '../models/{0}/infogan_checkpoint{1}.pth'.format(config_name, c))
                model_chpt.Gnet.eval()
                
                with torch.no_grad():
                    z_noise, con_noise = model_chpt.ginput_noise(n_prd_samples)
                    eval_chnpt_data = model_chpt.g_forward(z_noise, con_noise)
                    eval_chnpt_np = eval_chnpt_data.cpu().numpy().reshape(n_prd_samples, -1)
                prd_chnpt_data = prd.compute_prd_from_embedding(eval_chnpt_np, ref_np)
                chpr_prd_list.append(prd_chnpt_data)
            
        # Compute and save prd 
        prd_data = prd.compute_prd_from_embedding(eval_np, ref_np)
        F8_data = [prd.prd_to_max_f_beta_pair(prd_data[0], prd_data[1])] + \
            list(map(lambda x: prd.prd_to_max_f_beta_pair(x[0], x[1]), 
                     chpr_prd_list))
        
        all_prd_data = [prd_data] + chpr_prd_list
        prd_scores += all_prd_data
        prd_dict[config_name]['prd_data'] = all_prd_data
        prd_dict[config_name]['F8_data'] = F8_data

        all_names = ['_'.join(config_name.split('_')[-3:])] + chpnt_list
        prd_models += all_names
        prd_dict[config_name]['prd_names'] = all_names
        
    
        # plt.figure(3)
        # plt.clf()
        # for i in range(len(F8_data)):
        #     plt.scatter(F8_data[i][0], F8_data[i][1], label=all_names[i], 
        #                 color=color_list[i])
        # plt.legend()
        # plt.title(all_names[0] + ' F8')
        # plt.xlabel('F8 (recall)')
        # plt.ylabel('F1/8 (precision)')
        # plot_name_F8 = 'prd_per_model/descaled{0}_yumi_fixedBaseline{1}_seed_{2}F8'.format(
        #     config_name, int(0 if baseline_indices is None else 1), random_seed)
        # plt.savefig(plot_name_F8)
        # plt.xlim((0, 1))
        # plt.ylim((0, 1))
        # plt.savefig(plot_name_F8 + '_normalised')
        # plt.close()
        
        # short_model_names = ['_'.join(prd_models[i].split('_')[-3:]) \
        #                      for i in range(len(prd_models))]  
        # plot_name = 'prd_per_model/descaled{0}_yumi_fixedBaseline{1}_seed_{2}prds.png'.format(
        #     config_name, int(0 if baseline_indices is None else 1), random_seed)
        # prd.plot(prd_scores, short_model_names, out_path=plot_name, color_list=color_list)
    return prd_dict


def compare_vae_prd(config_dict, baseline_indices, random_seed=1201, 
                    color_list=[]):
    """Calculates PRD for a given list of models."""
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    prd_dict = {config_name: {'prd_data': [], 'F8_data': [], 'prd_names': []} \
                for config_name in config_dict.keys()}
    parent_dir = '../'
    prd_scores = []
    for config_name, ld in config_dict.items():

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Device is: {}'.format(device))
    
        # Load the trained model
        filename = '../models/VAEs/{}.mdl'.format(config_name)
        model = models.FullyConnecteDecoder(ld, 7*79)
        model_dict = {}
        for k, v in torch.load(filename, map_location=device).items():
            if 'decoder' in k:
                k_new = '.'.join(k.split('.')[1:])
                model_dict[k_new] = v
        model.load_state_dict(model_dict)
    
        # Number of samples to calculate PR on
        n_prd_samples = len(baseline_indices)
        
        # Get the ground truth np array from the test split
        path_to_data = parent_dir + 'dataset/robot_trajectories/yumi_joint_pose.npy'
        test_dataset = TrajDataset(path_to_data, device, scaled=True)
        ref_np = test_dataset.get_subset(len(test_dataset), n_prd_samples,
                                         fixed_indices=baseline_indices)
    
        # Get the sampled np array
        with torch.no_grad():
            z_noise = torch.empty((n_prd_samples, ld), device=device).normal_()
            eval_data = model(z_noise)
            eval_np = test_dataset.scale(eval_data).reshape(n_prd_samples, -1)
            
        # Compute and save prd 
        prd_data = prd.compute_prd_from_embedding(eval_np, ref_np)
        F8_data = [prd.prd_to_max_f_beta_pair(prd_data[0], prd_data[1])] 
        
        prd_scores.append(prd_data)
        prd_dict[config_name]['prd_data'] = [prd_data]
        prd_dict[config_name]['F8_data'] = F8_data

        prd_dict[config_name]['prd_names'] = config_name
        
    # plt.figure(3)
    # plt.clf()
    # for config_name in prd_dict.keys():
    #     plt.scatter(prd_dict[config_name]['F8_data'][0][0], 
    #                 prd_dict[config_name]['F8_data'][0][1], label=config_name)
    # plt.legend()
    # plt.title('VAE F8')
    # plt.xlabel('F8 (recall)')
    # plt.ylabel('F1/8 (precision)')
    # plot_name_F8 = 'prd_per_model/vaes_yumi_fixedBaseline{0}_seed_{1}F8'.format(
    #     int(0 if baseline_indices is None else 1), random_seed)
    # plt.savefig(plot_name_F8)
    # plt.xlim((0, 1))
    # plt.ylim((0, 1))
    # plt.savefig(plot_name_F8 + '_normalised')
    # plt.close()
    

    # plot_name = 'prd_per_model/vaes_yumi_fixedBaseline{0}_seed_{1}prds.png'.format(
    #     int(0 if baseline_indices is None else 1), random_seed)
    # prd.plot(prd_scores, list(prd_dict.keys()), out_path=plot_name, color_list=color_list)
    return prd_dict
    

def index_to_ld(i):
    if i <= 3:
        return 2
    elif 4 <= i and i <= 6:
        return 3
    else:
        return 6

def infogan_name_to_index(name):
    lmbd, ld = name.split('_')[-3:-1]
    lambda_to_index = {'l01': 0, 'l15': 1, 'l35': 2}
    ld_to_index = {'s2': 1, 's3': 4, 's6': 7}
    return 'gan' + str(ld_to_index[ld] + lambda_to_index[lmbd])


def plot_dgm_prds(vae_res_dict, infogan_res_dict):
    group1 = ['vae' + str(i) for i in range(1, 6)]
    group2 = ['vae' + str(i) for i in range(6, 10)]
    group3 = ['gan' + str(i) for i in range(1, 6)]
    group4 = ['gan' + str(i) for i in range(6, 10)]
    
    ylim = (0.8, 1)
    xlim = (0.5, 1)
    ylim = (0, 1)
    xlim = (0, 1)
    plt.figure(5)
    # plt.suptitle('PRD F8')
    
    plt.subplot(2, 2, 1)
    for model in vae_res_dict.keys():
        model_name = ''.join(model.split('_'))
        if model_name in group1:
            f8 = vae_res_dict[model]['F8_data']
            plt.scatter(f8[0][0], f8[0][1], label=model_name, marker='D')
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.ylabel('F1/8 (precision)')
    plt.tick_params(axis='x', bottom=False, labelbottom=False, )
    plt.legend()
    
    plt.subplot(2, 2, 2)
    for model in vae_res_dict.keys():
        model_name = ''.join(model.split('_'))
        if model_name in group2:
            f8 = vae_res_dict[model]['F8_data']
            plt.scatter(f8[0][0], f8[0][1], label=model_name, marker='D')
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.tick_params(axis='both', left=False, top=False, right=False, 
                    bottom=False, labelleft=False, labeltop=False, 
                    labelright=False, labelbottom=False)
    plt.legend()
    
    plt.subplot(2, 2, 3)
    for model in infogan_res_dict.keys():
        model_name = infogan_name_to_index(model)
        if model_name in group3:
            f8 = infogan_res_dict[model]['F8_data']
            plt.scatter(f8[0][0], f8[0][1], label=model_name, marker='D')
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.legend()
    
    plt.subplot(2, 2, 4)
    for model in infogan_res_dict.keys():
        model_name = infogan_name_to_index(model)
        if model_name in group4:
            f8 = infogan_res_dict[model]['F8_data']
            plt.scatter(f8[0][0], f8[0][1], label=model_name, marker='D')
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.tick_params(axis='y', left=False,labelleft=False)
    plt.xlabel('F8 (recall)')
    plt.legend()
    plt.show()
    

if __name__ == '__main__':    
    
    max_ind = 10000
    n_points = 1000
    base = np.random.choice(max_ind, n_points, replace=False)
    
    yumi_gan_models = get_model_names() 
    yumi_vae_models = {'vae_' + str(i): index_to_ld(i) for i in range(1, 10)}
    
    color_list = ['black', 'gold', 'orange', 'red', 'green', 'cyan', 
                  'dodgerblue', 'blue', 'darkviolet', 'magenta', 'deeppink']
    
    # with open('vae_prd_res.pkl', 'rb') as f:
    #     vae_res_dict = pickle.load(f)
    
    # with open('infogan_prd_res.pkl', 'rb') as f:
    #     infogan_res_dict = pickle.load(f)
    # infogan_res_dict = compare_gan_prd(
    #         yumi_gan_models, baseline_indices=base, random_seed=1602, 
    #         chpnt_list=[], color_list=color_list)
    # vae_res_dict = compare_vae_prd(yumi_vae_models, base, random_seed=1201, 
    #                                    color_list=color_list)
    plot_dgm_prds(vae_res_dict, infogan_res_dict)
    
    
    
    
    if False:
        vae_res_dict = compare_vae_prd(yumi_vae_models, base, random_seed=1201, 
                                       color_list=color_list)
        
        with open('vae_prd_res_descaled.pkl', 'wb') as f:
                pickle.dump(vae_res_dict, f)
                
        infogan_res_dict = compare_gan_prd(
            yumi_gan_models, baseline_indices=base, random_seed=1602, 
            chpnt_list=[], color_list=color_list)
            
        with open('infogan_prd_res_descled.pkl', 'wb') as f:
            pickle.dump(infogan_res_dict, f)
    
    if False:
        infogan_res_dict = compare_gan_prd(
            yumi_gan_models, baseline_indices=base, random_seed=1602, 
            chpnt_list=[str(i) for i in range(99, 1000, 100)], color_list=color_list)
            
        with open('infogan_prd_res.pkl', 'wb') as f:
            pickle.dump(infogan_res_dict, f)
    
    if False:
        plt.figure(1)
        plt.clf()
        chpnt = 2
        plt.suptitle('InfoGAN PRD checkpoint 199')
        plt.subplot(1, 2, 1)
        for model in infogan_res_dict.keys():
            model_name = '_'.join(model.split('_')[-3:-1])
            precision, recall = infogan_res_dict[model]['prd_data'][chpnt]
            plt.plot(recall, precision, label=model_name, alpha=0.5, linewidth=3)
                     # color=color_list[chpnt])
            plt.legend()
        
        plt.subplot(1, 2, 2)
        for model in infogan_res_dict.keys():
            model_name = '_'.join(model.split('_')[-3:-1])
            F8_data = infogan_res_dict[model]['F8_data']
            plt.scatter(F8_data[chpnt][0], F8_data[chpnt][1], alpha=0.5)
            plt.xlabel('F8 (recall)')
            plt.ylabel('F1/8 (precision)')
            plt.xlim((0, 1))
            plt.ylim((0, 1))
        plt.show()
        
        
        plt.figure(2)
        plt.clf()
        chpnt = 0
        plt.suptitle('InfoGAN PRD')
    
        for model in infogan_res_dict.keys():
            model_name = '_'.join(model.split('_')[-3:-1])
            if 'l15' in model_name:
                F8_data = infogan_res_dict[model]['F8_data']
                plt.scatter(F8_data[chpnt][0], F8_data[chpnt][1],label=model_name)
                plt.legend()
                plt.xlabel('F8 (recall)')
                plt.ylabel('F1/8 (precision)')
                # plt.xlim((0, 1))
                # plt.ylim((0, 1))
        plt.show()
        
        