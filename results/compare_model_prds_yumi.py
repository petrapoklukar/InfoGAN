#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:42:14 2020

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
    prd_models = []   
    prd_scores = []
    for config_name in configs:
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
        # print(eval_config)
    
        # Number of samples to calculate PR on
        n_prd_samples = eval_config['n_prd_samples']
        
        # Get the ground truth np array from the test split
        path_to_data = parent_dir + config_file['data_config']['path_to_data']
        test_dataset = TrajDataset(path_to_data, device, scaled=False)
        ref_np = test_dataset.get_subset(len(test_dataset), n_prd_samples,
                                         fixed_indices=baseline_indices)
    
        test_dataset_gan = TrajDataset(path_to_data, device, scaled=True)
        ref_gan_np = test_dataset_gan.get_subset(len(test_dataset_gan), n_prd_samples,
                                         fixed_indices=baseline_indices)
        
        # Get the sampled np array
        with torch.no_grad():
            z_noise, con_noise = model.ginput_noise(n_prd_samples)
            eval_data = model.g_forward(z_noise, con_noise)
            
            eval_gan_np = eval_data.numpy().reshape(n_prd_samples, 7, 79)      
            eval_np = test_dataset.descale(eval_data).reshape(n_prd_samples, 7, 79)

            
            
        # Load the chpt
        chpr_prd_list = []
        if chpnt_list != []:
            print('checkpoints: ', chpnt_list)
            for c in chpnt_list:
                model_chpt = infogan.InfoGAN(config_file)
                model_chpt.load_checkpoint(
                    '../models/{0}/infogan_checkpoint{1}.pth'.format(config_name, c))
                model_chpt.Gnet.eval()
                
                with torch.no_grad():
                    z_noise, con_noise = model_chpt.ginput_noise(n_prd_samples)
                    eval_chnpt_data = model_chpt.g_forward(z_noise, con_noise)
                    eval_chnpt_np = eval_chnpt_data.cpu().numpy()
                    eval_chnpt_smooth_np = moving_average(eval_chnpt_np, n=20).reshape(n_prd_samples, -1)
                prd_chnpt_data = prd.compute_prd_from_embedding(eval_chnpt_smooth_np, ref_np)
                chpr_prd_list.append(prd_chnpt_data)
            
        # Compute and save prd 
        prd_data_normal = prd.compute_prd_from_embedding(eval_np.reshape(n_prd_samples, -1), ref_np)
        print('n:', prd.prd_to_max_f_beta_pair(prd_data_normal[0], prd_data_normal[1]))
        
        for i in range(2, 20):
            eval_np_smooth = moving_average(eval_np, n=i).reshape(n_prd_samples, -1)
            prec, rec = prd.compute_prd_from_embedding(eval_np_smooth, ref_np)
            print('s' + str(i) + ':', prd.prd_to_max_f_beta_pair(prec, rec))
        
        print('\n')
        prd_data_normal = prd.compute_prd_from_embedding(eval_gan_np.reshape(n_prd_samples, -1), ref_gan_np)
        print('n:', prd.prd_to_max_f_beta_pair(prd_data_normal[0], prd_data_normal[1]))
        
        for i in range(2, 20):
            eval_gan_np_smooth = moving_average(eval_gan_np, n=i).reshape(n_prd_samples, -1)
            prec, rec = prd.compute_prd_from_embedding(eval_gan_np_smooth, ref_gan_np)
            print('s' + str(i) + ':', prd.prd_to_max_f_beta_pair(prec, rec))
        
        # prd_data = prd.compute_prd_from_embedding(eval_np_smooth, ref_np)
        # F8_data = [prd.prd_to_max_f_beta_pair(prd_data[0], prd_data[1])] + \
        #     list(map(lambda x: prd.prd_to_max_f_beta_pair(x[0], x[1]), 
        #              chpr_prd_list))
        # print(F8_data)
        # all_prd_data = [prd_data] + chpr_prd_list
        # prd_scores += all_prd_data
        # prd_dict[config_name]['prd_data'] = all_prd_data
        # prd_dict[config_name]['F8_data'] = F8_data

        # infogan_name = infogan_name_to_index(config_name)
        # all_names = [infogan_name] + chpnt_list
        # prd_models += all_names
        # prd_dict[config_name]['prd_names'] = all_names
        
    
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
    plt.figure(3)
    plt.clf()
    for config_name in prd_dict.keys():
        label = prd_dict[config_name]['prd_names'][0]
        plt.scatter(prd_dict[config_name]['F8_data'][0][0], 
                    prd_dict[config_name]['F8_data'][0][1], label=label)
    plt.legend()
    plt.title('InfoGAN F8')
    plt.xlabel('F8 (recall)')
    plt.ylabel('F1/8 (precision)')
    plot_name_F8 = 'prd_per_model/infogans_yumi_fixedBaseline{0}_seed_{1}F8'.format(
        int(0 if baseline_indices is None else 1), random_seed)
    plt.savefig(plot_name_F8)
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.savefig(plot_name_F8 + '_normalised')
    plt.close()
    
    plot_name = 'prd_per_model/infogans_yumi_fixedBaseline{0}_seed_{1}prds.png'.format(
        int(0 if baseline_indices is None else 1), random_seed)
    prd.plot(prd_scores, list(prd_dict.keys()), out_path=plot_name, color_list=color_list)
    
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
        test_dataset = TrajDataset(path_to_data, device, scaled=False)
        ref_np = test_dataset.get_subset(len(test_dataset), n_prd_samples,
                                         fixed_indices=baseline_indices)
    
        # Get the sampled np array
        with torch.no_grad():
            z_noise = torch.empty((n_prd_samples, ld), device=device).normal_()
            eval_data = model(z_noise)
            eval_np = eval_data.reshape(n_prd_samples, -1)
            
        # Compute and save prd 
        prd_data = prd.compute_prd_from_embedding(eval_np, ref_np)
        F8_data = [prd.prd_to_max_f_beta_pair(prd_data[0], prd_data[1])] 
        
        prd_scores.append(prd_data)
        prd_dict[config_name]['prd_data'] = [prd_data]
        prd_dict[config_name]['F8_data'] = F8_data

        vae_name = ''.join(config_name.split('_'))
        prd_dict[config_name]['prd_names'] = vae_name
        
    plt.figure(3)
    plt.clf()
    for config_name in prd_dict.keys():
        label = prd_dict[config_name]['prd_names']
        plt.scatter(prd_dict[config_name]['F8_data'][0][0], 
                    prd_dict[config_name]['F8_data'][0][1], label=label)
    plt.legend()
    plt.title('VAE F8')
    plt.xlabel('F8 (recall)')
    plt.ylabel('F1/8 (precision)')
    plot_name_F8 = 'prd_per_model/vaes_yumi_fixedBaseline{0}_seed_{1}F8'.format(
        int(0 if baseline_indices is None else 1), random_seed)
    plt.savefig(plot_name_F8)
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.savefig(plot_name_F8 + '_normalised')
    plt.close()
    

    plot_name = 'prd_per_model/vaes_yumi_fixedBaseline{0}_seed_{1}prds.png'.format(
        int(0 if baseline_indices is None else 1), random_seed)
    prd.plot(prd_scores, list(prd_dict.keys()), out_path=plot_name, 
             color_list=color_list)
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
    xlim = (0.55, 1)
    # ylim = (0, 1)
    # xlim = (0, 1)
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
    


def get_ref_samples(baseline_indices):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_prd_samples = len(baseline_indices)
    path_to_data = '../dataset/robot_trajectories/yumi_joint_pose.npy'
    test_dataset = TrajDataset(path_to_data, device, scaled=False)
    ref_np = test_dataset.get_subset(
        len(test_dataset), n_prd_samples, fixed_indices=baseline_indices, 
        reshape=False)
    return ref_np
    

def get_vae_samples(config_name, ld, n_prd_samples):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the trained model
    filename = '../models/models/{}.mdl'.format(config_name)
    model = models.FullyConnecteDecoder(ld, 7*79)
    model_dict = {}
    for k, v in torch.load(filename, map_location=device).items():
        if 'decoder' in k:
            k_new = '.'.join(k.split('.')[1:])
            model_dict[k_new] = v
    model.load_state_dict(model_dict)

    # Get the sampled np array
    with torch.no_grad():
        z_noise = torch.empty((n_prd_samples, ld), device=device).normal_()
        eval_np = model(z_noise).detach().numpy().reshape(-1, 7, 79)
    return eval_np
    

def get_infogan_samples(config_name, ld, n_prd_samples, chpnt=''):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the trained model
    model = models.FullyConnecteDecoder(ld, 7*79)
    filename = '../models/models/{0}.pt'.format(config_name)
    model_dict = torch.load(filename, map_location=device)
    model.load_state_dict(model_dict)
    model = model.to(device)
    
#    parent_dir = '../'
#    config_file = os.path.join(parent_dir, 'configs', config_name + '.py')
#    export_directory = os.path.join(parent_dir, 'models', config_name)
#
#    print(' *- Config name: {0}'.format(config_name))
#    
#    config_file = SourceFileLoader(config_name, config_file).load_module().config 
#    config_file['train_config']['exp_name'] = config_name
#    config_file['train_config']['exp_dir'] = export_directory # the place where logs, models, and other stuff will be stored
#
#    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#    config_file['train_config']['device'] = device
#    print('Device is: {}'.format(device))
#
#    # Load the trained model
#    model = infogan.InfoGAN(config_file)
#    eval_config = config_file['eval_config']
#    eval_config['filepath'] = parent_dir + eval_config['filepath'].format(config_name)
#    
#    if chpnt:
#        model.load_checkpoint(
#            '../models/{0}/infogan_checkpoint{1}.pth'.format(config_name, chpnt))
#        model.Gnet.eval()
#
#    else:
#        model.load_model(eval_config)
#        print(eval_config)
        
    # Get the ground truth np array from the test split
#    path_to_data = parent_dir + config_file['data_config']['path_to_data']
    path_to_data = '../dataset/robot_trajectories/yumi_joint_pose.npy'
    test_dataset = TrajDataset(path_to_data, device, scaled=False)

    # Get the sampled np array
    with torch.no_grad():
        z_noise = torch.empty((n_prd_samples, ld), device=device).uniform_(-1, 1)
        eval_data = model(z_noise).detach()
        eval_np = test_dataset.descale(eval_data).reshape(-1, 7, 79)
        
#        z_noise, con_noise = model.ginput_noise(n_prd_samples)
#        eval_data = model.g_forward(z_noise, con_noise)
#        eval_np = test_dataset.descale(eval_data).reshape(-1, 7, 79)
        eval_np_smooth = moving_average(eval_np, n=10)
    return eval_np_smooth, eval_np


def plot_example_traj(vae_ref_np, vae_eval_np, infogan_ref_np, infogan_eval_np):
        
    plt.figure(1)
    plt.clf()
    
    plt.subplot(2, 2, 1)
    mean = np.mean(vae_ref_np, axis=0)
    std = np.std(vae_ref_np, axis=0)
    for i in range(7):
        x = np.arange(len(mean[:, i]))
        plt.plot(x, mean[:, i], )
        plt.fill_between(x, mean[:, i] - std[:, i], 
                         mean[:, i] + std[:, i], alpha=.1)
    plt.title('VAE ref')
    
    plt.subplot(2, 2, 2)
    mean = np.mean(vae_eval_np, axis=0)
    std = np.std(vae_eval_np, axis=0)
    for i in range(7):
        x = np.arange(len(mean[:, i]))
        plt.plot(x, mean[:, i], )
        plt.fill_between(x, mean[:, i] - std[:, i], 
                         mean[:, i] + std[:, i], alpha=.1)
    plt.title('VAE eval')
    
    plt.subplot(2, 2, 3)
    mean = np.mean(infogan_ref_np, axis=0)
    std = np.std(infogan_ref_np, axis=0)
    for i in range(7):
        x = np.arange(len(mean[:, i]))
        plt.plot(x, mean[:, i], )
        plt.fill_between(x, mean[:, i] - std[:, i], 
                         mean[:, i] + std[:, i], alpha=.1)
    plt.title('InfoGAN ref')
    
    plt.subplot(2, 2, 4)
    mean = np.mean(infogan_eval_np, axis=0)
    std = np.std(infogan_eval_np, axis=0)
    for i in range(7):
        x = np.arange(len(mean[:, i]))
        # N = 20
        # y_padded = np.pad(mean[:, i], (N//2, N-1-N//2), mode='edge')
        # y_smooth = np.convolve(y_padded, np.ones((N,))/N, mode='valid') 
        plt.plot(x, mean[:, i], )
        # plt.plot(x, y_smooth, )
        plt.fill_between(x, mean[:, i] - std[:, i], 
                         mean[:, i] + std[:, i], alpha=.1)
    plt.title('InfoGAN eval')  
        
    plt.show()
    
def moving_average_mean(x, n):
    axis = 0
    x_padded = np.pad(x, ((n//2, n-1-n//2), (0, 0)), mode='edge')
    x_smooth = np.apply_along_axis(lambda ax: np.convolve(ax, np.ones((n,))/n, mode='valid'), 
                                   axis, x_padded)
    return x_smooth
    
def moving_average(x, n):
    axis = 2
    y_padded = np.pad(x, ((0, 0), (0, 0), (n//2, n-1-n//2)), mode='edge')
    y_smooth = np.apply_along_axis(
            lambda ax: np.convolve(ax, np.ones((n,))/n, mode='valid'), 
            axis, y_padded)
    return y_smooth 

def moving_average_plot(x, n):
    y_padded = np.pad(x, ((0, 0), (n//2, n-1-n//2), (0, 0)), mode='edge')
    axis = 1
    y_smooth = np.apply_along_axis(lambda ax: np.convolve(ax, np.ones((n,))/n, mode='valid'), 
                                   axis, y_padded)
    return y_smooth 



if __name__ == '__main__':    
    
    max_ind = 10000
    n_points = 2000
    base = np.random.choice(max_ind, n_points, replace=False)
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
    
#    yumi_gan_models = get_model_names() 
#    yumi_vae_models = {'vae_' + str(i): index_to_ld(i) for i in range(1, 10)}
    
    color_list = ['black', 'gold', 'orange', 'red', 'green', 'cyan', 
                  'dodgerblue', 'blue', 'darkviolet', 'magenta', 'deeppink']
    
    # with open('vae_prd_res.pkl', 'rb') as f:
    #     vae_res_dict = pickle.load(f)
    
    # with open('infogan_prd_res_descled.pkl', 'rb') as f:
    #     infogan_res_dict = pickle.load(f)

    # plot_dgm_prds(vae_res_dict, infogan_res_dict)
    
    if False:
        ref_np = get_ref_samples(base).reshape(-1, 7*79)
    
        data_o = {}
        for model, ld in yumi_gan_models.items():
            infogan_eval_np, _ = get_infogan_samples(model, ld, n_points)
            infogan_eval_np = infogan_eval_np.reshape(-1, 7*79)
            
            prec, rec = prd.compute_prd_from_embedding(
                    infogan_eval_np, ref_np, num_clusters=20) 
            f8, f18 = prd.prd_to_max_f_beta_pair(prec, rec)
            data_o[model] = {'precision': f8, 'recall': f18}
            
        for model, ld in yumi_vae_models.items():
            vae_eval_np = get_vae_samples(model, ld, n_points)
            vae_eval_np = vae_eval_np.reshape(-1, 7*79)
            
            prec, rec = prd.compute_prd_from_embedding(
                    vae_eval_np, ref_np, num_clusters=20) 
            f8, f18 = prd.prd_to_max_f_beta_pair(prec, rec)
            data_o[model] = {'precision': f8, 'recall': f18}
        
        data = {}    
        for key, value in data_o.items():
            if 'infogan' in key:
                new_key = key.split('_')[0][4:]
                data[new_key] = value
            else:
                data[key] = value
    
    if True:
        vae_group1 = ['vae' + str(i) for i in range(1, 6)]
        vae_group2 = ['vae' + str(i) for i in range(6, 10)]
        gan_group1 = ['gan' + str(i) for i in range(1, 6)]
        gan_group2 = ['gan' + str(i) for i in range(6, 10)]
        
        def plot_gan_vae_pr():
            s = 10
            clusters= 20
            
            
            plt.figure(12, figsize=(10, 10))
            plt.clf()
#            plt.suptitle('Improved PR scores')


            prec_list = list(map(lambda k: 
                data[k]['precision'] if 'vae' in k else \
                data[k]['precision'], data.keys()))
            rec_list = list(map(lambda k: 
                data[k]['recall'] if 'vae' in k else \
                data[k]['recall'], data.keys()))    
    
            prec_min = np.round(min(prec_list) - 0.5 * 10**(-2), 2)
            prec_max = np.round(max(prec_list) + 0.5 * 10**(-2), 2)
            rec_min = np.round(min(rec_list) - 0.5 * 10**(-2), 2)
            rec_max = np.round(max(rec_list) + 0.5 * 10**(-2), 2)
            
            # ------------- Plot IPR results
            xlim = (prec_min, prec_max)
            ylim = (rec_min, rec_max)
            limit_axis = False
            plt.subplot(2, 2, 1)
            for model_name in data.keys():
                label = '{2}_n{0}_s{1}'.format(clusters, s, model_name)
                if model_name in vae_group1:
                    x = data[model_name]['precision']
                    y = data[model_name]['recall']
                    plt.scatter(x, y, alpha=0.7, label=model_name, marker='D')
#                    if show_legend:
            plt.legend(loc='upper left')
            if limit_axis:
                plt.xlim((0.95, 1.001))
                plt.ylim((0.4, 0.44))
                plt.yticks(np.arange(0.4, 0.45, 0.01))
            
            plt.ylabel('recall')
            
            plt.subplot(2, 2, 2)
            for model_name in data.keys():
                if model_name in vae_group2:
                    label = '{2}_n{0}_s{1}'.format(clusters, s, model_name)
                    x = data[model_name]['precision']
                    y = data[model_name]['recall']
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
                    x = data[model_name]['precision']
                    y = data[model_name]['recall']
                    
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
                    label = '{2}_n{0}_s{1}'.format(clusters, s, model_name)
                    x = data[model_name]['precision']
                    y = data[model_name]['recall']
                    plt.scatter(x, y, alpha=0.7, label=model_name, marker='D')
#                    if show_legend:
            plt.legend(loc='lower left')
            if limit_axis:
                plt.xlim((0.95-0.001, 1.001))
                plt.ylim((0.4-0.006, 1.001))
            plt.xlabel('precision')
                    
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.3, wspace=0.3)
            plt.savefig('PR_clusters{0}_s{1}_n_samples{2}'.format(
                    clusters, s, n_points))
            plt.show()
            
            
        def plot_agg_gan_vae_pr():

            clusters = 10
            agg_fn = lambda x, y: x + y
            fig2 = plt.figure(9, figsize=(10, 10))
            plt.clf()
            gs = fig2.add_gridspec(2, 2)
            
#            fig2.suptitle('Aggregated IPR scores, nhood = {0}, gan_smoothening = {1}, n_samples = {2}'.format(
#                    nhood, s, n_samples)) #fontsize=16)
            
            vae_agg_results_g1 = []
            vae_agg_results_g2 = []
            gan_agg_results_g1 = []
            gan_agg_results_g2 = []
            for model_name in data.keys():
                x = data[model_name]['precision']
                y = data[model_name]['recall']
            
                if model_name in vae_group1:
                    vae_agg_results_g1.append((model_name, agg_fn(x, y)))
                if model_name in vae_group2:
                    vae_agg_results_g2.append((model_name, agg_fn(x, y)))
                elif model_name in gan_group1:
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
            plt.savefig('AggPR_clusters{0}_s{1}_n_samples{2}'.format(
                    clusters, 10, n_points))
            plt.show()
        
#        vae_eval_np = get_vae_samples('vae1', 2, n_points)
#        
#        plt.figure(1)
#        plt.clf()
#        plt.subplot(2, 2, 1)
#        x = np.mean(ref_np.transpose(0, 2, 1), axis=0)
#        for i in range(7):
#            plt.plot(x[:, i])
#            
#        plt.subplot(2, 2, 2)
#        x = np.mean(vae_eval_np.transpose(0, 2, 1), axis=0)
#        for i in range(7):
#            plt.plot(x[:, i])
#            
#        plt.subplot(2, 2, 3)
#        x = moving_average_mean(np.mean(infogan_eval_np.transpose(0, 2, 1), axis=0), 5)
#        for i in range(7):
#            plt.plot(x[:, i])
#            
#        plt.subplot(2, 2, 4)
#        x = np.mean(infogan_eval_np.transpose(0, 2, 1), axis=0)
#        for i in range(7):
#            plt.plot(x[:, i])
#        plt.show()
#
#        ref_np = ref_np.reshape(-1, 7*79)
#        vae_eval_np = vae_eval_np.reshape(-1, 7*79)
#        infogan_eval_np = infogan_eval_np.reshape(-1, 7*79)
#        
#        prec, rec = prd.compute_prd_from_embedding(
#            vae_eval_np, ref_np, num_clusters=20) 
#        print(prd.prd_to_max_f_beta_pair(prec, rec))
#        
#        prec, rec = prd.compute_prd_from_embedding(
#            infogan_eval_np, ref_np, num_clusters=20) 
#        print(prd.prd_to_max_f_beta_pair(prec, rec))
        
        
        
        
        

            
    
    
    
    
    
    
    
    
    if False:
        vae_res_dict = compare_vae_prd(yumi_vae_models, base, random_seed=1201, 
                                        color_list=color_list)
        
        with open('vae_prd_res.pkl', 'wb') as f:
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
        
        