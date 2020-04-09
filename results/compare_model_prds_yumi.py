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
import InfoGAN_yumi as models

class TrajDataset(Dataset):
    def __init__(self, data_filename, device=None):
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.np_data = np.load(data_filename)
        self.min = np.min(self.np_data, axis=0)
        self.max = np.max(self.np_data, axis=0)
        unit_scaled = (self.np_data - self.min) / (self.max - self.min)
        self.data_scaled = 2 * unit_scaled - 1
        self.data = torch.from_numpy(self.data_scaled).float()
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


def compare_prd(configs, baseline_indices=None, random_seed=1201):
    """Calculates PRD for a given list of models."""
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    prd_scores = []
    prd_models = []   
    parent_dir = '../'
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
        model = models.InfoGAN(config_file)
        eval_config = config_file['eval_config']
        eval_config['filepath'] = parent_dir + eval_config['filepath'].format(config_name)
        model.load_model(eval_config)
        print(eval_config)
    
        # Load the 999 chpt
        model_chpt999 = models.InfoGAN(config_file)
        model_chpt999.load_checkpoint(
            '../models/{0}/infogan_checkpoint999.pth'.format(config_name))
        model_chpt999.Gnet.eval()
        
        # Load the 1999 chpt
        model_chpt1999 = models.InfoGAN(config_file)
        model_chpt1999.load_checkpoint(
            '../models/{0}/infogan_checkpoint1999.pth'.format(config_name))
        model_chpt1999.Gnet.eval()
    
        # Number of samples to calculate PR on
        n_prd_samples = eval_config['n_prd_samples']
        
        # Get the ground truth np array from the test split
        path_to_data = parent_dir + config_file['data_config']['path_to_data']
        test_dataset = TrajDataset(path_to_data, device)
        ref_np = ref_np = test_dataset.get_subset(len(test_dataset), n_prd_samples,
                                                  fixed_indices=baseline_indices)
    
        # Get the sampled np array
        with torch.no_grad():
            z_noise, con_noise = model.ginput_noise(n_prd_samples)
            eval_data = model.g_forward(z_noise, con_noise)
            eval_np = eval_data.cpu().numpy().reshape(n_prd_samples, -1)
            
            z_noise, con_noise = model_chpt999.ginput_noise(n_prd_samples)
            eval_chnpt999_data = model_chpt999.g_forward(z_noise, con_noise)
            eval_chnpt999_np = eval_chnpt999_data.cpu().numpy().reshape(n_prd_samples, -1)
            
            z_noise, con_noise = model_chpt1999.ginput_noise(n_prd_samples)
            eval_chnpt1999_data = model_chpt1999.g_forward(z_noise, con_noise)
            eval_chnpt1999_np = eval_chnpt1999_data.cpu().numpy().reshape(n_prd_samples, -1)
        
        # Compute and save prd 
        prd_data = prd.compute_prd_from_embedding(eval_np, ref_np)
        prd_chnpt999_data = prd.compute_prd_from_embedding(eval_chnpt999_np, ref_np)
        prd_chnpt1999_data = prd.compute_prd_from_embedding(eval_chnpt1999_np, ref_np)
        prd_scores += [prd_data, prd_chnpt999_data, prd_chnpt1999_data]
        prd_models += [config_name, config_name + '_chpnt999', config_name + '_chpnt1999']
    
    short_model_names = ['_'.join(prd_models[i].split('_')[-3:]) for i in range(len(prd_models))]
    plot_name = '{}_yumi_fixedBaseline{}_seed_{}prds.png'.format(config_name,
        int(0 if baseline_indices is None else 1), random_seed)
    prd.plot(prd_scores, short_model_names, out_path=plot_name)
    

if __name__ == '__main__':    
    
    max_ind = 10000
    n_points = 1000
    base = np.random.choice(max_ind, n_points, replace=False)
    
    yumi_models = get_model_names()
    yumi_models = [yumi_models[2]]
    
    
    # compare_prd(yumi_models, baseline_indices=None, random_seed=1112)
    compare_prd(yumi_models, baseline_indices=base, random_seed=1112)
    # compare_prd('continuous', continuous_models, baseline_indices=None)
    # compare_prd('continuous', continuous_models, baseline_indices=base)