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

class ImageDataset(Dataset):
    def __init__(self, dataset_name, path_to_data, device=None, train_split=True):
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        transform_list = transforms.Compose([transforms.Resize(32),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5,), (0.5,))
                                             ]) 
        if dataset_name == 'MNIST':
            self.data = datasets.MNIST(path_to_data, train=train_split,
                                 download=True, transform=transform_list)
    
    def get_subset(self, max_ind, n_points, fixed_indices=None):
        if fixed_indices is None:
            self.prd_indices = np.random.choice(max_ind, n_points, replace=False)
        else:
            self.prd_indices = fixed_indices
        subset_list = [self.data[i][0].numpy() for i in self.prd_indices]
        return np.array(subset_list).reshape(n_points, -1)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx][0]
    

def get_model_names(filename='prd_models.txt'):
    with open('prd_models.txt') as f:
        lines = f.read().splitlines()
        all_models = [list(group) for k, group in groupby(lines, lambda x: x == "") \
                      if not k]
        general_models = all_models[0]
        continuous_models = all_models[1]
    return general_models, continuous_models


def compare_prd(model_type, configs, baseline_indices=None):
    """Calculates PRD for a given list of models."""
    
    if model_type == 'general':
        import InfoGAN_general as models
    else: 
        import InfoGAN_continuous as models

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
    
        # Init the model
        model = models.InfoGAN(config_file)
        eval_config = config_file['eval_config']
        eval_config['filepath'] = parent_dir + eval_config['filepath'].format(config_name)
        model.load_model(eval_config)
        print(eval_config)
    
        # Number of samples to calculate PR on
        n_prd_samples = eval_config['n_prd_samples']
        
        # Get the ground truth np array from the test split
        path_to_data = parent_dir + config_file['data_config']['path_to_data']
        test_dataset = ImageDataset('MNIST', path_to_data, train_split=False)
        ref_np = test_dataset.get_subset(len(test_dataset), n_prd_samples, 
                                         fixed_indices=baseline_indices)
    
        # Get the sampled np array from a trained model
        if model_type == 'general':
            with torch.no_grad():
                z_noise, cat_noise, con_noise = model.ginput_noise(n_prd_samples)
                eval_data = model.Gnet((z_noise, cat_noise, con_noise))
        else: 
            with torch.no_grad():
                z_noise, con_noise = model.ginput_noise(n_prd_samples)
                eval_data = model.Gnet((z_noise, con_noise))
        eval_np = eval_data.numpy().reshape(n_prd_samples, -1)
        
        # Compute and save prd 
        prd_data = prd.compute_prd_from_embedding(eval_np, ref_np)
        prd_scores.append(prd_data)
        prd_models.append(config_name)
    
    short_model_names = [prd_models[i].split('_')[-1] for i in range(len(prd_models))]
    plot_name = '{}_fixedBaseline{}_prds.png'.format(
        model_type, int(0 if baseline_indices is None else 1))
    prd.plot(prd_scores, short_model_names, out_path=plot_name)
    

if __name__ == '__main__':    
    
    max_ind = 10000
    n_points = 1000
    base = np.random.choice(max_ind, n_points, replace=False)
    
    general_models, continuous_models = get_model_names()
    
    compare_prd('general', general_models, baseline_indices=None)
    compare_prd('general', general_models, baseline_indices=base)
    compare_prd('continuous', continuous_models, baseline_indices=None)
    compare_prd('continuous', continuous_models, baseline_indices=base)