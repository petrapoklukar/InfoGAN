#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 15:26:08 2020

@author: petrapoklukar
"""

import torch
from torch.utils.data import Dataset, DataLoader
import InfoGAN_yumi as models
import numpy as np
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from importlib.machinery import SourceFileLoader
import os
import argparse
import prd_score as prd
import pickle

parser = argparse.ArgumentParser(description='VAE training for robot motion trajectories')
parser.add_argument('--config_name', default=None, type=str, help='the path to save/load the model')
parser.add_argument('--train', default=0, type=int, help='set it to train the model')
parser.add_argument('--chpnt_path', default='', type=str, help='set it to train the model')
parser.add_argument('--eval', default=0, type=int, help='evaluates the trained model')
parser.add_argument('--compute_prd', default=0, type=int, help='evaluates the trained model with precision and recall')
parser.add_argument('--device', default=None, type=str, help='the device for training, cpu or cuda')

class TrajDataset(Dataset):
    def __init__(self, data_filename, device=None):
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.data = torch.from_numpy(
                np.load(data_filename)).float()#.to(self.device)
        self.num_samples = self.data.shape[0]                
    
    def get_subset(self, max_ind, n_points):
        self.prd_indices = np.random.choice(max_ind, n_points, replace=False)
        subset_list = [self.data[i].numpy() for i in self.prd_indices]
        return np.array(subset_list).reshape(n_points, -1)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]#.to(self.device)


if __name__ == '__main__':
    args = parser.parse_args()
    
    # # Laptop TESTING
    # args.config_name = 'InfoGAN_yumi_l10_u1s1'
    # args.train = 0
    # args.chpnt_path = ''#'models/InfoGAN_MINST_testing/infogan_lastCheckpoint.pth'
    # args.device = None
    # args.eval = 0
    # args.compute_prd = 1
    
    # Load config
    config_file = os.path.join('.', 'configs', args.config_name + '.py')
    export_directory = os.path.join('.', 'models', args.config_name)
    if (not os.path.isdir(export_directory)):
        os.makedirs(export_directory)
    
    print(' *- Config name: {0}'.format(args.config_name))
    
    config_file = SourceFileLoader(args.config_name, config_file).load_module().config 
    config_file['train_config']['exp_name'] = args.config_name
    config_file['train_config']['exp_dir'] = export_directory # the place where logs, models, and other stuff will be stored
    
    # Set the device
    if args.device == None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = args.device
        
    config_file['train_config']['device'] = device
    print('Device is: {}'.format(device))

    # Load the data 
    path_to_data = config_file['data_config']['path_to_data']
    # Laptop TESTING
    # dataset = TrajDataset(path_to_data, device)[:1000]
    dataset = TrajDataset(path_to_data, device)

    dloader = DataLoader(dataset, batch_size=config_file['train_config']['batch_size'],
                         shuffle=True, num_workers=2)
    # dloader_iter = iter(dloader)
    # x = dloader_iter.next().to('cpu').numpy()
    # print('Input data is of shape: {}'.format(x.shape))

    # Init the model
    model = models.InfoGAN(config_file)
    
    # Train the model
    if args.train:
        model.train_model(dloader, chpnt_path=args.chpnt_path)

    # Evaluate the model
    if args.eval:
        eval_config = config_file['eval_config']
        eval_config['filepath'] = eval_config['filepath'].format(args.config_name)
        print(eval_config)
        if not args.train:
            model.load_model(eval_config)
        else:
            model.Qnet.eval()
            model.Snet.eval()
            model.Dnet.eval()
            model.Qnet.eval()
                    
        n_con_samples = eval_config['n_con_test_samples']
        n_con_repeats = eval_config['n_con_repeats']
        con_var_range = eval_config['con_var_range']
        
        # Fix one structured continuous noise variable
        fixed_con_noise = model.sample_fixed_noise(
            'equidistant', n_con_samples - 1, var_range=con_var_range)

        for con_noise_id in range(model.con_c_dim):
            for repeat in range(n_con_repeats):
                
                # Sample the rest and keep the fixed one
                z_noise, con_noise = model.ginput_noise(n_con_samples)
                con_noise[:, con_noise_id] = fixed_con_noise
                
                # Generate an image
                gen_x = model.Gnet((z_noise, con_noise)).detach()
                filename = 'evalImages_fixcont{0}_r{1}'.format(str(con_noise_id),
                                            str(repeat))
                model.plot_traj_grid(gen_x, filename, model.test_dir,
                                      n=n_con_samples)
                
        # Fix usual noise variable
        for con_noise_id in range(model.con_c_dim):
            for repeat in range(n_con_repeats):
                
                # Sample the rest and keep the fixed one
                _, con_noise = model.ginput_noise(n_con_samples)
                z_noise = model.sample_fixed_noise(
                    'normal', 1, noise_dim=model.z_dim)
                z_noise = z_noise.expand(
                    (n_con_samples, model.z_dim))
                
                # Generate an image
                gen_x = model.Gnet((z_noise, con_noise)).detach()
                filename = 'evalImages_fixusual_r{0}'.format(str(repeat))
                model.plot_traj_grid(gen_x, filename, model.test_dir,
                                      n=n_con_samples)

    # Evaluate the model
    if args.compute_prd:
        eval_config = config_file['eval_config']
        eval_config['filepath'] = eval_config['filepath'].format(args.config_name)
        print(eval_config)
        if not args.train:
            model.load_model(eval_config)
        else:
            model.Qnet.eval()
            model.Snet.eval()
            model.Dnet.eval()
            model.Qnet.eval()
        
        n_prd_samples = eval_config['n_prd_samples']
        
        # Get the ground truth np array
        test_dataset = TrajDataset(path_to_data, device)
        ref_np = test_dataset.get_subset(len(test_dataset), n_prd_samples)

        # Get the sampled np array
        with torch.no_grad():
            z_noise, con_noise = model.ginput_noise(n_prd_samples)
            eval_data = model.Gnet((z_noise, con_noise))
            eval_np = eval_data.cpu().numpy().reshape(n_prd_samples, -1)
     
        # Compute prd
        prd_data = prd.compute_prd_from_embedding(eval_np, ref_np)
        prd.plot([prd_data], [args.config_name], 
                 out_path='models/{0}/prd.png'.format(args.config_name))

        
        
    

        
        
        
        
        
        