#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 09:18:18 2019

@author: petrapoklukar
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import InfoGAN_models
import InfoGAN_cont as model

import matplotlib
matplotlib.use('Qt5Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import numpy as np
from importlib.machinery import SourceFileLoader
import os
import argparse

parser = argparse.ArgumentParser(description='VAE training for robot motion trajectories')
#parser.add_argument('--train', default=False,action='store_true', help='set it to train the model')
#parser.add_argument('--config_name', default=None, type=str, help='the path to save/load the model')
#parser.add_argument('--path_to_data', default=None, type=str, help='the path to load the data')
#parser.add_argument('--device', default=None, type=str, help='the device for training, cpu or cuda')

class TrajDataset(Dataset):
    def __init__(self, data_filename, device=None):
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.data = torch.from_numpy(
                np.load(data_filename)).float().to(self.device)
        self.num_samples = self.data.shape[0]
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == '__main__':
    args = parser.parse_args()
    
    # TESTING
    args.config_name = 'InfoGAN_cont_test'
    args.path_to_data = 'dataset/robot_trajectories/yumi_joint_pose.npy'
    args.device = None
    args.train = None
    args.eval = True
    
    config_file = os.path.join('.', 'configs', args.config_name + '.py')
    export_directory = os.path.join('.', 'models', args.config_name)
    if (not os.path.isdir(export_directory)):
        os.makedirs(export_directory)
    
    print(' *- Config name: {0}'.format(args.config_name))
    
    config_file = SourceFileLoader(args.config_name, config_file).load_module().config 
    config_file['train_config']['exp_name'] = args.config_name
    config_file['train_config']['exp_dir'] = export_directory # the place where logs, models, and other stuff will be stored
    config_file['data_config']['data_dir'] = args.path_to_data
    
    if args.device == None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = args.device
        
    config_file['train_config']['device'] = device
    print('Device is: {}'.format(device))

    traj_dataset = TrajDataset(args.path_to_data, device)[:1000]
    trajectory_loader = DataLoader(traj_dataset, batch_size=config_file['train_config']['batch_size'],
                                   shuffle=True, num_workers=0)
    traj_iter = iter(trajectory_loader)
    x = traj_iter.next().to('cpu').numpy()
    batch_size, n_joints, traj_length = x.shape
    config_file['data_config']['data_batch_shape'] = x.shape
    config_file['data_config']['input_size'] = n_joints*traj_length


    # Init the model
    model = model.InfoGAN(config_file)
    
    # Train the model
    if args.train:
        model.train_infogan(trajectory_loader)

    # Evaluate the model
    if args.eval:
        eval_config = config_file['eval_config']
        if not args.train:
            model.load_model(eval_config)
        
        # Fix usual noise and sample structured continous noise
        n_samples = eval_config['n_test_samples']
        fix_z_noise = torch.empty((1, model.z_dim), device=model.device).normal_()
        fix_z_noise = fix_z_noise.expand((n_samples, model.z_dim))
        sampled_cont_noise = model.sample_fixed_noise(
                n_samples, model.con_c_dim, ntype='uniform')
        gen_con_samples = model.generator(
                (fix_z_noise, sampled_cont_noise)).view(-1, n_joints, traj_length)
        gen_con_samples = gen_con_samples.detach().view(-1, n_joints, traj_length)
        
        plt.figure(1)
        for i in range(n_samples):
            plt.subplot(n_samples, 1, i+1)
            for j in range(6):
                plt.plot(x[i][0], x[i][j+1], color='b')
                plt.plot(x[i][0], gen_con_samples[i][j+1], color='r')
        plt.show()
        plt.savefig(eval_config['savefig_path'] + 'fixedUsualNoise')
        
        # Fix the structured noise and sample usual noise
        fix_cont_noise = torch.empty((1, model.con_c_dim), device=model.device).normal_()
        fix_cont_noise = fix_cont_noise.expand((n_samples, model.con_c_dim))
        sampled_z_noise = model.sample_fixed_noise(
                n_samples, model.z_dim, ntype='normal')
        gen_nor_samples = model.generator((sampled_z_noise, fix_cont_noise))
        gen_nor_samples = gen_nor_samples.detach().view(-1, n_joints, traj_length)
        
        plt.figure(2)
        for i in range(n_samples):
            plt.subplot(n_samples, 1, i+1)
            for j in range(6):
                plt.plot(x[i][0], x[i][j+1], color='b')
                plt.plot(x[i][0], gen_nor_samples[i][j+1], color='r')
        plt.show()
        plt.savefig(eval_config['savefig_path'] + 'fixedStructuredNoise')
    