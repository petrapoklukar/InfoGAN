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
import InfoGAN as InfoGAN

import matplotlib.pyplot as plt
import numpy as np
from importlib.machinery import SourceFileLoader
import os
import argparse

parser = argparse.ArgumentParser(description='VAE training for robot motion trajectories')
parser.add_argument('--train', default=False,action='store_true', help='set it to train the model')
parser.add_argument('--path_to_config', default=None, type=str, help='the path to save/load the model')
parser.add_argument('--path_to_data', default=None, type=str, help='the path to load the data')
parser.add_argument('--device', default=None, type=str, help='the device for training, cpu or cuda')

class TrajDataset(Dataset):
    def __init__(self, data_filename):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = torch.from_numpy(
                np.load(data_filename, allow_pickle=True)).float().to(self.device)
        self.num_samples = self.data.shape[0]
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == '__main__':
    args = parser.parse_args()
    
    config_file = os.path.join('.', 'configs', args.path_to_config + '.py')
    directory = os.path.join('.', 'models', args.path_to_config)
    if (not os.path.isdir(directory)):
        os.makedirs(directory)
    
    print(' *- Training:')
    print('    - VAE: {0}'.format(args.path_to_config))
    
    config_file = SourceFileLoader(args.path_to_config, config_file).load_module().config 
    config_file['exp_info']['exp_name'] = args.path_to_config
    config_file['exp_info']['exp_dir'] = directory # the place where logs, models, and other stuff will be stored

    if args.device == None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = args.device
    print('Device is: {}'.format(device))

    traj_dataset = TrajDataset(args.path_to_data, device)
    trajectory_loader = DataLoader(traj_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=0)
    traj_iter = iter(trajectory_loader)
    x = traj_iter.next().to('cpu').numpy()

    n_joints = x.shape[1]
    traj_length = x.shape[2]

    # Init the model
    model = InfoGAN(config_file)
    
    # Train the model
    if args.train:
        model.train_infogan(trajectory_loader)

    # Evaluate the model
    if args.eval:
        if not args.train:
            model.load_model(args.model_filename)
        lv, _ = model.evaluate(trajectory_loader)

        if args.latent_size == 1:
            plt.hist(lv, bins='auto')
            plt.show()

        elif args.latent_size == 2:

            plt.plot(lv[:,0], lv[:,1],'.')
            plt.show()

        # Visualize few restored data
        z,_ = v_autoencoder.encode(x)
        xhat = v_autoencoder.decode(z).reshape(x.shape)

        nsample = min(x.shape[0],5)
        x = x[:nsample, :, :]
        xhat = xhat[:nsample, : ,:]


        #traj = x[:nsample].reshape(-1,traj_length)
        #traj_hat = xhat[:nsample].reshape(-1,traj_length)
        for i in range(nsample):
            plt.subplot(nsample,1,i+1)
            for j in range(7):
                plt.plot(x[i][0], x[i][j+1], color='b')
                plt.plot(x[i][0], xhat[i][j+1], color='r')
        plt.show()