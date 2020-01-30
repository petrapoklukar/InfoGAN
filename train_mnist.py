#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 09:35:41 2020

@author: petrapoklukar
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import InfoGAN_models
import InfoGAN_general as model

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import numpy as np
from importlib.machinery import SourceFileLoader
import os
import argparse

parser = argparse.ArgumentParser(description='VAE training for robot motion trajectories')
parser.add_argument('--config_name', default=None, type=str, help='the path to save/load the model')
parser.add_argument('--train', default=0, type=int, help='set it to train the model')
parser.add_argument('--chpnt_path', default='', type=str, help='set it to train the model')
parser.add_argument('--eval', default=0, type=int, help='evaluates the trained model')
parser.add_argument('--device', default=None, type=str, help='the device for training, cpu or cuda')


class ImageDataset(Dataset):
    def __init__(self, dataset_name, path_to_data, device=None):
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        transform_list = transforms.Compose([transforms.ToTensor()]) 
        if dataset_name == 'MNIST':
            self.data = datasets.MNIST(path_to_data, train=True,
                                 download=True, transform=transform_list)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx][0]


if __name__ == '__main__':
    args = parser.parse_args()
    
    # Laptop TESTING
#    args.config_name = 'InfoGAN_MINST'
#    args.train = 1
#    args.chpnt_path = 'models/InfoGAN_MINST/infogan_checkpoint9.pth'
#    args.device = None
#    args.eval = None
    
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
    dataset = ImageDataset('MNIST', path_to_data)
    # Laptop TESTING
#    dataset =  torch.utils.data.Subset(dataset, np.arange(100))
    dloader = DataLoader(dataset, batch_size=config_file['train_config']['batch_size'],
                         shuffle=True, num_workers=0)
    dloader_iter = iter(dloader)
    imgs = dloader_iter.next()
    print('Input data is of shape: {}'.format(imgs.shape))

    # Init the model
    model = model.InfoGAN(config_file)
    
    # Train the model
    if args.train:
        model.train_infogan(dloader, chpnt_path=args.chpnt_path)

    # Evaluate the model
    if args.eval:
        eval_config = config_file['eval_config']
        if not args.train:
            model.load_model(eval_config)
        if device.type == 'cpu': 
            # Reimport interactive backend
            import matplotlib
            matplotlib.use('Qt5Agg') # Must be before importing matplotlib.pyplot or pylab!
            import matplotlib.pyplot as plt
        
        # Fix usual noise and the categorical noise, sample structured continous noise
        n_samples = eval_config['n_test_samples']
        fix_z_noise = torch.empty((1, model.z_dim), device=model.device).normal_()
        fix_z_noise = fix_z_noise.expand((n_samples, model.z_dim))
        
        fix_cat_noise = torch.empty((1, model.cat_c_dim), device=model.device).uniform_()
        fix_cat_noise = fix_cat_noise.expand((n_samples, model.cat_c_dim)) 
        
        sampled_cont_noise = model.sample_fixed_noise(
                n_samples, model.con_c_dim, ntype='normal')
        gen_con_samples = model.generator((fix_z_noise, fix_cat_noise, sampled_cont_noise))
        gen_con_samples = gen_con_samples.detach().squeeze()
        
        plt.figure(1)
        for i in range(n_samples):
            plt.subplot(n_samples, 1, i+1)
            plt.imshow(gen_con_samples[i])
        plt.savefig(eval_config['savefig_path'] + 'sampledStructuredContNoise')
        plt.close()
        
        # Fix usual noise and the continous noise, sample structured categorical noise
        fix_cont_noise = torch.empty((1, model.con_c_dim), device=model.device).normal_()
        fix_cont_noise = fix_cont_noise.expand((n_samples, model.con_c_dim))
        
        sampled_cat_noise = model.sample_fixed_noise(
                n_samples, model.cat_c_dim, ntype='uniform')
        gen_nor_samples = model.generator((fix_z_noise, sampled_cat_noise, fix_cont_noise))
        gen_nor_samples = gen_nor_samples.detach().squeeze()
        
        plt.figure(2)
        for i in range(n_samples):
            plt.subplot(n_samples, 1, i+1)
            plt.imshow(gen_con_samples[i])
        plt.savefig(eval_config['savefig_path'] + 'sampledStructuredCatNoise')
        plt.close()