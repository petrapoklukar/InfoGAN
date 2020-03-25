#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 18:36:30 2020

@author: petrapoklukar
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import InfoGAN_continuous as models
import numpy as np
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from importlib.machinery import SourceFileLoader
import os
import argparse
import prd_score as prd

parser = argparse.ArgumentParser(description='VAE training for robot motion trajectories')
parser.add_argument('--config_name', default=None, type=str, help='the path to save/load the model')
parser.add_argument('--train', default=0, type=int, help='set it to train the model')
parser.add_argument('--chpnt_path', default='', type=str, help='set it to train the model')
parser.add_argument('--eval', default=0, type=int, help='evaluates the trained model')
parser.add_argument('--compute_prd', default=0, type=int, help='evaluates the trained model with precision and recall')
parser.add_argument('--device', default=None, type=str, help='the device for training, cpu or cuda')

class ImageDataset(Dataset):
    def __init__(self, dataset_name, path_to_data, device=None, train_split=True):
        if not device:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        transform_list = transforms.Compose([transforms.Resize(32),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5,), (0.5,))
                                             ]) 
        if dataset_name == 'MNIST':
            self.data = datasets.MNIST(path_to_data, train=train_split,
                                 download=True, transform=transform_list)
            
    def get_subset(self, max_ind, n_points):
        self.prd_indices = np.random.choice(max_ind, n_points, replace=False)
        subset_list = [self.data[i][0].numpy() for i in self.prd_indices]
        return np.array(subset_list).reshape(n_points, -1)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx][0]


if __name__ == '__main__':
    args = parser.parse_args()
    
    # # Laptop TESTING
    # args.config_name = 'InfoGAN_MINST_testing'
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
    dataset = ImageDataset('MNIST', path_to_data)
    # Laptop TESTING
    # dataset =  torch.utils.data.Subset(dataset, np.arange(5000))
    dloader = DataLoader(dataset, batch_size=config_file['train_config']['batch_size'],
                         shuffle=True, num_workers=2)
    dloader_iter = iter(dloader)
    imgs = dloader_iter.next()
    print('Input data is of shape: {}'.format(imgs.shape))

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
            
        if str(device) == 'cpu': 
            # Reimport interactive backend
            import matplotlib
            matplotlib.use('Qt5Agg') # Must be before importing matplotlib.pyplot or pylab!
            import matplotlib.pyplot as plt
        
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
                gen_x = model.Gnet((z_noise, con_noise))
                gen_x_plotrescale = (gen_x + 1.) / 2.0 # Cause of tanh activation
                filename = 'evalImages_fixcont{0}_r{1}'.format(str(con_noise_id),
                                            str(repeat))
                model.plot_image_grid(gen_x_plotrescale, filename, model.test_dir,
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
        test_dataset = ImageDataset('MNIST', path_to_data, train_split=False)
        ref_np = test_dataset.get_subset(len(test_dataset), n_prd_samples)

        # Get the sampled np array
        with torch.no_grad():
            z_noise, con_noise = model.ginput_noise(n_prd_samples)
            eval_data = model.Gnet((z_noise, con_noise))
            eval_np = eval_data.numpy().reshape(n_prd_samples, -1)
     
        # Compute prd
        prd_data = prd.compute_prd_from_embedding(eval_np, ref_np)
        prd.plot([prd_data], [args.config_name], 
                 out_path='models/{0}/prd.png'.format(args.config_name))

        
        
        

        
        
        
        
        
        