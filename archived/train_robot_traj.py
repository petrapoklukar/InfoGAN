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
        self.np_data = np.load(data_filename)
        self.min = np.min(self.np_data, axis=0)
        self.max = np.max(self.np_data, axis=0)
        unit_scaled = (self.np_data - self.min) / (self.max - self.min)
        self.data_scaled = 2 * unit_scaled - 1
        self.data = torch.from_numpy(self.data_scaled).float()
        self.num_samples = self.data.shape[0]                
    
    def get_subset(self, max_ind, n_points, reshape=True):
        self.prd_indices = np.random.choice(max_ind, n_points, replace=False)
        subset_list = [self.data[i].numpy() for i in self.prd_indices]
        if reshape: 
            return np.array(subset_list).reshape(n_points, -1)
        else: 
            return np.array(subset_list)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == '__main__':
    args = parser.parse_args()
    
    # Laptop TESTING
    # args.config_name = 'InfoGAN_MINST_testing'
    # args.train = 0
    # args.chpnt_path = ''#'models/InfoGAN_MINST_testing/infogan_lastCheckpoint.pth'
    # args.device = None
    # args.eval = 0
    # args.compute_prd = 0
    
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
    # dataset = TrajDataset(path_to_data, device)[:256]
    dataset = TrajDataset(path_to_data, device)

    dloader = DataLoader(dataset, batch_size=config_file['train_config']['batch_size'],
                         shuffle=True, num_workers=2)

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
                gen_x = model.g_forward(z_noise, con_noise).detach()
                filename = 'evalImages_fixcont{0}_r{1}'.format(str(con_noise_id),
                                            str(repeat))
                model.plot_traj_grid(gen_x, filename, model.test_dir,
                                      n=n_con_samples)
                
        # Fix usual noise variable
        if config_file['data_config']['use_usual_noise']:
            for con_noise_id in range(model.con_c_dim):
                for repeat in range(n_con_repeats):
                    
                    # Sample the rest and keep the fixed one
                    _, con_noise = model.ginput_noise(n_con_samples)
                    z_noise = model.sample_fixed_noise(
                        'normal', 1, noise_dim=model.z_dim)
                    z_noise = z_noise.expand(
                        (n_con_samples, model.z_dim))
                    
                    # Generate an image
                    gen_x = model.g_forward(z_noise, con_noise).detach()
                    filename = 'evalImages_fixusual_r{0}'.format(str(repeat))
                    model.plot_traj_grid(gen_x, filename, model.test_dir,
                                          n=n_con_samples)
        
        # Plot original
        test_dataset = TrajDataset(path_to_data, device)
        ref_np = torch.from_numpy(
            test_dataset.get_subset(len(test_dataset), n_con_samples, reshape=False))
                
        filename = 'evalImages_original'
        model.plot_traj_grid(ref_np, filename, model.test_dir,
                              n=n_con_samples)

    # Evaluate the model
    if args.compute_prd:
        eval_config = config_file['eval_config']
        eval_config['filepath'] = eval_config['filepath'].format(args.config_name)

        compute_chpnt_prds = eval_config['compute_chpnt_prds']
        chnpt_list = eval_config['chnpt_list']
        
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
            eval_data = model.g_forward(z_noise, con_noise)
            eval_np = eval_data.cpu().numpy().reshape(n_prd_samples, -1)
     
        prd_data = prd.compute_prd_from_embedding(eval_np, ref_np)
        
        # Only evaluate the last model
        if not compute_chpnt_prds:
            # Compute prd
            prd.plot([prd_data], [args.config_name], 
                     out_path='models/{0}/prd.png'.format(args.config_name))
        
        # Evaluate the intermediate checkponts 
        else:
            # Load the chpt
            chpr_prd_list = []
            for c in chnpt_list:
                model_chpt = models.InfoGAN(config_file)
                model_chpt.load_checkpoint(
                    'models/{0}/infogan_checkpoint{1}.pth'.format(args.config_name, c))
                model_chpt.Gnet.eval()
                
                with torch.no_grad():
                    z_noise, con_noise = model_chpt.ginput_noise(n_prd_samples)
                    eval_chnpt_data = model_chpt.g_forward(z_noise, con_noise)
                    eval_chnpt_np = eval_chnpt_data.cpu().numpy().reshape(n_prd_samples, -1)
                
                prd_chnpt_data = prd.compute_prd_from_embedding(eval_chnpt_np, ref_np)
                chpr_prd_list.append(prd_chnpt_data)
            
            # Compute and save prd 
            short_model_name = '_'.join(args.config_name.split('_')[-3:])
            all_prd_data = [prd_data] + chpr_prd_list
            all_names = [short_model_name] + chnpt_list
            prd.plot(all_prd_data, all_names,
                     out_path='models/{0}/prd_chpnts.png'.format(args.config_name))
        

        
        
    

        
        
        
        
        
        