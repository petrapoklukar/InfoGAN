#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 18:39:33 2020

@author: petrapoklukar
"""

config = {
        'Gnet_config': {
                'class_name': 'FullyConnectedGNet',
                'latent_dim': 74,
                'linear_dims': [256, 512, 1024],
                'dropout': 0.3,
                'image_channels': 1,
                'image_size': 32,
                'bias': True
                },
        
        'Snet_config': {
                'class_name': 'FullyConnectedSNet',
                'linear_dims': [512, 256],
                'dropout': 0.3,
                'image_channels': 1,
                'image_size': 32,
                'bias': True
                },
        
        'Dnet_config': {
                'class_name': 'FullyConnectedDNet',
                'linear_dims': [512, 256],
                'dropout': 0,
                'image_channels': 1,
                'image_size': 32,
                'bias': True
                },
                
        'Qnet_config': {
                'class_name': 'FullyConnectedQNet',
                'last_layer_dim': 256, # see layer_dims in discriminator
                'layer_dims': None,
                'dropout': 0.3,
                'bias': True
                },
                
        'data_config': {
                'input_size': None,
                'usual_noise_dim': 62,
                'structured_cat_dim': 10, 
                'structured_con_dim': 2,
                'total_noise': 74,
                'path_to_data': '../datasets/MNIST'
                },

        'train_config': {
                'batch_size': 256,
                'epochs': 300,
                'snapshot': 50, 
                'console_print': 1,
                'optim_type': 'Adam',
                'Goptim_lr_schedule': [(0, 2e-4)],
                'Goptim_b1': 0.7,
                'Goptim_b2': 0.999,
                'Doptim_lr_schedule': [(0, 2e-4)],
                'Doptim_b1': 0.7,
                'Doptim_b2': 0.999,
                
                'input_noise': False,
                'input_variance_increase': None, 
                'Dnet_update_step': 1, 
                'monitor_Gnet': 1, 
                'Gnet_progress_repeat': 10, 
                
                'grad_clip': True, 
                'Snet_D_grad_clip': 50, 
                'Dnet_D_grad_clip': 50, 
                'Gnet_G_grad_clip': 50, 
                'Snet_G_grad_clip': 50, 
                'Qnet_G_grad_clip': 50, 
                
                'lambda_cat': 1,
                'lambda_con': 0.1, 
                
                'filename': 'infogan',
                'random_seed': 1201,
                },
                
        'eval_config': {
                'filepath': 'models/{0}/infogan_model.pt',
                'load_checkpoint': False,
                'n_cat_test_samples': 25,
                'n_cat_repeats': 2,
                'n_con_test_samples': 100,
                'n_con_repeats': 5,
                'cat_repeat': 10, 
                'con_var_range': 2,
                'n_prd_samples': 1000
                }
        }