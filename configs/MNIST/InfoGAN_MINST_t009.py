#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:01:45 2020

@author: petrapoklukar
"""

config = {
        'Gnet_config': {
                'class_name': 'FullyConnectedGNet',
                'latent_dim': 74,
                'linear_dims': [256, 512, 1024],
                'dropout': 0.3,
                'output_dim': 32*32,
                'output_reshape_dims': [-1, 1, 32, 32],
                'out_activation': 'tanh',
                'bias': True
                },
        
        'Snet_config': {
                'class_name': 'FullyConnectedSNet',
                'linear_dims': [512, 256],
                'dropout': 0,
                'output_dim': 32*32,
                'bias': True
                },
        
        'Dnet_config': {
                'class_name': 'FullyConnectedDNet',
                'linear_dims': [512, 256],
                'dropout': 0,
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
                'epochs': 150,
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
                'monitor_Gnet': 5, 
                'Gnet_progress_repeat': 10, 
                
                'grad_clip': True, 
                'Snet_D_grad_clip': 100, 
                'Dnet_D_grad_clip': 100, 
                'Gnet_G_grad_clip': 100, 
                'Snet_G_grad_clip': 100, 
                'Qnet_G_grad_clip': 100, 
                
                'lambda_cat': 0.1,
                'lambda_con': 1, 
                
                'filename': 'infogan',
                'random_seed': 1201,
                },
                
        'eval_config': {
                'filepath': 'models/{0}/infogan_model.pt',
                'load_checkpoint': False,
                'n_cat_test_samples': 25,
                'n_cat_repeats': 1,
                'n_con_test_samples': 100,
                'n_con_repeats': 3,
                'cat_repeat': 10, 
                'con_var_range': 2,
                'n_prd_samples': 1000
                }
        }