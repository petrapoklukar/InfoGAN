#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 12:18:48 2020

@author: petrapoklukar
"""

config = {
        'generator_config': {
                'class_name': 'ConvolutionalGenerator',
                'layer_dims': [1024, 7*7*128],
                'channel_dims': [128, 64, 1],
                'init_size': 7
                },
        
        'discriminator_config': {
                'class_name': 'ConvolutionalDiscriminator',
                'channel_dims': [1, 64, 128],
                'layer_dims': [128*5*5, 1024, 128]
                },
                
        'Qnet_config': {
                'class_name': 'QNet',
                'last_layer_dim': 128, # see layer_dims in discriminator
                },
        
        'data_config': {
                'input_size': None,
                'usual_noise_dim': 62,
                'structured_cat_dim': 10, 
                'structured_con_dim': 2,
                'total_noise': 74,
                'path_to_data': '../datasets/MNIST'
                },
                
        'optim_config': {
                'optim_type': 'Adam',
                'gen_lr': 1e-3,
                'gen_b1': 0.9,
                'gen_b2': 0.999,
                'dis_lr': 2e-4,
                'dis_b1': 0.9,
                'dis_b2': 0.999
                },    
                
        'train_config': {
                'batch_size': 64,
                'epochs': 1000,
                'snapshot': 5, 
                'console_print': 1,
                'gen_lr_schedule': [(0, 1e-3)],
                'dis_lr_schedule': [(0, 2e-4)],
                'lambda_cat': 1.,
                'lambda_con': 0.1, 
                'filename': 'infogan',
                'random_seed': 1201
                },
                
        'eval_config': {
                'filepath': 'models/InfoGAN_MNIST/infogan_model.pt',
                'load_checkpoint': False,
                'n_test_samples': 25,
                'n_repeats': 5
                }
        }