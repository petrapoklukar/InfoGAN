#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 13:59:33 2019

@author: petrapoklukar
"""

config = {
        'generator_config': {
                'class_name': 'FullyConnectedGenerator',
                'layer_dims': [9, 128, 256, 256, 512]
                },
        
        'discriminator_config': {
                'class_name': 'FullyConnectedDiscriminator',
                'layer_dims': [1000, 500, 250, 250, 50]
                },
                
        'Qnet_config': {
                'class_name': 'QNet',
                'last_layer_dim': 50,
                },
                
        'data_config': {
                'input_size': 553,
                'usual_noise_dim': 1,
                'structured_cat_dim': 0, 
                'structured_con_dim': 6,
                },
                
        'optim_config': {
                'optim_type': 'Adam',
                'gen_lr': 1e-3,
                'gen_b1': 0.9,
                'gen_b2': 0.999,
                'dis_lr': 1e-3,
                'dis_b1': 0.9,
                'dis_b2': 0.999
                },    
                
        'train_config': {
                'batch_size': 64,
                'epochs': 5,
                'snapshot': 2, 
                'console_print': 1,
                'lr_schedule': [(0, 1e-3), (100, 1e-4)],
                'lambda_cat': 0,
                'lambda_con': 0.1, 
                'filename': 'infogan',
                'random_seed': 1201
                },
                
        'eval_config': {
                'filepath': 'models/InfoGAN_test/infogan_model.pt',
                'load_checkpoint': False
                }
        }
