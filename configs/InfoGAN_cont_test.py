#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 17:07:52 2019

@author: petrapoklukar
"""

config = {
        'generator_config': {
                'class_name': 'FullyConnectedGenerator',
                'layer_dims': [128, 256, 512]
                },
        
        'discriminator_config': {
                'class_name': 'FullyConnectedDiscriminator',
                'layer_dims': [512, 256, 128, 64]
                },
                
        'Qnet_config': {
                'class_name': 'QNet',
                'last_layer_dim': 64,
                },
                
        'data_config': {
                'input_size': 553,
                'usual_noise_dim': 1,
                'structured_cat_dim': 0, 
                'structured_con_dim': 6,
                'total_noise': 7
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
                'gen_lr_schedule': [(0, 1e-3)],
                'dis_lr_schedule': [(0, 2e-4)],
                'lambda_cat': None,
                'lambda_con': 0.1, 
                'filename': 'infogan',
                'random_seed': 1201
                },
                
        'eval_config': {
                'filepath': 'models/InfoGAN_cont_test/infogan_model.pt',
                'load_checkpoint': False,
                'n_test_samples': 25,
                'n_repeats': 5
#                'n_test_samples': 4,
#                'savefig_path': 'models/InfoGAN_cont_test/'
                }
        }