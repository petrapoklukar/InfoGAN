#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 10:28:35 2020

@author: petrapoklukar
"""


config = {
        'Gnet_config': {
                'class_name': 'FullyConnectedGNet',
                'latent_dim': 2,
                'linear_dims': [128, 256, 512],
                'dropout': 0.3,
                'output_dim': 7*79,
                'output_reshape_dims': [-1, 7, 79],
                'bias': True,
                'out_activation': None
                },
        
        'Snet_config': {
                'class_name': 'FullyConnectedSNet',
                'linear_dims': [512, 256],
                'dropout': 0,
                'output_dim': 7*79,
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
                'layer_dims': [256, 128],
                'dropout': 0.3,
                'bias': True
                },
                
        'data_config': {
                'input_size': 7*79,
                'n_joints': 7,
                'traj_length': 79,
                'usual_noise_dim': 1, 
                'structured_con_dim': 1,
                'structured_cat_dim': None,
                'total_noise': 2,
                'path_to_data': 'dataset/robot_trajectories/yumi_joint_pose.npy',
                },

        'train_config': {
                'batch_size': 256,
                'epochs': 2000,
                'snapshot': 500,
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
                'Gnet_progress_nimg': 9,
                
                'grad_clip': True, 
                'Snet_D_grad_clip': 100, 
                'Dnet_D_grad_clip': 100, 
                'Gnet_G_grad_clip': 100, 
                'Snet_G_grad_clip': 100, 
                'Qnet_G_grad_clip': 100, 
                
                'lambda_con': 0.5, 
                
                'filename': 'infogan',
                'random_seed': 1201,
                },
                
        'eval_config': {
                'filepath': 'models/{0}/infogan_model.pt',
                'savefig_path': 'models/{0}/Testing/{1}.png',
                'load_checkpoint': False,
                'n_con_test_samples': 9,
                'n_con_repeats': 3,
                'con_var_range': 2,
                'n_prd_samples': 1000
                }
        }