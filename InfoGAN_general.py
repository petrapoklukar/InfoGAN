#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:42:12 2020

@author: petrapoklukar

Original InfoGAN implementation.
"""

import torch.nn as nn
import torch.optim as optim
import torch
import InfoGAN_models
import InfoGAN
import numpy as np
import os
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

class InfoGAN_general(InfoGAN):
    def __init__(self, config):
        super(InfoGAN_general, self).__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training parameters
        train_config = config['train_config']
        self.Gnet_progress_repeat = train_config['Gnet_progress_repeat']
        self.Gnet_progress_nimg = int(
            config['data_config']['structured_cat_dim'] * self.Gnet_progress_repeat)
        
        # Info parameters
        self.z_dim = self.data_config['usual_noise_dim']
        self.cat_c_dim = self.data_config['structured_cat_dim']
        self.con_c_dim = self.data_config['structured_con_dim']
        self.fix_noise()
        self.lambda_cat = train_config['lambda_cat']
        self.lambda_con = train_config['lambda_con']
        

    # ---------------------- #
    # --- Init functions --- #
    # ---------------------- #
    def init_losses(self):
        """Initialises the loss."""
        # GAN Loss function
        self.bce_loss = torch.nn.BCELoss().to(self.device)  
        # Discrete latent codes 
        self.ce_loss = torch.nn.CrossEntropyLoss().to(self.device)
        # Continuous latent codes are model with the gaussian function below

    
    def fix_noise(self):
        """Fixes noise to monitor the progress of the generator."""
        batch_cat_codes = np.arange(self.cat_c_dim).repeat(self.Gnet_progress_repeat)
        z_noise, cat_noise, con_noise = self.ginput_noise(
                self.Gnet_progress_nimg, batch_cat_c_dim=batch_cat_codes)
        
        self.fixed_z_noise = z_noise
        self.fixed_cat_noise = cat_noise
        self.fixed_con_noise = con_noise

    # ---------------------------- #
    # --- Monitoring functions --- #
    # ---------------------------- #        
    def sq_else_perm(self, img):
        """Needeed for handling RGB vs greyscaled images."""
        grayscale = True if img.shape[1] == 1 else False
        return img.squeeze() if grayscale else img.permute(1,2,0)
    
    def plot_image_grid(self, images, filename, directory, n=0):
        """Plots a grid of (generated) images."""
        
        n_subplots = np.sqrt(n).astype(int) if n!=0 else self.Gnet_progress_nimg
        plot_range = n_subplots ** 2
        images = self.sq_else_perm(images)
        for i in range(plot_range):
            plt.subplot(n_subplots, n_subplots, 1 + i)
            plt.axis('off')
            plt.imshow(images[i].detach().cpu().numpy())
        plt.savefig(directory + filename)
        plt.clf()
        plt.close()
        

    def sample_fixed_noise(self, ntype, n_samples, noise_dim=None, var_range=1):
        """Samples one type of noise only"""
        if ntype == 'uniform':
            return torch.empty((n_samples, noise_dim), device=self.device).uniform_(-1, 1)
        elif ntype == 'normal':
            return torch.empty((n_samples, noise_dim), device=self.device).normal_()
        elif ntype == 'equidistant':
            return torch.from_numpy((np.arange(n_samples + 1) / n_samples) * 4 - 2)
        else:
            raise ValueError('Noise type {0} not recognised.'.format(ntype))
            
    # -------------------------- #
    # --- Training functions --- #
    # -------------------------- #
    def ginput_noise(self, batch_size, batch_cat_c_dim=None):
        """
        Generates uninformed noise, structured discrete noise and 
        structured continuous noise.
        """
        # the usual uninformed noise
        z_noise = torch.empty((batch_size, self.z_dim), 
                              device=self.device).normal_() # b, x_dim
        
        # structured discrete code noise
        if batch_cat_c_dim is None:
            # Generates a batch of random categorical codes if no is specified
            batch_cat_c_dim = np.random.randint(0, self.cat_c_dim, batch_size)
        cat_noise = np.zeros((batch_size, self.cat_c_dim)) 
        cat_noise[range(batch_size), batch_cat_c_dim] = 1.0 # bs, dis_classes
        cat_noise = torch.Tensor(cat_noise).to(self.device) 
        
        # structured continuous code noise
        con_noise = torch.empty((batch_size, self.con_c_dim),
                                device=self.device).uniform_(-1, 1)
        return z_noise, cat_noise, con_noise
    
    
    def train_model(self, train_dataloader, chpnt_path=''):
        """Trains an InfoGAN."""
        
        print(('\nPrinting model specifications...\n' + 
               ' *- Path to the model: {0}\n' + 
               ' *- Number of epochs: {1}\n' + 
               ' *- Batch size: {2}\n' 
               ).format(self.model_path, self.epochs, self.batch_size))
        
        if chpnt_path: 
            # Pick up the last epochs specs
            self.load_checkpoint(chpnt_path)
    
        else:
            # Initialise the models, weights and optimisers
            self.init_models_and_weights()
            self.start_Goptim_epoch, self.Goptim_lr = self.Goptim_lr_schedule.pop(0)
            self.start_Doptim_epoch, self.Doptim_lr = self.Doptim_lr_schedule.pop(0)
            assert(self.start_Goptim_epoch == self.start_Doptim_epoch)
            self.start_epoch = self.start_Doptim_epoch
            try:
                self.Goptim_lr_update_epoch, self.new_Goptim_lr = self.Goptim_lr_schedule.pop(0)
                self.Doptim_lr_update_epoch, self.new_Doptim_lr = self.Doptim_lr_schedule.pop(0)
            except:
                self.Goptim_lr_update_epoch = self.start_epoch - 1
                self.new_Goptim_lr = self.Goptim_lr
                
                self.Doptim_lr_update_epoch = self.start_epoch - 1
                self.new_Doptim_lr = self.Doptim_lr

            self.init_optimisers()
            self.epoch_losses = []
            self.D_sgrad_norms, self.D_dgrad_norms = [], []
            self.G_ggrad_norms, self.G_qgrad_norms = [], []
            
            self.D_sgrad_total_norm, self.D_dgrad_total_norm = [], []
            self.G_ggrad_total_norm, self.G_qgrad_total_norm = [], []
            
            print((' *- G_optimiser' + 
                   '    *- Learning rate: {0}\n' + 
                   '    *- Next lr update at {1} to the value {2}\n' + 
                   '    *- Remaining lr schedule: {3}'
                   ).format(self.Goptim_lr, self.Goptim_lr_update_epoch, 
                            self.new_Goptim_lr, self.Goptim_lr_schedule))            
            print((' *- D_optimiser' + 
                   '    *- Learning rate: {0}\n' + 
                   '    *- Next lr update at {1} to the value {2}\n' + 
                   '    *- Remaining lr schedule: {3}'
                   ).format(self.Doptim_lr, self.Doptim_lr_update_epoch, 
                            self.new_Doptim_lr, self.Doptim_lr_schedule))            

        self.print_model_params()
        self.init_losses()
        print('\nStarting to train the model...\n' )        
        for self.current_epoch in range(self.start_epoch, self.epochs):
            self.Gnet.train()
            self.Snet.train()
            self.Dnet.train()
            self.Qnet.train()
            assert(self.Gnet.training)
            
            epoch_loss = np.zeros(7)
            epochs_D_snet_norms = []
            epochs_D_dnet_norms = []
            epochs_G_gnet_norms = []
            epochs_G_qnet_norms = []
            for i, x in enumerate(train_dataloader):
                
                # Ground truths
                batch_size = x.shape[0]

                real_x = x.to(self.device)        
                ones = torch.ones(batch_size, device=self.device)
                zeros = torch.zeros(batch_size, device=self.device)
                
                # ------------------------------- #
                # --- Train the Discriminator --- #
                # ------------------------------- #
                self.optimiser_D.zero_grad()
        
                # Usual discriminator Loss for real images
                real_x = self.dinput_noise(real_x)
                d_real_x = self.d_forward(real_x)                
                assert torch.sum(torch.isnan(d_real_x)) == 0, d_real_x
                assert(d_real_x >= 0.).all(), d_real_x
                assert(d_real_x <= 1.).all(), d_real_x
                d_real_loss = self.bce_loss(d_real_x, ones)
                D_x = d_real_x.mean().item()
                
                # Usual discriminator for fake images
                z_noise, cat_noise, con_noise = self.ginput_noise(batch_size)
                fake_x = self.Gnet((z_noise, cat_noise, con_noise)).detach() 
                assert torch.sum(torch.isnan(fake_x)) == 0, fake_x
                fake_x_input = self.dinput_noise(fake_x.detach())
                d_fake_x = self.d_forward(fake_x_input)
                assert(d_fake_x >= 0.).all(), d_fake_x
                assert(d_fake_x <= 1.).all(), d_fake_x
                d_fake_loss = self.bce_loss(d_fake_x, zeros)
                D_G_z1 = d_fake_x.mean().item()
               
                # Total discriminator loss. 
                total_d_loss = d_real_loss + d_fake_loss 
                total_d_loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                            self.Dnet.parameters(), self.Dnet_D_grad_clip) 
                    torch.nn.utils.clip_grad_norm_(
                            self.Snet.parameters(), self.Snet_D_grad_clip) 
                self.optimiser_D.step()
                
                # Track Snet's and Dnet's gradients            
                b_snet_norms = self.get_gradients(self.Snet)
                epochs_D_snet_norms.append(b_snet_norms)
                b_D_snet_norm_total = np.around(np.linalg.norm(np.array(b_snet_norms)),
                                                decimals=3)

                b_dnet_norms = self.get_gradients(self.Dnet)
                epochs_D_dnet_norms.append(b_dnet_norms)
                b_D_dnet_norm_total = np.around(np.linalg.norm(np.array(b_dnet_norms)), 
                                              decimals=3)

                # --------------------------- #
                # --- Train the Generator --- #
                # --------------------------- #
                self.optimiser_G.zero_grad()
                
                # Usual generator loss
                z_noise, cat_noise, con_noise = self.ginput_noise(batch_size)
                fake_x = self.Gnet(((z_noise, cat_noise, con_noise)))
                fake_x_input = self.dinput_noise(fake_x)
                d_fake_x = self.d_forward(fake_x_input)
                g_loss = self.bce_loss(d_fake_x, ones)
                D_G_z2 = d_fake_x.mean().item()
               
                # Info loss for the Snet and Qnet
                # - Resampled ground truth labels, see eq 5 in the paper
                sampled_labels = np.random.randint(0, self.cat_c_dim, batch_size)
                gt_labels = torch.LongTensor(sampled_labels).to(self.device)
        
                # - Resample noise, labels and code as generator input
                z_noise, cat_noise, con_noise = self.ginput_noise(
                        batch_size, batch_cat_c_dim=sampled_labels)
        
                # - Push it through QNet
                gen_x = self.Gnet((z_noise, cat_noise, con_noise))
                q_cat_code, q_con_mean, q_con_logvar = self.q_forward(gen_x) 
        
                G_i_loss = self.lambda_cat * self.ce_loss(q_cat_code, gt_labels) + \
                    self.lambda_con * self.gaussian_loss(con_noise, q_con_mean, 
                                                         q_con_logvar)
                
                total_g_loss = g_loss + G_i_loss 
                total_g_loss.backward()
                
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                            self.Gnet.parameters(), self.Gnet_G_grad_clip) 
                    torch.nn.utils.clip_grad_norm_(
                            self.Qnet.parameters(), self.Qnet_G_grad_clip) 
                self.optimiser_G.step()
                
                # Track Gnet's, Snet's and Qnet's gradients
                b_gnet_norms = self.get_gradients(self.Gnet)
                epochs_G_gnet_norms.append(b_gnet_norms)
                b_G_gnet_norm_total = np.around(np.linalg.norm(np.array(b_gnet_norms)), 
                                              decimals=3)
                              
                b_qnet_norms = self.get_gradients(self.Qnet)
                epochs_G_qnet_norms.append(b_qnet_norms)
                b_G_qnet_norm_total = np.around(np.linalg.norm(np.array(b_qnet_norms)), 
                                              decimals=3)
                
                epoch_loss += self.format_loss([
                        total_d_loss, total_g_loss, d_real_loss, d_fake_loss, 
                        g_loss, G_i_loss])
        
            # ------------------------ #
            # --- Log the training --- #
            # ------------------------ #      
            epoch_loss /= len(train_dataloader)
            print(
                "[Epoch %d/%d]\n\t[Total D loss: %f] [Total G loss: %f]"
                % (self.current_epoch, self.epochs, epoch_loss[0], epoch_loss[1]))
            print(
                "\t[D_real_loss: %f] [D_fake_loss: %f]"
                % (epoch_loss[2], epoch_loss[3]))
            print(
                "\t[G_loss: %f] [G_i_loss: %f]"
                % (epoch_loss[4], epoch_loss[4]))
            print(
                "\t[D_x %f] [D_G_z1: %f] [D_G_z2: %f]"
                % (D_x, D_G_z1, D_G_z2))
            print("\t[D_optim_Snet_norm_mean]: ", np.mean(epochs_D_snet_norms, axis=0))
            print("\t[D_optim_Dnet_norm_mean]: ", np.mean(epochs_D_dnet_norms, axis=0))
            print("\t[G_optim_Gnet_norm_mean]: ", np.mean(epochs_G_gnet_norms, axis=0))
            print("\t[G_optim_Qnet_norm_mean]: ", np.mean(epochs_G_qnet_norms, axis=0))
            
            self.epoch_losses.append(epoch_loss)
            self.D_sgrad_norms.append(np.mean(epochs_D_snet_norms, axis=0))
            self.D_dgrad_norms.append(np.mean(epochs_D_dnet_norms, axis=0))
            self.G_ggrad_norms.append(np.mean(epochs_G_gnet_norms, axis=0))
            self.G_qgrad_norms.append(np.mean(epochs_G_qnet_norms, axis=0))

            self.D_sgrad_total_norm.append(b_D_snet_norm_total)
            self.D_dgrad_total_norm.append(b_D_dnet_norm_total)
            self.G_ggrad_total_norm.append(b_G_gnet_norm_total)
            self.G_qgrad_total_norm.append(b_G_qnet_norm_total)

            self.plot_model_loss() 
            self.plot_gradients()
            self.save_checkpoint(epoch_loss)
            
            if (self.current_epoch + 1) % self.snapshot == 0:
                # Save the checkpoint & logs, plot snapshot losses
                self.save_checkpoint(epoch_loss, keep=False)
                self.save_logs()

                # Plot snapshot losses
                self.plot_snapshot_loss()
                
            if (self.current_epoch + 1) % self.monitor_Gnet == 0:
                gen_x = self.Gnet((self.fixed_z_noise, self.fixed_cat_noise,
                                   self.fixed_con_noise)) 
                gen_x_plotrescale = (gen_x + 1.) / 2.0 # Cause of tanh activation
                   
                filename = 'genImages' + str(self.current_epoch)
                self.plot_image_grid(gen_x_plotrescale, filename, self.train_dir, n=100)
                
        
        # ---------------------- #
        # --- Save the model --- #
        # ---------------------- # 
        print('Training completed.')
        self.plot_model_loss()
        self.Gnet.eval()
        self.Snet.eval()
        self.Dnet.eval()
        self.Qnet.eval()
        torch.save({
                'Gnet': self.Gnet.state_dict(),
                'Snet': self.Snet.state_dict(),
                'Dnet': self.Dnet.state_dict(),
                'Qnet': self.Qnet.state_dict()}, 
                self.model_path)       
        self.save_logs()
        
    # ---------------------------------- #
    # --- Saving & Loading functions --- #
    # ---------------------------------- #
    def save_logs(self, ):
        """Saves a txt file with logs"""
        log_filename = self.save_path + '_logs.txt'
        epoch_losses = np.stack(self.epoch_losses)
        
        with open(log_filename, 'w') as f:
            f.write('Model {0}\n\n'.format(self.config['train_config']['filename']))
            f.write( str(self.config) )
            f.writelines(['\n\n', 
                    '*- Model path: {0}\n'.format(self.model_path),
                    '*- Generator learning rate schedule: {0}\n'.format(
                        self.init_Goptim_lr_schedule),
                    '*- Discriminator learning rate schedule: {0}\n'.format(
                        self.init_Doptim_lr_schedule),
                    '*- Training epoch losses: (total_d_loss, total_g_loss, ' + 
                    'd_real_loss, d_fake_loss, g_loss, G_i_loss)\n'
                    ])
            f.writelines(list(map(
                    lambda t: '{0:>3} Epoch {7}: ({1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}, {5:.2f}, {6:.2f})\n'.format(
                            '', t[0], t[1], t[2], t[3], t[4], t[5]), 
                    epoch_losses)))
        print(' *- Model saved.\n')
    
    
    def save_checkpoint(self, epoch_loss, keep=False):
        """Saves a checkpoint during the training."""
        if keep:
            path = self.save_path + '_checkpoint{0}.pth'.format(self.current_epoch)
            checkpoint_type = 'epoch'
        else:
            path = self.save_path + '_lastCheckpoint.pth'
            checkpoint_type = 'last'
        training_dict = {
                'last_epoch': self.current_epoch,
                
                'Gnet_state_dict': self.Gnet.state_dict(),
                'Snet_state_dict': self.Snet.state_dict(),
                'Dnet_state_dict': self.Dnet.state_dict(),
                'Qnet_state_dict': self.Qnet.state_dict(),
            
                'optimiser_D_state_dict': self.optimiser_D.state_dict(),
                'optimiser_G_state_dict': self.optimiser_G.state_dict(),
                
                'last_epoch_loss': epoch_loss,
                'epoch_losses': self.epoch_losses,

                'snapshot': self.snapshot,
                'console_print': self.console_print,
                
                'current_Goptim_lr': self.Goptim_lr,
                'Goptim_lr_update_epoch': self.Goptim_lr_update_epoch, 
                'new_Goptim_lr': self.new_Goptim_lr, 
                'Goptim_lr_schedule': self.Goptim_lr_schedule,
                
                'current_Doptim_lr': self.Doptim_lr,
                'Doptim_lr_update_epoch': self.Doptim_lr_update_epoch, 
                'new_Doptim_lr': self.new_Doptim_lr, 
                'Doptim_lr_schedule': self.Doptim_lr_schedule,
                
                'D_sgrad_norms': self.D_sgrad_norms,
                'D_dgrad_norms': self.D_dgrad_norms,
                'G_ggrad_norms': self.G_ggrad_norms,
                'G_qgrad_norms': self.G_qgrad_norms,
                
                'D_sgrad_total_norm': self.D_sgrad_total_norm, 
                'D_dgrad_total_norm': self.D_dgrad_total_norm,
                'G_ggrad_total_norm': self.G_ggrad_total_norm, 
                'G_qgrad_total_norm': self.G_qgrad_total_norm
                }
        
        torch.save({**training_dict, **self.config}, path)
        print(' *- Saved {1} checkpoint {0}.'.format(self.current_epoch, checkpoint_type))
    
        
    def load_checkpoint(self, path):
        """
        Loads a checkpoint and initialises the models to continue training.
        """
        checkpoint = torch.load(path, map_location=self.device)
                
        self.init_Gnet()
        self.init_Snet()
        self.init_Dnet()
        self.init_Qnet()
        
        self.Gnet.load_state_dict(checkpoint['Gnet_state_dict'])
        self.Snet.load_state_dict(checkpoint['Snet_state_dict'])
        self.Dnet.load_state_dict(checkpoint['Dnet_state_dict'])
        self.Qnet.load_state_dict(checkpoint['Qnet_state_dict'])
        
        self.Goptim_lr = checkpoint['current_Goptim_lr']
        self.Goptim_lr_update_epoch = checkpoint['Goptim_lr_update_epoch']
        self.new_Goptim_lr = checkpoint['new_Goptim_lr']
        self.Goptim_lr_schedule = checkpoint['Goptim_lr_schedule']
        
        self.Doptim_lr = checkpoint['current_Doptim_lr']
        self.Doptim_lr_update_epoch = checkpoint['Doptim_lr_update_epoch']
        self.new_Doptim_lr = checkpoint['new_Doptim_lr']
        self.Doptim_lr_schedule = checkpoint['Doptim_lr_schedule']
        
        self.D_sgrad_norms = checkpoint['D_sgrad_norms']
        self.D_dgrad_norms = checkpoint['D_dgrad_norms']
        self.G_ggrad_norms = checkpoint['G_ggrad_norms']
        self.G_qgrad_norms = checkpoint['G_qgrad_norms']
                
        self.D_sgrad_total_norm = checkpoint['D_sgrad_total_norm']
        self.D_dgrad_total_norm = checkpoint['D_dgrad_total_norm']
        self.G_ggrad_total_norm = checkpoint['G_ggrad_total_norm']
        self.G_qgrad_total_norm = checkpoint['G_qgrad_total_norm']
        
        self.init_optimisers()
        self.optimiser_D.load_state_dict(checkpoint['optimiser_D_state_dict'])
        self.optimiser_G.load_state_dict(checkpoint['optimiser_G_state_dict'])
                
        self.snapshot = checkpoint['snapshot']
        self.console_print = checkpoint['console_print']
        self.current_epoch = checkpoint['last_epoch']
        self.start_epoch = checkpoint['last_epoch'] + 1
        self.epoch_losses = checkpoint['epoch_losses']

        print(('\nCheckpoint loaded.\n' + 
               ' *- Last epoch {0} with loss {1}.\n' 
               ).format(checkpoint['last_epoch'], 
               checkpoint['last_epoch_loss']))
        print(' *- G optimiser:' +
              ' *- Current lr {0}, next update on epoch {1} to the value {2}'.format(
                      self.Goptim_lr, self.Goptim_lr_update_epoch, self.new_Goptim_lr)
              )
        print(' *- D optimiser:' +
              ' *- Current lr {0}, next update on epoch {1} to the value {2}'.format(
                self.Doptim_lr, self.Doptim_lr_update_epoch, self.new_Doptim_lr)
              )

        self.Gnet.train()
        self.Snet.train()
        self.Dnet.train()
        self.Qnet.train()
        assert(self.Gnet.training)
        

# --------------- #
# --- Testing --- #
# --------------- #
if __name__ == '__main__':
    config = {
        'Gnet_config': {
                'class_name': 'FullyConnectedGNet',
                'latent_dim': 100,
                'linear_dims': [256, 512, 1024],
                'dropout': 0.3,
                'image_channels': 1,
                'image_size': 32,
                'bias': True
                },
        
        'Snet_config': {
                'class_name': 'FullyConnectedSNet_Architecture2',
                'linear_dims': [512, 256],
                'dropout': 0,
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
                'batch_size': 128,
                'epochs': 100,
                'snapshot': 20, 
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
                'Gnet_progress_nimg': 100, 
                
                'grad_clip': False, 
                'Snet_D_grad_clip': None, 
                'Dnet_D_grad_clip': None, 
                'Gnet_G_grad_clip': None, 
                'Snet_G_grad_clip': None, 
                'Qnet_G_grad_clip': None, 
                
                'lambda_cat': 1,
                'lambda_con': 0.1, 
                
                'filename': 'gan',
                'random_seed': 1201,
                'exp_dir': 'models/DUMMY'
                }
        }
    
    model = InfoGAN_general(config)