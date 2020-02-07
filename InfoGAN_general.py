#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:21:00 2019

@author: petrapoklukar

An InfoGAN with both categorical and continuous structured latent codes.
Added tips from https://github.com/soumith/ganhacks.
"""

import torch.nn as nn
import torch.optim as optim
import torch
import InfoGAN_models
import numpy as np
import os
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

class InfoGAN(nn.Module):
    def __init__(self, config):
        super(InfoGAN, self).__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Models parameters
        self.data_config = config['data_config']
        self.z_dim = self.data_config['usual_noise_dim']
        self.cat_c_dim = self.data_config['structured_cat_dim']
        self.con_c_dim = self.data_config['structured_con_dim']

        # Training parameters
        train_config = config['train_config']
        self.batch_size = train_config['batch_size']
        self.epochs = train_config['epochs']
        self.current_epoch = None
        self.start_epoch = None
        self.snapshot = train_config['snapshot']
        self.console_print = train_config['console_print']
        
        self.gen_lr_schedule = train_config['gen_lr_schedule']
        self.init_gen_lr_schedule = train_config['gen_lr_schedule']
        self.dis_lr_schedule = train_config['dis_lr_schedule']
        self.init_dis_lr_schedule = train_config['dis_lr_schedule']

        self.lambda_cat = train_config['lambda_cat']
        self.lambda_con = train_config['lambda_con']
        
        self.exp_dir = train_config['exp_dir']
        self.save_path = train_config['exp_dir'] + '/' + train_config['filename']
        self.model_path = self.save_path + '_model.pt'
        self.create_dirs()
        
        # Fix random seed
        torch.manual_seed(train_config['random_seed'])
        np.random.seed(train_config['random_seed'])

    # ---------------------- #
    # --- Init functions --- #
    # ---------------------- #
    def create_dirs(self):
        """Creates folders for saving training logs."""
        self.test_dir = '{0}/Testing/'.format(self.exp_dir)
        self.train_dir = '{0}/Training/'.format(self.exp_dir)
        if (not os.path.isdir(self.test_dir)):
            os.makedirs(self.test_dir)
        if (not os.path.isdir(self.train_dir)):
            os.makedirs(self.train_dir)
        
    def init_generator(self):
        """Initialises the generator."""
        try:
            print(self.config['generator_config'])
            class_ = getattr(InfoGAN_models, self.config['generator_config']['class_name'])
            self.generator = class_(self.config['generator_config'], self.data_config).to(self.device)
            print(' *- Initialised generator: ', self.config['generator_config']['class_name'])
        except: 
            raise NotImplementedError(
                    'Generator class {0} not recognized'.format(
                            self.config['generator_config']['class_name']))
    
    def init_discriminator(self):
        """Initialises the discriminator."""
        try:
            class_ = getattr(InfoGAN_models, self.config['discriminator_config']['class_name'])
            self.discriminator = class_(self.config['discriminator_config'], self.data_config).to(self.device)
            print(' *- Initialised discriminator: ', self.config['discriminator_config']['class_name'])
        except: 
            raise NotImplementedError(
                    'Discriminator class {0} not recognized'.format(
                            self.config['discriminator_config']['class_name']))
    
    def init_Qnet(self):
        """Initialises the q network."""
        try:
            class_ = getattr(InfoGAN_models, self.config['Qnet_config']['class_name'])
            self.Qnet = class_(self.config['Qnet_config'], self.data_config).to(self.device)
            print(' *- Initialised QNet: ', self.config['Qnet_config']['class_name'])
        except: 
            raise NotImplementedError(
                    'QNet class {0} not recognized'.format(
                            self.config['Qnet_config']['class_name']))
    
    def init_weights(self):
        # TODO: tune this
        """Custom weight init"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                torch.nn.init.constant_(m.weight.data, 1)
                torch.nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, std=1e-3)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0)

    def init_optimisers(self):
        """Initialises the optimisers."""
        optim_config = self.config['optim_config']
        optim_type = optim_config['optim_type']
        if optim_type == 'Adam':
            # Generator & QNet optimiser
            self.optimiser_G = optim.Adam(
                    list(self.generator.parameters()) + list(self.Qnet.parameters()),
                    lr=optim_config['gen_lr'], 
                    betas=(optim_config['gen_b1'], optim_config['gen_b2']))
            print(' *- Initialised generator optimiser: Adam')
            # Discriminator optimiser
            self.optimiser_D = optim.Adam(
                    self.discriminator.parameters(), 
                    lr=optim_config['dis_lr'], 
                    betas=(optim_config['dis_b1'], optim_config['dis_b2']))
            print(' *- Initialised discriminator optimiser: Adam')
        else: 
            raise NotImplementedError(
                    'Optimiser {0} not recognized'.format(optim_type))
    
    def init_losses(self):
        """Initialises the losses"""
        # GAN Loss function
        self.gan_loss = torch.nn.BCELoss().to(self.device)
        # Discrete latent codes 
        self.categorical_loss = torch.nn.CrossEntropyLoss().to(self.device)
        # Continuous latent codes are model with the gaussian function above
    
    def gaussian_loss(self, x, mean, logvar):
        """Computes the exact Gaussian loss"""
        HALF_LOG_TWO_PI = 0.91893
        nonbatch_dims = list(range(1, len(x.shape)))
        var = torch.exp(logvar)   
        batch_loss = torch.sum(
                HALF_LOG_TWO_PI + 0.5 * logvar + 0.5 * ((x - mean) / var) ** 2,
                dim=nonbatch_dims) # batch_size
        avg_loss = torch.mean(batch_loss)
        return avg_loss

    # ---------------------------- #
    # --- Monitoring functions --- #
    # ---------------------------- #    
    def count_parameters(self, model):
        """Counts the total number of trainable parameters in the model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def print_model_params(self):
        """Prints specifications of the trainable parameters."""
        def print_trainable_param(model, n_params):
            print(' *- Model parameters: {0}'.format(n_params))
            for name, param in model.named_parameters():
                if param.requires_grad:
                    spacing = 1
                    print('{0:>2}{1}\n\t of dimension {2}'.format('', name, spacing),  
                          list(param.shape))
                    
        num_dparameters = self.count_parameters(self.discriminator) 
        print_trainable_param(self.discriminator, num_dparameters)
        self.config['discriminator_config']['n_model_params'] = num_dparameters
        
        num_gparameters = self.count_parameters(self.generator) 
        print_trainable_param(self.generator, num_gparameters)
        self.config['generator_config']['n_model_params'] = num_gparameters

        num_iparameters = self.count_parameters(self.Qnet) 
        print_trainable_param(self.Qnet, num_iparameters)
        self.config['Qnet_config']['n_model_params'] = num_iparameters

    def plot_snapshot_loss(self):
        """
        Plots discriminator, generator and Qnet losses at each snapshot 
        interval.
        """
        plt_data = np.stack(self.epoch_losses)
        plt_labels = ['d_loss', 'g_loss', 'i_loss']
        for i in range(3):
            plt.subplot(3,1,i+1)
            plt.plot(np.arange(self.snapshot)+(self.current_epoch//self.snapshot)*self.snapshot,
                     plt_data[self.current_epoch-self.snapshot+1:self.current_epoch+1, i], 
                     label=plt_labels[i])
            plt.ylabel(plt_labels[i])
            plt.xlabel('# epochs')
            plt.legend()
        plt.savefig(self.train_dir + 'SnapshotLosses_{0}'.format(self.current_epoch))
        plt.clf()
        plt.close()
    
    def plot_model_loss(self):
        """Plots epochs vs discriminator, generator and Qnet losses."""
        plt_data = np.stack(self.epoch_losses)
        plt_labels = ['d_loss', 'g_loss', 'i_loss']
        for i in range(3):
            plt.subplot(3,1,i+1)
            plt.plot(np.arange(self.current_epoch+1),
                     plt_data[:, i], 
                     label=plt_labels[i])
            plt.ylabel(plt_labels[i])
            plt.xlabel('# epochs')
            plt.legend()
        plt.savefig(self.save_path + '_Losses')
        plt.clf()
        plt.close()
        
        fig, ax = plt.subplots()
        ax.plot(plt_data[:, 0], 'go-', linewidth=3, label='D loss')
        ax.plot(plt_data[:, 1], 'bo--', linewidth=2, label='G loss')
        ax.plot()
        ax.legend()
        ax.set_xlim(0, self.epochs)
        ax.set(xlabel='# epochs', ylabel='loss', title='Discriminator vs Generator loss')
        plt.savefig(self.save_path + '_DvsGLoss')
        plt.close()
        
        fig2, ax2 = plt.subplots()
        ax2.plot(plt_data[:, 2], 'go-', linewidth=3, label='I loss')
        ax2.plot()
        ax2.set_xlim(0, self.epochs)
        ax2.set(xlabel='# epochs', ylabel='loss', title='Information loss')
        plt.savefig(self.save_path + '_ILoss')
        plt.close()
        
        fig2, ax2 = plt.subplots()
        ax.plot(plt_data[:, 0], 'go-', linewidth=2, label='D loss')
        ax.plot(plt_data[:, 1], 'bo-', linewidth=2, label='G loss')
        ax2.plot(plt_data[:, 2], 'ro-', linewidth=2, label='I loss')
        ax2.plot()
        ax2.set_xlim(0, self.epochs)
        ax2.set(xlabel='# epochs', ylabel='loss', title='All losses')
        plt.savefig(self.save_path + '_Allosses')
        plt.close()
    
    def sq_else_perm(self, img):
        """"""
        grayscale = True if img.shape[1] == 1 else False
        return img.squeeze() if grayscale else img.permute(1,2,0)
    
    def plot_image_grid(self, images, filename, directory, n=25):
        """Plots a grid of (generated) images."""
        n_subplots = np.sqrt(n).astype(int)
        plot_range = n_subplots ** 2
        images = self.sq_else_perm(images)
        for i in range(plot_range):
            plt.subplot(n_subplots, n_subplots, 1 + i)
            plt.axis('off')
            plt.imshow(images[i].detach().cpu().numpy())
        plt.savefig(directory + filename)
        plt.clf()
        plt.close()
        
    def format_loss(self, losses_list):
        """Rounds the loss and returns an np array"""
        reformatted = list(map(lambda x: round(x.item(), 2), losses_list))
        reformatted.append(self.current_epoch)
        return np.array(reformatted)
    
    def evaluate(self, data_loader):
        """"Evaluatates the performance of a InfoGAN"""
        self.eval()
        assert(not self.generator.training)

        sum_d_loss = 0.0
        sum_g_loss = 0.0
        sum_i_loss = 0.0

        batch_size = data_loader.batch_size
        latent_var = np.zeros((len(data_loader)*batch_size, self.latent_size))
        labels = np.zeros(len(data_loader)*batch_size)
        for i, x in enumerate(data_loader):
            label =[]
            if isinstance(x, list):
                label = x[1].to(self.device)
                x = x[0].to(self.device)
            d_loss, g_loss, i_loss, z_noise, dis_noise, con_noise = self.forward(x)
            
            sum_d_loss += d_loss.item()
            sum_g_loss += g_loss.item()
            sum_i_loss += i_loss.item()

            lv = np.copy(con_noise.detach().numpy())
            c_batch_size = lv.shape[0]
            latent_var[i*batch_size:i*batch_size+c_batch_size,:] = lv
            if len(label)>0:
                labels[i*batch_size:i*batch_size+c_batch_size] = label

        print('discriminator_loss: %.6e' %(sum_d_loss / len(data_loader)))
        print('generator_loss: %.6e' %(sum_g_loss / len(data_loader)))
        print('Qnet_loss: %.6e' %(sum_i_loss / len(data_loader)))
        return latent_var, labels
    
    # -------------------------- #
    # --- Training functions --- #
    # -------------------------- #
    def forward(self, x):
        """Forward pass through InfoGAN"""
        self.eval()
        # Ground truths
        batch_size = x.shape[0]
        real_x = x.to(self.device)        
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)

        # Loss for real images
        real_pred, _ = self.discriminator(real_x)
        d_real_loss = self.gan_loss(real_pred, real_labels)

        # Loss for fake images
        z_noise, dis_noise, con_noise = self.noise(batch_size)
        fake_x = self.generator((z_noise, dis_noise, con_noise)).detach()
        fake_pred = self.discriminator(fake_x)
        d_fake_loss = self.gan_loss(fake_pred, fake_labels)

        # Total discriminator loss
        d_loss = d_real_loss + d_fake_loss

        # - THE USUAL GENERATOR LOSS        
        # Loss measures generator's ability to fool the discriminator
        # Push fake samples through the update discriminator
        fake_pred, fake_features = self.discriminator(fake_x)
        g_loss = self.gan_loss(fake_pred, real_labels)

        # - INFORMATION LOSS
        # Sampled ground truth labels, see eq 5 in the paper
        sampled_labels = np.random.randint(0, self.n_classes, batch_size)
        gt_labels = torch.LongTensor(sampled_labels).to(self.device)

        # Sample noise, labels and code as generator input
        z_noise, dis_noise, con_noise = self.noise(
                batch_size, batch_dis_classes=sampled_labels)

        # Push it through QNet
        gen_x = self.generator((z_noise, dis_noise, con_noise))
        _, gen_features = self.discriminator(gen_x) 
        pred_dis_code, pred_con_mean, pred_con_logvar = self.QNet(gen_features)

        i_loss = self.lambda_cat * self.categorical_loss(pred_dis_code, gt_labels) + \
            self.lambda_con * self.gaussian_loss(con_noise, pred_con_mean, pred_con_logvar)
        
        g_loss += i_loss 
        return d_loss, g_loss, i_loss, z_noise, dis_noise, con_noise
        
    def noise(self, batch_size, batch_cat_c_dim=None):
        """
        Generates uninformed noise, structured discrete noise and 
        structured continuous noise.
        """
        # the usual uninformed noise
        z_noise = torch.empty((batch_size, self.z_dim), requires_grad=False, 
                              device=self.device).normal_() # b, x_dim
        
        # structured discrete code noise
        if batch_cat_c_dim is None:
            # Generates a batch of random discrete codes if no is specified
            batch_cat_c_dim = np.random.randint(0, self.cat_c_dim, batch_size)
        dis_noise = np.zeros((batch_size, self.cat_c_dim)) 
        dis_noise[range(batch_size), batch_cat_c_dim] = 1.0 # bs, dis_classes
        dis_noise = torch.Tensor(dis_noise).to(self.device) 
        
        # structured continuous code noise
        con_noise = torch.empty((batch_size, self.con_c_dim), requires_grad=False, 
                               device=self.device).uniform_(-1, 1)
        return z_noise, dis_noise, con_noise

    def sample_fixed_noise(self, n_samples, noise_dim, ntype):
        """Samples one type of noise only"""
        if ntype == 'uniform':
            return torch.empty((n_samples, noise_dim), device=self.device).uniform_(-1, 1)
        elif ntype == 'normal':
            return torch.empty((n_samples, noise_dim), device=self.device).normal_()
        else:
            raise ValueError('Noise type {0} not recognised.'.format(ntype))
    
    def train_infogan(self, train_dataloader, chpnt_path=''):
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
            self.init_discriminator()
            self.init_generator()
            self.init_Qnet()
            self.init_weights()
            self.start_gen_epoch, self.gen_lr = self.gen_lr_schedule.pop(0)
            self.start_dis_epoch, self.dis_lr = self.dis_lr_schedule.pop(0)
            assert(self.start_gen_epoch == self.start_dis_epoch)
            self.start_epoch = self.start_dis_epoch
            try:
                self.gen_lr_update_epoch, self.new_gen_lr = self.gen_lr_schedule.pop(0)
                self.dis_lr_update_epoch, self.new_dis_lr = self.dis_lr_schedule.pop(0)
            except:
                self.gen_lr_update_epoch, self.new_gen_lr = self.start_epoch - 1, self.gen_lr
                self.dis_lr_update_epoch, self.new_dis_lr = self.start_epoch - 1, self.dis_lr

            self.init_optimisers()
            self.epoch_losses = []
            print((' *- Generator' + 
                   '    *- Learning rate: {0}\n' + 
                   '    *- Next lr update at {1} to the value {2}\n' + 
                   '    *- Remaining lr schedule: {3}'
                   ).format(self.gen_lr, self.gen_lr_update_epoch, self.new_gen_lr, 
                   self.gen_lr_schedule))            
            print((' *- Discriminator' + 
                   '    *- Learning rate: {0}\n' + 
                   '    *- Next lr update at {1} to the value {2}\n' + 
                   '    *- Remaining lr schedule: {3}'
                   ).format(self.dis_lr, self.dis_lr_update_epoch, self.new_dis_lr, 
                   self.dis_lr_schedule))            

        self.print_model_params()
        self.init_losses()
        print('\nStarting to train the model...\n' )        
        for self.current_epoch in range(self.start_epoch, self.epochs):
            self.train()
            assert(self.generator.training)
            
            epoch_loss = np.zeros(4)
            for i, x in enumerate(train_dataloader):
                
                # Ground truths
                batch_size = x.shape[0]
#                x = x.view(batch_size, -1) # DEBUG 553 on yumidata
                real_x = x.to(self.device)        
                real_labels = torch.ones(batch_size, 1, device=self.device)
                fake_labels = torch.zeros(batch_size, 1, device=self.device)
                
                # ------------------------------- #
                # --- Train the Discriminator --- #
                # ------------------------------- #
                self.optimiser_D.zero_grad()
        
                # Loss for real images
                real_pred, _ = self.discriminator(real_x)                
                assert torch.sum(torch.isnan(real_pred)) == 0, real_pred
                assert(real_pred >= 0.).all(), real_pred
                assert(real_pred <= 1.).all(), real_pred
                d_real_loss = self.gan_loss(real_pred, real_labels)
        
                # Loss for fake images
                z_noise, dis_noise, con_noise = self.noise(batch_size)
                fake_x = self.generator((z_noise, dis_noise, con_noise)).detach()
                assert torch.sum(torch.isnan(fake_x)) == 0, fake_x
                fake_pred, _ = self.discriminator(fake_x)
                assert(fake_pred >= 0.).all(), fake_pred
                assert(fake_pred <= 1.).all(), fake_pred
                d_fake_loss = self.gan_loss(fake_pred, fake_labels)
        
                # Total discriminator loss
                d_loss = d_real_loss + d_fake_loss
        
                d_loss.backward()
                self.optimiser_D.step()

                # ---------------------------------- #
                # --- Train the Generator & QNet --- #
                # ---------------------------------- #
                self.optimiser_G.zero_grad()
        
                # - THE USUAL GENERATOR LOSS        
                # Loss measures generator's ability to fool the discriminator
                # Push fake samples through the update discriminator
                fake_pred, fake_features = self.discriminator(fake_x)
                g_loss = self.gan_loss(fake_pred, real_labels)
        
                # - INFORMATION LOSS
                # Sampled ground truth labels, see eq 5 in the paper
                sampled_labels = np.random.randint(0, self.cat_c_dim, batch_size)
                gt_labels = torch.LongTensor(sampled_labels).to(self.device)
        
                # Sample noise, labels and code as generator input
                z_noise, dis_noise, con_noise = self.noise(
                        batch_size, batch_cat_c_dim=sampled_labels)
        
                # Push it through QNet
                gen_x = self.generator((z_noise, dis_noise, con_noise))
                _, gen_features = self.discriminator(gen_x) 
                pred_dis_code, pred_con_mean, pred_con_logvar = self.Qnet(gen_features)
        
                i_loss = self.lambda_cat * self.categorical_loss(pred_dis_code, gt_labels) + \
                    self.lambda_con * self.gaussian_loss(con_noise, pred_con_mean, pred_con_logvar)
                
                g_loss += i_loss 
                g_loss.backward()
                self.optimiser_G.step()
                epoch_loss += self.format_loss([d_loss, g_loss, i_loss])
        
            # ------------------------ #
            # --- Log the training --- #
            # ------------------------ #      
            epoch_loss /= len(train_dataloader)
            print(
                "[Epoch %d/%d] [D loss: %f] [G loss: %f] [info loss: %f]"
                % (self.current_epoch, self.epochs, epoch_loss[0], 
                   epoch_loss[1], epoch_loss[2]))
                    
            # TODO: add logger here
            self.epoch_losses.append(epoch_loss)
            self.plot_model_loss()       
            self.save_checkpoint(epoch_loss)
            
            if (self.current_epoch + 1) % self.snapshot == 0:
    
                # Save the checkpoint & logs, plot snapshot losses
                self.save_checkpoint(epoch_loss, keep=True)
                self.save_logs()

                # Plot snapshot losses
                self.plot_snapshot_loss()
                
                # Plot images generated from random noise
                z_noise, dis_noise, con_noise = self.noise(25)
                gen_x = self.generator((z_noise, dis_noise, con_noise))
#                gen_x_plotrescale = (gen_x + 1.) / 2.0 # Cause of tanh activation
                filename = 'genImages' + str(self.current_epoch)
                self.plot_image_grid(gen_x, filename, self.train_dir, n=25)
                
                # Plot images generated from the first discrete code
                z_noise, dis_noise, con_noise = self.noise(25, batch_cat_c_dim=0)
                gen_x = self.generator((z_noise, dis_noise, con_noise))
#                gen_x_plotrescale = (gen_x + 1.) / 2.0 # Cause of tanh activation
                filename = 'genImages_c0' + str(self.current_epoch)
                self.plot_image_grid(gen_x, filename, self.train_dir, n=25)
        
        # ---------------------- #
        # --- Save the model --- #
        # ---------------------- # 
        print('Training completed.')
        self.plot_model_loss()
        self.eval()
        torch.save({
                'discriminator': self.discriminator.state_dict(), 
                'generator': self.generator.state_dict(), 
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
                    '*- Generator learning rate schedule: {0}\n'.format(self.init_gen_lr_schedule),
                    '*- Discriminator learning rate schedule: {0}\n'.format(self.init_dis_lr_schedule),
                    '*- Training epoch losses: (model_loss, recon_loss, kl_loss)\n'
                    ])
            f.writelines(list(map(
                    lambda t: '{0:>3} Epoch {4}: ({1:.2f}, {2:.2f}, {3:.2f})\n'.format(
                            '', t[0], t[1], t[2],  t[3]), 
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
                
                'discriminator_state_dict': self.discriminator.state_dict(),
                'generator_state_dict': self.generator.state_dict(),
                'Qnet_state_dict': self.Qnet.state_dict(),                
            
                'optimiser_D_state_dict': self.optimiser_D.state_dict(),
                'optimiser_G_state_dict': self.optimiser_G.state_dict(),
                
                'last_epoch_loss': epoch_loss,
                'epoch_losses': self.epoch_losses,

                'lambda_cat': self.lambda_cat, 
                'lambda_con': self.lambda_con, 

                'snapshot': self.snapshot,
                'console_print': self.console_print,
                
                'current_gen_lr': self.gen_lr,
                'gen_lr_update_epoch': self.gen_lr_update_epoch, 
                'new_gen_lr': self.new_gen_lr, 
                'gen_lr_schedule': self.gen_lr_schedule,
                
                'current_dis_lr': self.dis_lr,
                'dis_lr_update_epoch': self.dis_lr_update_epoch, 
                'new_dis_lr': self.new_dis_lr, 
                'dis_lr_schedule': self.dis_lr_schedule
                }
        torch.save({**training_dict, **self.config}, path)
        print(' *- Saved {1} checkpoint {0}.'.format(self.current_epoch, checkpoint_type))
    
        
    def load_checkpoint(self, path):
        """
        Loads a checkpoint and initialises the models to continue training.
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.init_discriminator()
        self.init_generator()
        self.init_Qnet()
        
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.Qnet.load_state_dict(checkpoint['Qnet_state_dict'])
        
        self.gen_lr = checkpoint['current_gen_lr']
        self.gen_lr_update_epoch = checkpoint['gen_lr_update_epoch']
        self.new_gen_lr = checkpoint['new_gen_lr']
        self.gen_lr_schedule = checkpoint['gen_lr_schedule']
        
        self.dis_lr = checkpoint['current_dis_lr']
        self.dis_lr_update_epoch = checkpoint['dis_lr_update_epoch']
        self.new_dis_lr = checkpoint['new_dis_lr']
        self.dis_lr_schedule = checkpoint['dis_lr_schedule']
        
        self.init_optimisers()
        self.optimiser_D.load_state_dict(checkpoint['optimiser_D_state_dict'])
        self.optimiser_G.load_state_dict(checkpoint['optimiser_G_state_dict'])
                
        self.snapshot = checkpoint['snapshot']
        self.console_print = checkpoint['console_print']
        self.start_epoch = checkpoint['last_epoch'] + 1
        self.epoch_losses = checkpoint['epoch_losses']

        print(('\nCheckpoint loaded.\n' + 
               ' *- Last epoch {0} with loss {1}.\n' 
               ).format(checkpoint['last_epoch'], 
               checkpoint['last_epoch_loss']))
        print(' *- Generator:' +
              ' *- Current lr {0}, next update on epoch {1} to the value {2}'.format(
                      self.gen_lr, self.gen_lr_update_epoch, self.new_gen_lr)
              )
        print(' *- Discriminator:' +
              ' *- Current lr {0}, next update on epoch {1} to the value {2}'.format(
                self.dis_lr, self.dis_lr_update_epoch, self.new_dis_lr)
              )

        self.train()
        assert(self.Qnet.training)
        
    def load_model(self, eval_config):
        """Loads a trained InfoGAN model into eval mode"""   

        filename = eval_config['filepath']
        model_dict = torch.load(filename, map_location=self.device)
        data_config = self.config['data_config']

        # Load the Discriminator        
        d_config = self.config['discriminator_config']
        discriminator = getattr(InfoGAN_models, d_config['class_name'])
        self.discriminator = discriminator(d_config, data_config).to(self.device)
        d_model = model_dict['discriminator']
        if eval_config['load_checkpoint']:
            self.discriminator.load_state_dict(d_model['model_state_dict'])
            print(' *- Loaded discriminator checkpoint.')
        else:
            self.discriminator.load_state_dict(d_model)

        # Load the Generator
        g_config = self.config['generator_config']
        generator = getattr(InfoGAN_models, g_config['class_name'])
        self.generator = generator(g_config, data_config).to(self.device)
        g_model = model_dict['generator']
        if eval_config['load_checkpoint']:
            self.generator.load_state_dict(g_model['model_state_dict'])
            print(' *- Loaded checkpoint.')
        else:
            self.generator.load_state_dict(g_model)

        # Load the QNet            
        Qnet_config = self.config['Qnet_config']
        Qnet = getattr(InfoGAN_models, Qnet_config['class_name'])
        self.Qnet = Qnet(Qnet_config, data_config).to(self.device)
        Qnet_model = model_dict['Qnet']
        if eval_config['load_checkpoint']:
            self.Qnet.load_state_dict(Qnet_model['model_state_dict'])
            print(' *- Loaded checkpoint.')
        else:
            self.Qnet.load_state_dict(Qnet_model)
        
        self.eval()
        assert(not self.generator.training)

# --------------- #
# --- Testing --- #
# --------------- #
if __name__ == '__main__':
    
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
                    'epochs': 5,
                    'snapshot': 2, 
                    'console_print': 1,
                    'gen_lr_schedule': [(0, 1e-3)],
                    'dis_lr_schedule': [(0, 2e-4)],
                    'lambda_cat': 1,
                    'lambda_con': 0.1, 
                    'filename': 'infogan',
                    'random_seed': 1201
                    },
                    
            'eval_config': {
                    'filepath': 'models/InfoGAN_test/infogan_q.pt',
                    'load_checkpoint': False
                    }
            }
    
    model = InfoGAN(config)