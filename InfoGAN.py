#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:21:00 2019

@author: petrapoklukar

Added tips from https://github.com/soumith/ganhacks
"""

import torch.nn as nn
import torch.optim as optim
import torch
import InfoGAN_models
import numpy as np
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

class InfoGAN(nn.Module):
    def __init__(self, config):
        super(InfoGAN, self).__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Models parameters
        gen_config = config['generator_config']
        self.z_dim = gen_config['usual_noise_dim']
        self.dis_classes = gen_config['structured_cat_dim']
        self.con_c_dim = gen_config['structured_con_dim']

        # Training parameters
        train_config = config['train_config']
        self.batch_size = train_config['batch_size']
        self.epochs = train_config['epochs']
        self.current_epoch = None
        self.start_epoch = None
        self.snapshot = train_config['snapshot']
        self.console_print = train_config['console_print']
        
        self.lr_schedule = train_config['lr_schedule']
        self.init_lr_schedule = train_config['lr_schedule']

        self.lambda_cat = train_config['lambda_cat']
        self.lambda_con = train_config['lambda_con']
        
        self.save_path = train_config['exp_dir'] + '/' + train_config['filename']
        self.model_path = self.save_path + '_model.pt'
        
        # Fix random seed
        torch.manual_seed(train_config['random_seed'])
        np.random.seed(train_config['random_seed'])


    def init_generator(self):
        """Initialises the generator."""
        try:
            class_ = getattr(InfoGAN_models, self.config['generator_config']['class_name'])
            self.generator = class_(self.config['generator_config']).to(self.device)
            print(' *- Initialised generator: ', self.config['generator_config']['class_name'])
        except: 
            raise NotImplementedError(
                    'Generator class {0} not recognized'.format(
                            self.config['generator_config']['class_name']))
    
    def init_discriminator(self):
        """Initialises the discriminator."""
        try:
            class_ = getattr(InfoGAN_models, self.config['discriminator_config']['class_name'])
            self.discriminator = class_(self.config['discriminator_config']).to(self.device)
            print(' *- Initialised discriminator: ', self.config['discriminator_config']['class_name'])
        except: 
            raise NotImplementedError(
                    'Discriminator class {0} not recognized'.format(
                            self.config['discriminator_config']['class_name']))
    
    def init_Qnet(self):
        """Initialises the q network."""
        try:
            class_ = getattr(InfoGAN_models, self.config['Qnet_config']['class_name'])
            self.Qnet = class_(self.config['Qnet_config']).to(self.device)
            print(' *- Initialised QNet: ', self.config['Qnet_config']['class_name'])
        except: 
            raise NotImplementedError(
                    'Discriminator class {0} not recognized'.format(
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
    
    def init_losses(self):
        """Initialises the losses"""
        # GAN Loss function
        self.gan_loss = torch.nn.BCELoss().to(self.device)
        # Discrete latent codes 
        self.categorical_loss = torch.nn.CrossEntropyLoss().to(self.device)
        # Continuous latent codes are model with the gaussian function above
    
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
        plt.savefig(self.save_path + '_SnapshotLosses_{0}'.format(self.current_epoch))
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
        ax.plot(plt_data[:, 2], 'go-', linewidth=3, label='D loss')
        ax.plot(plt_data[:, 1], 'bo--', linewidth=2, label='G loss')
        ax.plot()
        ax.legend()
        ax.set_xlim(0, self.epochs)
        ax.set(xlabel='# epochs', ylabel='loss', title='Discriminator vs Generator loss')
        plt.savefig(self.save_path + '_DvsGLoss')
        plt.close()
        
        fig2, ax2 = plt.subplots()
        ax2.plot(plt_data[:, 0], 'go-', linewidth=3, label='I loss')
        ax2.plot()
        ax2.set_xlim(0, self.epochs)
        ax2.set(xlabel='# epochs', ylabel='loss', title='Information loss')
        plt.savefig(self.save_path + '_ILoss')
        plt.close()
        
    def format_loss(self, losses_list):
        """Rounds the loss and returns an np array"""
        reformatted = list(map(lambda x: round(x.item(), 2), losses_list))
        return np.array(reformatted)
    
    def forward(self):
        pass
    
    def noise(self, batch_size, batch_dis_classes=None):
        """
        Generates uninformed noise, structured discrete noise and 
        structured continuous noise.
        """
        # the usual uninformed noise
        z_noise = torch.empty((batch_size, self.z_dim), requires_grad=False, 
                              device=self.device).normal_() # b, x_dim
        
        # structured discrete code noise
        if batch_dis_classes is None:
            batch_dis_classes = np.random.randint(0, self.dis_classes, batch_size)
        dis_noise = np.zeros((batch_size, self.dis_classes)) 
        dis_noise[range(batch_size), batch_dis_classes] = 1.0 # bs, dis_classes
        
        # structured continuous code noise
        con_noise = torch.empty((batch_size, self.con_c_dim), requires_grad=False, 
                               device=self.device).uniform_(-1, 1)
        return z_noise, dis_noise, con_noise
    
    
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
            self.start_epoch, self.lr = self.lr_schedule.pop(0)
            try:
                self.lr_update_epoch, self.new_lr = self.lr_schedule.pop(0)
            except:
                self.lr_update_epoch, self.new_lr = self.start_epoch - 1, self.lr

            self.init_optimisers()
            self.epoch_losses = []
            print((' *- Learning rate: {0}\n' + 
                   ' *- Next lr update at {1} to the value {2}\n' + 
                   ' *- Remaining lr schedule: {3}'
                   ).format(self.lr, self.lr_update_epoch, self.new_lr, 
                   self.lr_schedule))            

        self.print_model_params()
        self.init_losses()
        print('\nStarting to train the model...\n' )        
        for self.current_epoch in range(self.start_epoch, self.epochs):
            self.generator.tran()
            self.discriminator.train()
            self.Qnet.train()
            epoch_loss = np.zeros(4)
            for i, x in enumerate(train_dataloader):
                
                # Ground truths
                batch_size = x.shape[0]
                real_x = x.to(self.device)        
                real_labels = torch.ones(batch_size, device=self.device)
                fake_labels = torch.zeros(batch_size, device=self.device)
        
                # ------------------------------- #
                # --- Train the Discriminator --- #
                # ------------------------------- #
                self.optimiser_D.zero_grad()
        
                # Loss for real images
                real_pred, _ = self.discriminator(real_x)
                d_real_loss = self.gan_loss(real_pred, real_labels)
        
                # Loss for fake images
                z_noise, dis_noise, con_noise = self.noise(batch_size)
                fake_x = self.generator(z_noise, dis_noise, con_noise).detach()
                fake_pred = self.discriminator(fake_x)
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
                sampled_labels = np.random.randint(0, self.n_classes, batch_size)
                gt_labels = torch.LongTensor(sampled_labels, device=model.device)
        
                # Sample noise, labels and code as generator input
                z_noise, dis_noise, con_noise = self.noise(
                        batch_size, batch_dis_classes=sampled_labels)
        
                # Push it through QNet
                gen_x = self.generator(z_noise, dis_noise, con_noise)
                _, gen_features = self.discriminator(gen_x) 
                pred_dis_code, pred_con_mean, pred_con_logvar = self.QNet(gen_features)
        
                i_loss = self.lambda_cat * self.categorical_loss(pred_dis_code, gt_labels) + \
                    self.lambda_con * self.gaussian_loss(con_noise, pred_con_mean, pred_con_logvar)
                
                g_loss += i_loss 
                g_loss.backward()
                self.optimiser_G.step()
                epoch_loss += self.format_loss([d_loss, g_loss, i_loss, self.current_epoch])
        
            # ------------------------ #
            # --- Log the training --- #
            # ------------------------ #      
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [info loss: %f]"
                % (self.current_epoch, self.epochs, i, len(train_dataloader), 
                   epoch_loss[0], epoch_loss[1], epoch_loss[2]))
                    
            # TODO: add logger here
            epoch_loss /= len(train_dataloader)
            self.epoch_losses.append(epoch_loss)
            self.plot_model_loss()       
            self.save_checkpoint(epoch_loss)
            
            if (self.current_epoch + 1) % self.snapshot == 0:
    
                # Save the checkpoint & logs, plot snapshot losses
                self.save_checkpoint(epoch_loss, keep=True)
                self.save_logs()

                # Plot snapshot losses
                self.plot_snapshot_loss()
                
        
        # ------------------------ #
        # --- Save the model --- #
        # ------------------------ # 
        print('Training completed.')
        self.plot_model_loss()
        self.model.eval()
        torch.save(self.model.state_dict(), self.model_path)       
        self.save_logs()
        
    
    def save_logs(self, ):
        """Saves a txt file with logs"""
        log_filename = self.save_path + '_logs.txt'
        epoch_losses = np.stack(self.epoch_losses)
        
        with open(log_filename, 'w') as f:
            f.write('Model {0}\n\n'.format(self.config['filename']))
            f.write( str(self.config) )
            f.writelines(['\n\n', 
                    '*- Model path: {0}\n'.format(self.model_path),
                    '*- Learning rate schedule: {0}\n'.format(self.init_lr_schedule),
                    '*- Training epoch losses: (model_loss, recon_loss, kl_loss)\n'
                    ])
            f.writelines(list(map(
                    lambda t: '{0:>3} Epoch {4}: ({1:.2f}, {2:.2f}, {3:.2f})\n'.format(
                            '', t[0], t[1], t[2],  t[3]), 
                    epoch_losses)))
        print(' *- Model saved.\n')
    
    
    def save_checkpoint(self, epoch_loss, keep=False):
        """
        Saves a checkpoint during the training.
        """
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
                'current_lr': self.lr,
                'lr_update_epoch': self.lr_update_epoch, 
                'new_lr': self.new_lr, 
                'lr_schedule': self.lr_schedule
                }
        torch.save({**training_dict, **self.opt}, path)
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
        
        self.lr = checkpoint['current_lr']
        self.lr_update_epoch = checkpoint['lr_update_epoch']
        self.new_lr = checkpoint['new_lr']
        self.lr_schedule = checkpoint['lr_schedule']
        
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
        print(' *- Current lr {0}, next update on epoch {1} to the value {2}'.format(
                self.lr, self.lr_update_epoch, self.new_lr)
              )
        
        self.discriminator.train()
        self.generator.train()
        self.Qnet.train()
        




if __name__ == '__main__':
    
    config = {
            'generator_config': {
                    'class_name': 'FullyConnectedGenerator',
                    'usual_noise_dim': 1,
                    'structured_cat_dim': 0, 
                    'structured_con_dim': 6,
                    'layer_dims': [7, 128, 256, 256, 512, 200]
                    },
            
            'discriminator_config': {
                    'class_name': 'FullyConnectedDiscriminator',
                    'layer_dims': [200, 1000, 500, 250, 250, 50]
                    },
                    
            'Qnet_config': {
                    'class_name': 'QNet',
                    'last_layer_dim': 50,
                    'n_continuous_codes': 6,
                    'n_categorical_codes': 0
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
                    'epochs': 1000,
                    'snapshot': 100, 
                    'console_print': 1,
                    'lr_schedule': [(0, 1e-3), (100, 1e-4)],
                    'lambda_cat': 1,
                    'lambda_con': 0.1, 
                    'exp_dir': '/some_dummy_path',
                    'filename': 'infogan',
                    'random_seed': 1201
                    }
            }
    
    model = InfoGAN(config)