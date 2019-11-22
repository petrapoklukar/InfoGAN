#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:21:00 2019

@author: petrapoklukar
"""

import torch.nn as nn
import torch.optim as optim
import torch
import itertools
import InfoGAN_models
import numpy as np

class InfoGAN(nn.Module):
    def __init__(self, config):
        super(InfoGAN, self).__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Models, weights
        self.init_discriminator()
        self.init_generator()
        self.init_weights()
        
        # Parameters
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
            optimizer_G = optim.Adam(self.generator.parameters(), 
                                     lr=optim_config['gen_lr'], 
                                     betas=(optim_config['gen_b1'], optim_config['gen_b2'])
                                     )
            print(' *- Initialised generator optimiser: Adam')
            optimizer_D = optim.Adam(self.discriminator.parameters(), 
                                     lr=optim_config['dis_lr'], 
                                     betas=(optim_config['dis_b1'], optim_config['dis_b2'])
                                     )
            print(' *- Initialised discriminator optimiser: Adam')
            optimizer_I = optim.Adam(
                itertools.chain(self.generator.parameters(), self.discriminator.parameters()), 
                lr=optim_config['minfo_lr'], 
                betas=(optim_config['minfo_b1'], optim_config['minfo_b2'])
            )
            print(' *- Initialised optimiser for mutual information: Adam')
            return optimizer_D, optimizer_G, optimizer_I
        else: 
            raise NotImplementedError(
                    'Optimiser {0} not recognized'.format(optim_type))
    
    def init_losses(self):
        """Initialises the losses"""
        # GAN Loss function
        self.gan_loss = torch.nn.BCELoss().to(self.device)
        # Discrete latent codes 
        self.categorical_loss = torch.nn.CrossEntropyLoss().to(self.device)
        # Continuous latent codes
        self.continuous_loss = torch.nn.MSELoss().to(self.device)
    
    def forward(self):
        pass
    

    def noise(self, batch_size, batch_dis_classes=None):
    
        # he usual uninformed noise
        z_noise = torch.empty((batch_size, self.z_dim), requires_grad=False, 
                              device=self.device).normal_() # b, x_dim
        
        # discrete code noise
        if batch_dis_classes is None:
            batch_dis_classes = np.random.randint(0, self.dis_classes, batch_size)
        dis_noise = np.zeros((batch_size, self.dis_classes)) 
        dis_noise[range(batch_size), batch_dis_classes] = 1.0 # bs, dis_classes
        
        # continuous code noise
        con_noise = torch.empty((batch_size, self.con_c_dim), requires_grad=False, 
                               device=self.device).uniform_(-1, 1)
        return z_noise, dis_noise, con_noise
    
    
    def train_infogan(self, train_dataloader):
        """Trains an InfoGAN."""
        
        optimizer_D, optimizer_G, optimizer_I = self.init_optimisers()
        self.init_losses()
        
        # TODO: not finished yet
        for self.current_epoch in range(self.start_epoch, self.epochs):
            self.generator.tran()
            self.discriminator.train()
            for i, x in enumerate(train_dataloader):
                
                # Ground truths
                batch_size = x.shape[0]
                real_x = x.to(self.device)        
                real_labels = torch.ones(batch_size, requires_grad=False, device=self.device)
                fake_labels = torch.zeros(batch_size, requires_grad=False, device=self.device)
        
        
                # ---------------------
                #  Train Discriminator
                # ---------------------
        
                optimizer_D.zero_grad()
        
                # Loss for real images
                real_pred, _, _ = self.discriminator(real_x)
                d_real_loss = self.gan_loss(real_pred, real_labels)
        
                # Loss for fake images
                z_noise, dis_noise, con_noise = self.noise(batch_size)
                fake_x = self.generator(z_noise, dis_noise, con_noise).detach()
                fake_pred, _, _ = self.discriminator(fake_x)
                d_fake_loss = self.gan_loss(fake_pred, fake_labels)
        
                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2
        
                d_loss.backward()
                optimizer_D.step()


                # -----------------
                #  Train Generator
                # -----------------
        
                optimizer_G.zero_grad()
        
                # Sample new noise and push it through the generator 
                z_noise, dis_noise, con_noise = self.noise(batch_size)
                fake_x = self.generator(z_noise, dis_noise, con_noise)
        
                # Loss measures generator's ability to fool the discriminator
                validity, _, _ = self.discriminator(fake_x)
                g_loss = self.gan_loss(validity, real_labels)
        
                g_loss.backward()
                optimizer_G.step()
        
        
                # ------------------
                # Information Loss
                # ------------------
        
                optimizer_I.zero_grad()
        
                # Sampled ground truth labels
                sampled_labels = np.random.randint(0, self.n_classes, batch_size)
                gt_labels = torch.LongTensor(sampled_labels, device=model.device)
        
                # Sample noise, labels and code as generator input
                z_noise, dis_noise, con_noise = self.noise(batch_size, batch_dis_classes=sampled_labels)
        
                gen_x = self.generator(z_noise, dis_noise, con_noise)
                _, pred_dis_code, pred_con_code = self.discriminator(gen_x)
        
                i_loss = self.lambda_cat * self.categorical_loss(pred_dis_code, gt_labels) + \
                    self.lambda_con * self.continuous_loss(pred_con_code, con_noise)
        
                i_loss.backward()
                optimizer_I.step()
        
                # --------------
                # Log Progress
                # --------------
        
#                print(
#                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [info loss: %f]"
#                    % (self.current_epoch, self.epochs, i, len(dataloader), d_loss.item(), g_loss.item(), i_loss.item())
#                )
#                batches_done = self.epochs * len(train_dataloader) + i
#                if batches_done % self.sample_interval == 0:
#                    sample_image(n_row=10, batches_done=batches_done)




if __name__ == '__main__':
    
    config = {
            'generator_config': {
                    'class_name': 'FullyConnectedGenerator',
                    'input_size': 74,
                    'output_size': 200
                    },
            
            'discriminator_config': {
                    'class_name': 'FullyConnectedDiscriminator',
                    'input_size': 200,
                    'n_continuous_codes': 10,
                    'n_categorical_codes': 4
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