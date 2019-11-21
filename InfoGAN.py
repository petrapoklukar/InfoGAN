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
        # Loss functions
        self.adversarial_loss = torch.nn.MSELoss().to(self.device)
        self.categorical_loss = torch.nn.CrossEntropyLoss().to(self.device)
        self.continuous_loss = torch.nn.MSELoss().to(self.device)
    
    def forward(self):
        pass
    
    def train_infogan(self, train_dataloader):
        """Trains an InfoGAN."""
        
        optimizer_D, optimizer_G, optimizer_I = self.init_optimisers()
        self.init_losses()
        
        # TODO: not finished yet
        for self.current_epoch in range(self.start_epoch, self.epochs):
            for i, x in enumerate(train_dataloader):
                
                batch_size = x.shape[0]
        
                # Adversarial ground truths
                valid = torch.ones(batch_size, requires_grad=False, device=self.device)
                fake = torch.zeros(batch_size, requires_grad=False, device=self.device)
        
                # Configure input
                real_x = x.to(self.device)
                labels = to_categorical(labels.numpy(), num_columns=opt.n_classes)
        
                # -----------------
                #  Train Generator
                # -----------------
        
                optimizer_G.zero_grad()
        
                # Sample noise and labels as generator input
                z = torch.empty((batch_size, self.latent_dim), requires_grad=False, 
                                device=self.device).normal_()
                label_input = to_categorical(np.random.randint(0, self.n_classes, batch_size), 
                                             num_columns=self.n_classes)
                code_input = torch.empty((batch_size, self.code_dim), requires_grad=False, 
                                device=self.device).uniform_(-1, 1)
        
                # Generate a batch of images
                gen_out = self.generator(z, label_input, code_input)
        
                # Loss measures generator's ability to fool the discriminator
                validity, _, _ = self.discriminator(gen_out)
                g_loss = self.adversarial_loss(validity, valid)
        
                g_loss.backward()
                optimizer_G.step()
        
                # ---------------------
                #  Train Discriminator
                # ---------------------
        
                optimizer_D.zero_grad()
        
                # Loss for real images
                real_pred, _, _ = self.discriminator(real_x)
                d_real_loss = self.adversarial_loss(real_pred, valid)
        
                # Loss for fake images
                fake_pred, _, _ = self.discriminator(gen_out.detach())
                d_fake_loss = self.adversarial_loss(fake_pred, fake)
        
                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2
        
                d_loss.backward()
                optimizer_D.step()
        
                # ------------------
                # Information Loss
                # ------------------
        
                optimizer_I.zero_grad()
        
                # Sample labels
                sampled_labels = np.random.randint(0, self.n_classes, batch_size)
        
                # Ground truth labels
                gt_labels = torch.LongTensor(sampled_labels, device=model.device)
        
                # Sample noise, labels and code as generator input
                z = torch.empty((batch_size, self.latent_dim), requires_grad=False, 
                                device=self.device).normal_(0, 1)
                label_input = to_categorical(sampled_labels, num_columns=self.n_classes)
                code_input = torch.empty((batch_size, self.code_dim), requires_grad=False, 
                                device=self.device).uniform_(-1, 1)
        
                gen_out = self.generator(z, label_input, code_input)
                _, pred_label, pred_code = self.discriminator(gen_out)
        
                i_loss = self.lambda_cat * self.categorical_loss(pred_label, gt_labels) + \
                    self.lambda_con * self.continuous_loss(pred_code, code_input)
        
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