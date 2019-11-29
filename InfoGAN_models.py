#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 09:17:10 2019

@author: petrapoklukar

Base implementation thanks to https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/infogan/infogan.py
"""

import torch.nn as nn
import torch

# --------------------------------------------------------------- #
# --- To keep track of the dimensions of convolutional layers --- #
# --------------------------------------------------------------- #
class TempPrintShape(nn.Module):
    def __init__(self, message):
        super(TempPrintShape, self).__init__()
        self.message = message
        
    def forward(self, feat):
        print(self.message, feat.shape)
        return feat 
    

# ------------ #
# --- QNet --- #
# ------------ #
class QNet(nn.Module):
    def __init__(self, config):
        super(QNet, self).__init__()
        self.last_layer_dim = config['last_layer_dim']
        self.n_categorical_codes = config['n_categorical_codes']
        self.n_continuous_codes = config['n_continuous_codes']

        # Model structured continuous code as Gaussian
        self.con_layer_mean = nn.Linear(self.last_layer_dim, self.n_continuous_codes)
        self.con_layer_logvar = nn.Linear(self.last_layer_dim, self.n_continuous_codes)
        
        # M´Structured categorical code
        self.cat_layer = nn.Sequential(
                nn.Linear(self.last_layer_dim, self.n_categorical_codes), 
                nn.Softmax(dim=-1))

    def forward(self, x):
        cat_code = self.cat_layer(x) # Structured categorical code
        con_code_mean = self.con_layer_mean(x) # Structured continuous code
        con_code_logvar = self.con_layer_logvar(x) # Structured continuous code
        return cat_code, con_code_mean, con_code_logvar


# ---------------------------------------- #
# --- Linear Generator & Distriminator --- #
# ---------------------------------------- #
class FullyConnectedGenerator(nn.Module):
    def __init__(self, config):
        super(FullyConnectedGenerator, self).__init__()
        self.gen_config = config
        self.layer_dims = config['layer_dims']
        
        self.generator = nn.Sequential()
        for i in range(1, len(self.layer_dims) - 1):
            self.generator.add_module('lin' + str(i), nn.Linear(
                    self.layer_dims[i-1], self.layer_dims[i]))
            self.generator.add_module('bn' + str(i), nn.BatchNorm1d(
                    self.layer_dims[i]))
            self.generator.add_module('relu' + str(i), nn.LeakyReLU(0.1))
        self.generator.add_module('lin_last', nn.Linear(
                self.layer_dims[-2], self.layer_dims[-1]))
        self.generator.add_module('tanh_last', nn.Tanh())
    
    def forward(self, noise, labels, code):
        gen_input = torch.cat((noise, labels, code), -1)
        out = self.generator(gen_input)
        return out


class FullyConnectedDiscriminator(nn.Module):
    def __init__(self, config):
        super(FullyConnectedDiscriminator, self).__init__()
        self.dis_config = config
        self.layer_dims = config['layer_dims']
        
        self.discriminator = nn.Sequential()
        for i in range(1, len(self.layer_dims) - 1):
            self.discriminator.add_module('lin' + str(i), nn.Linear(
                    self.layer_dims[i-1], self.layer_dims[i]))
            self.discriminator.add_module('bn' + str(i), nn.BatchNorm1d(
                    self.layer_dims[i]))
            self.discriminator.add_module('relu' + str(i), nn.LeakyReLU(0.1))
        self.discriminator.add_module('lin_last', nn.Linear(
                self.layer_dims[-2], self.layer_dims[-1]))

        # Output layers
        self.d_out = nn.Sequential(
                nn.Linear(self.layer_dims[-1], 1),
                nn.Sigmoid())

    def forward(self, x):
        out = self.discriminator(x) # Needed for QNet
        validity = self.d_out(out) # Usual discriminator output
        return validity, out
    
    
# ----------------------------------------------- #
# --- Convolutional Generator & Distriminator --- #
# ----------------------------------------------- #
class ConvolutionalGenerator(nn.Module):
    def __init__(self, config):
        super(ConvolutionalGenerator, self).__init__()
        self.input_size = config['input_size']
        self.output_size = config['output_size']
        self.init_size = config['init_size'] # 8
        linlayer_dims = [self.input_size, 1024, self.init_size*self.init_size*128]
        self.convlayer_dims = [128, 64, 1]
        
        self.lin = nn.Sequential()
        for i in range(len(linlayer_dims) - 1):
            self.lin.add_module('lin' + str(i), nn.Linear(linlayer_dims[i], linlayer_dims[i+1]))
            self.lin.add_module('lin_relu' + str(i), nn.LeakyReLU(0.1))
            self.lin.add_module('lin_bn' + str(i), nn.BatchNorm1d(linlayer_dims[i+1]))
        
        self.conv = nn.Sequential()
        for i in range(1, len(self.convlayer_dims)):
            self.conv.add_module('conv_upsample' + str(i), nn.Upsample(scale_factor=4))
            self.conv.add_module('conv2d' + str(i), nn.Conv2d(
                    self.convlayer_dims[i-1], self.convlayer_dims[i], 4, stride=2))
            self.conv.add_module('conv_relu' + str(i), nn.LeakyReLU(0.1))
            self.conv.add_module('conv_bn' + str(i), nn.BatchNorm2d(self.convlayer_dims[i]))


    def forward(self, noise, labels, code):
        gen_input = torch.cat((noise, labels, code), -1)
        out = self.lin(gen_input)
        out = out.view(out.shape[0], self.convlayer_dims[0], self.init_size, self.init_size)
        img = self.conv(out)
        return img


class ConvolutionalDiscriminator(nn.Module):
    def __init__(self, config):
        super(ConvolutionalDiscriminator, self).__init__()
        self.input_size = config['input_size']
        self.output_size = config['output_size']
        self.convlayer_dims = [self.input_size, 128, 64]

        self.conv = nn.Sequential()
        for i in range(len(self.convlayer_dims) - 1):
            self.conv.add_module('conv2d' + str(i), nn.Conv2d(
                    self.convlayer_dims[i], self.convlayer_dims[i+1], 4, stride=2))
            self.conv.add_module('lrelu' + str(i), nn.LeakyReLU(0.1, inplace=True))
            if i > 0:
                self.conv.add_module('bn', nn.BatchNorm2d(self.convlayer_dims[i+1]))

        # The height and width of downsampled image
        ds_size = 5

        # Output layers
        self.d_out = nn.Sequential(
                nn.Linear(self.convlayer_dims[-1] * ds_size ** 2, 1),
                nn.Sigmoid())
#        self.categorical_layer = nn.Sequential(nn.Linear(convlayer_dims[-1] * ds_size ** 2, config['n_categorical_codes']), 
#                                               nn.Softmax(dim=-1))
#        self.continuous_layer = nn.Sequential(nn.Linear(convlayer_dims[-1] * ds_size ** 2, config['n_continuous_codes']))

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.shape[0], -1)
        validity = self.d_out(out)
#        categorical_code = self.categorical_layer(out)
#        continous_code = self.continuous_layer(out)
        return validity, out # categorical_code, continous_code
    
 