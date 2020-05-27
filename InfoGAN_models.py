#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 09:17:10 2019

@author: petrapoklukar

"""

import torch.nn as nn
import torch
import math
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
        
# ---------------------------------------- #
# --- Linear Generator & Distriminator --- #
# ---------------------------------------- #
class FullyConnectedGNet(nn.Module):
    def __init__(self, config):
        super(FullyConnectedGNet, self).__init__()
        self.latent_dim = config['latent_dim']
        self.linear_dims = config['linear_dims'] # [256, 512, 1024]
        self.output_reshape_dims = config['output_reshape_dims']        
        self.output_dim = config['output_dim']   
        self.dropout = config['dropout'] 
        self.bias = config['bias'] 

        self.lin = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(self.latent_dim, self.linear_dims[0], bias=self.bias)),
                ('bn1', nn.BatchNorm1d(self.linear_dims[0])),
                ('r1', nn.ReLU()),
                ('d1', nn.Dropout(p=self.dropout)),
                    
                ('fc2', nn.Linear(self.linear_dims[0], self.linear_dims[1], bias=self.bias)),
                ('bn2', nn.BatchNorm1d(self.linear_dims[1])),
                ('r2', nn.ReLU()),
                ('d2', nn.Dropout(p=self.dropout)),
                    
                ('fc3', nn.Linear(self.linear_dims[1], self.linear_dims[2], bias=self.bias)),
                ('bn3', nn.BatchNorm1d(self.linear_dims[2])),
                ('r3', nn.ReLU()),
                ('d3', nn.Dropout(p=self.dropout)),
                
                ('fc4', nn.Linear(self.linear_dims[2], self.output_dim, bias=self.bias))
            ]))
        
        if config['out_activation'] == 'tanh': 
            print(' *- Gnet: out_activation set to tanh')
            self.lin.add_module('tanh', nn.Tanh())
        # else:
        #     self.activation = lambda x: x
        #     print(' *- Gnet: out_activation set to None')
        
    def forward(self, *args):
        gen_input = torch.cat((*args), -1).view(-1, self.latent_dim)
        out_lin = self.lin(gen_input)
        # out_lin = self.activation(out_lin)
        out = out_lin.reshape(self.output_reshape_dims)
        return out
    
    
class FullyConnectedSNet(nn.Module):
    def __init__(self, config):
        super(FullyConnectedSNet, self).__init__()
        self.linear_dims = config['linear_dims'] # [512, 256]       
        self.output_dim = config['output_dim']
        self.dropout = config['dropout'] 
        self.bias = config['bias'] 

        self.lin = nn.Sequential(
                nn.Linear(self.output_dim, self.linear_dims[0], bias=self.bias),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
                
                nn.Linear(self.linear_dims[0], self.linear_dims[1], bias=self.bias),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
                )

    def forward(self, x):
        x = x.view(x.size(0), self.output_dim)
        return self.lin(x)


class FullyConnectedDNet(nn.Module):
    def __init__(self, config):
        super(FullyConnectedDNet, self).__init__()
        self.linear_dims = config['linear_dims'] # [512, 256]
        self.bias = config['bias'] 

        self.out = nn.Sequential(
                nn.Linear(self.linear_dims[1], 1, bias=self.bias),
                nn.Sigmoid()
                )

    def forward(self, x):
        return self.out(x).view(-1)
    
    
class FullyConnectedQNet(nn.Module):
    def __init__(self, model_config, data_config):
        super(FullyConnectedQNet, self).__init__()
        self.layer_dims = model_config['layer_dims']
        self.last_layer_dim = model_config['last_layer_dim']
        self.n_categorical_codes = data_config['structured_cat_dim']
        self.n_continuous_codes = data_config['structured_con_dim']
        self.forward_pass = self.continous_forward
        self.bias = model_config['bias']
        self.dropout = model_config['dropout']
        
        # In case of a bigger network
        if self.layer_dims != None:
            self.lin = nn.Sequential()
            for i in range(len(self.layer_dims) - 1):    
                self.lin.add_module(
                        'lin' + str(i), 
                        nn.Linear(self.layer_dims[i], self.layer_dims[i+1], 
                        bias=self.bias))
                self.lin.add_module('dropout' + str(i), nn.Dropout(p=self.dropout))
            self.last_layer_dim = self.layer_dims[-1]
            self.forward_pass = self.full_forward_extended if self.n_categorical_codes else self.continous_forward_extended
        
        if self.n_categorical_codes is not None:
            # Structured categorical code
            self.forward_pass = self.full_forward if self.layer_dims == None else self.full_forward_extended
            print('forward_pass set to full.')
            # Note: no Softmax activation because it is included in the loss function
            self.cat_layer = nn.Sequential(
                    nn.Linear(self.last_layer_dim, self.n_categorical_codes,
                              bias=self.bias))
            
        # Model structured continuous code as Gaussian
        self.con_layer_mean = nn.Linear(self.last_layer_dim, self.n_continuous_codes,
                                        bias=self.bias)
        self.con_layer_logvar = nn.Linear(self.last_layer_dim, self.n_continuous_codes,
                                          bias=self.bias)
        
            
    
    def continous_forward(self, x):
        """Forward pass without structured categorical code"""
        con_code_mean = self.con_layer_mean(x) # Structured continuous code
        con_code_logvar = self.con_layer_logvar(x) # Structured continuous code
        return con_code_mean, con_code_logvar
    
    def continous_forward_extended(self, x):
        """Forward pass without structured categorical code"""
        x_out = self.lin(x)
        con_code_mean = self.con_layer_mean(x_out) # Structured continuous code
        con_code_logvar = self.con_layer_logvar(x_out) # Structured continuous code
        return con_code_mean, con_code_logvar
    
    def full_forward(self, x):
        """Forward pass with structured categorical and continuous codes"""
        cat_code = self.cat_layer(x) # Structured categorical code
        con_code_mean = self.con_layer_mean(x) # Structured continuous code
        con_code_logvar = self.con_layer_logvar(x) # Structured continuous code
        return cat_code, con_code_mean, con_code_logvar
    
    def full_forward_extended(self, x):
        """Forward pass with structured categorical and continuous codes"""
        x_out = self.lin(x)
        cat_code = self.cat_layer(x_out) # Structured categorical code
        con_code_mean = self.con_layer_mean(x_out) # Structured continuous code
        con_code_logvar = self.con_layer_logvar(x_out) # Structured continuous code
        return cat_code, con_code_mean, con_code_logvar

    def forward(self, x):
        return self.forward_pass(x)

    
class FullyConnecteDecoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(FullyConnecteDecoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, output_size)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)

    def forward(self, x):
        print(x.shape, type(x), x.dtype)
        print(self.fc1(x).shape)
        x = F.relu(self.bn1(self.fc1(x)))
        print(x.shape, type(x), x.dtype)
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x



# ------------------- ARCHIVED ------------------- #
# From the first InfoGAN implementation   
class FullyConnectedGenerator_archived(nn.Module):
    def __init__(self, gen_config, data_config):
        super(FullyConnectedGenerator_archived, self).__init__()
        self.gen_config = gen_config
        self.layer_dims = gen_config['layer_dims']
        self.n_categorical_codes = data_config['structured_cat_dim']
        self.layer_dims.append(data_config['input_size'])
        self.layer_dims.insert(0, data_config['total_noise'])
        
        self.generator = nn.Sequential()
        for i in range(1, len(self.layer_dims) - 1):
            self.generator.add_module('lin' + str(i), nn.Linear(
                    self.layer_dims[i-1], self.layer_dims[i]))
            self.generator.add_module('bn' + str(i), nn.BatchNorm1d(
                    self.layer_dims[i]))
            self.generator.add_module('relu' + str(i), nn.LeakyReLU(0.1))
        self.generator.add_module('lin_last', nn.Linear(
                self.layer_dims[-2], self.layer_dims[-1]))
#        self.generator.add_module('tanh_last', nn.Tanh())
    
    def forward(self, *args):
        gen_input = torch.cat((*args), -1)
        out = self.generator(gen_input)
        return out


class FullyConnectedDiscriminator_archived(nn.Module):
    def __init__(self, config, data_config):
        super(FullyConnectedDiscriminator_archived, self).__init__()
        self.dis_config = config
        self.layer_dims = config['layer_dims']
        self.layer_dims.insert(0, data_config['input_size'])
        
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
    
    
    
    
#---------------------------------- #
# --------------------------------- #
# --- ARCHIVED BELOW THIS POINT --- #
#---------------------------------- #
#---------------------------------- #
        
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
    
    
class LinToConv(nn.Module):
    def __init__(self, input_dim, n_channels):
        super(LinToConv, self).__init__()
        self.n_channels = n_channels
        self.width = int(np.sqrt(input_dim / n_channels))

    def forward(self, feat):
        feat = feat.view((feat.shape[0], self.n_channels, self.width, self.width))
        return feat


class ConvToLin(nn.Module):
    def __init__(self): 
        super(ConvToLin, self).__init__()

    def forward(self, feat):
        batch, channels, width, height = feat.shape
        feat = feat.view((batch, channels * width * height)) 
        return feat

# ------------ #
# --- QNet --- #
# ------------ #
class QNet(nn.Module):
    def __init__(self, model_config, data_config):
        super(QNet, self).__init__()
        self.last_layer_dim = model_config['last_layer_dim']
        self.n_categorical_codes = data_config['structured_cat_dim']
        self.n_continuous_codes = data_config['structured_con_dim']
        self.forward_pass = self.continous_forward
        
        # Model structured continuous code as Gaussian
        self.con_layer_mean = nn.Linear(self.last_layer_dim, self.n_continuous_codes,
                                        bias=False)
        self.con_layer_logvar = nn.Linear(self.last_layer_dim, self.n_continuous_codes,
                                          bias=False)
        
        if self.n_categorical_codes > 0:
            # Structured categorical code
            self.forward_pass = self.full_forward
            print('forward_pass set to full.')
            self.cat_layer = nn.Sequential(
                    nn.Linear(self.last_layer_dim, self.n_categorical_codes,
                              bias=False), 
                    nn.Softmax(dim=-1))
    
    def continous_forward(self, x):
        """Forward pass without structured categorical code"""
        con_code_mean = self.con_layer_mean(x) # Structured continuous code
        con_code_logvar = self.con_layer_logvar(x) # Structured continuous code
        return con_code_mean, con_code_logvar
    
    def full_forward(self, x):
        """Forward pass with structured categorical and continuous codes"""
        cat_code = self.cat_layer(x) # Structured categorical code
        con_code_mean = self.con_layer_mean(x) # Structured continuous code
        con_code_logvar = self.con_layer_logvar(x) # Structured continuous code
        return cat_code, con_code_mean, con_code_logvar

    def forward(self, x):
        return self.forward_pass(x)


class DNet(nn.Module):
    def __init__(self, dis_config):
        super(DNet, self).__init__()
        self.last_layer_dim = dis_config['last_layer_dim']
        
        # Output layers for discriminator
        self.d_out = nn.Sequential(
                nn.Linear(self.last_layer_dim, 1, bias=False),
                nn.Sigmoid())

    def forward(self, x):
        return self.d_out(x)
    
# ----------------------------------------------- #
# --- Convolutional Generator & Distriminator --- #
# ----------------------------------------------- #
class ConvolutionalGenerator(nn.Module):
    def __init__(self, gen_config, data_config):
        super(ConvolutionalGenerator, self).__init__()
        self.gen_config = gen_config
        self.n_categorical_codes = data_config['structured_cat_dim']

        self.layer_dims = gen_config['layer_dims']
        self.layer_dims.insert(0, data_config['total_noise'])
        
        self.channel_dims = gen_config['channel_dims']
        self.init_size = gen_config['init_size'] 
        assert(int(math.sqrt(self.layer_dims[-1]/self.channel_dims[0])) == self.init_size)

        self.lin = nn.Sequential()
        for i in range(len(self.layer_dims) - 1):
            self.lin.add_module('lin' + str(i), nn.Linear(
                    self.layer_dims[i], self.layer_dims[i+1], bias=False))
            self.lin.add_module('lin_relu' + str(i), nn.ReLU(inplace=True))
            self.lin.add_module('lin_bn' + str(i), nn.BatchNorm1d(self.layer_dims[i+1]))
        self.lin.add_module('lin_to_conv', LinToConv(self.layer_dims[-1], self.channel_dims[0]))
    
        # Another version would be replacing ConvTranspose2d with Conv2d with 
        # the same parameters and have an Upsample(factor=4) in front of each
        self.conv = nn.Sequential()
        for i in range(1, len(self.channel_dims) - 1):
            self.conv.add_module('conv2d' + str(i), nn.ConvTranspose2d(
                    self.channel_dims[i-1], self.channel_dims[i], 4, stride=2, 
                    bias=False))
            self.conv.add_module('conv_relu' + str(i), nn.ReLU(inplace=True))
            self.conv.add_module('conv_bn' + str(i), nn.BatchNorm2d(self.channel_dims[i]))
        self.conv.add_module('conv2d_last' + str(i), nn.ConvTranspose2d(
                    self.channel_dims[-2], self.channel_dims[-1], 4, stride=2, padding=3,
                    bias=False))
        self.conv.add_module('conv2d_tanh', nn.Tanh())
#        self.conv.add_module('conv2d_sigmoid', nn.Sigmoid())

    def forward(self, *args):
        gen_input = torch.cat((*args), -1)
        out = self.lin(gen_input)
        img = self.conv(out)
        return img


class ConvolutionalDiscriminator(nn.Module):
    def __init__(self, dis_config, data_config):
        super(ConvolutionalDiscriminator, self).__init__()
        self.dis_config = dis_config

        self.channel_dims = dis_config['channel_dims']
        self.layer_dims = dis_config['layer_dims']
        self.last_layer_dim = dis_config['last_layer_dim']
        self.n_categorical_codes = data_config['structured_cat_dim']
        self.n_continuous_codes = data_config['structured_con_dim']
        self.forward_pass = self.continous_forward

        self.conv = nn.Sequential()
        for i in range(len(self.channel_dims) - 1):
            self.conv.add_module('conv2d' + str(i), nn.Conv2d(
                    self.channel_dims[i], self.channel_dims[i+1], 4, stride=2,
                    bias=False))
            self.conv.add_module('lrelu' + str(i), nn.LeakyReLU(0.2, inplace=True))
            if i > 0:
                self.conv.add_module('bn' + str(i), nn.BatchNorm2d(self.channel_dims[i+1]))
        
        self.lin = nn.Sequential()
        self.lin.add_module('conv_to_lin', ConvToLin())
        for i in range(len(self.layer_dims) - 1):
            self.lin.add_module('lin' + str(i), nn.Linear(
                    self.layer_dims[i], self.layer_dims[i+1], bias=False))
            self.lin.add_module('lrelu' + str(i), nn.LeakyReLU(0.2, inplace=True))
            if i != len(self.layer_dims) - 2:
                self.lin.add_module('bn' + str(i), nn.BatchNorm1d(self.layer_dims[i+1]))
        
        # Output layers for discriminator
        self.d_out = nn.Sequential(
                nn.Linear(self.layer_dims[-1], 1, bias=False),
                nn.Sigmoid())
        
        # Output Gaussian layer for structured continuous codes
        self.con_layer_mean = nn.Linear(self.last_layer_dim, self.n_continuous_codes,
                                        bias=False)
        self.con_layer_logvar = nn.Linear(self.last_layer_dim, self.n_continuous_codes,
                                          bias=False)
        
        # Output Sofmax layer for structured categorical codes
        if self.n_categorical_codes > 0:
            # Structured categorical code
            self.forward_pass = self.full_forward
            print('forward_pass set to full.')
            self.cat_layer = nn.Sequential(
                    nn.Linear(self.last_layer_dim, self.n_categorical_codes,
                              bias=False), 
                    nn.Softmax(dim=-1))
    
    def continous_forward(self, validity, x):
        """Forward pass without structured categorical code"""
        con_code_mean = self.con_layer_mean(x) # Structured continuous code
        con_code_logvar = self.con_layer_logvar(x) # Structured continuous code
        return validity, con_code_mean, con_code_logvar
    
    def full_forward(self, validity, x):
        """Forward pass with structured categorical and continuous codes"""
        cat_code = self.cat_layer(x) # Structured categorical code
        con_code_mean = self.con_layer_mean(x) # Structured continuous code
        con_code_logvar = self.con_layer_logvar(x) # Structured continuous code
        return validity, cat_code, con_code_mean, con_code_logvar

    def forward(self, x):
        out_conv = self.conv(x) 
        out_lin = self.lin(out_conv) # Last shared layer
        
        # Add Q output to the usual discriminator output
        validity = self.d_out(out_lin) 
        return self.forward_pass(validity, out_lin) 
    
    
class ConvolutionalDiscriminator_withoutQNet(nn.Module):
    def __init__(self, dis_config, data_config):
        super(ConvolutionalDiscriminator_withoutQNet, self).__init__()
        self.dis_config = dis_config

        self.channel_dims = dis_config['channel_dims']
        self.layer_dims = dis_config['layer_dims']

        self.conv = nn.Sequential()
        for i in range(len(self.channel_dims) - 1):
            self.conv.add_module('conv2d' + str(i), nn.Conv2d(
                    self.channel_dims[i], self.channel_dims[i+1], 4, stride=2,
                    bias=False))
            self.conv.add_module('lrelu' + str(i), nn.LeakyReLU(0.2, inplace=True))
            if i > 0:
                self.conv.add_module('bn' + str(i), nn.BatchNorm2d(self.channel_dims[i+1]))
        
        self.lin = nn.Sequential()
        self.lin.add_module('conv_to_lin', ConvToLin())
        for i in range(len(self.layer_dims) - 1):
            self.lin.add_module('lin' + str(i), nn.Linear(
                    self.layer_dims[i], self.layer_dims[i+1], bias=False))
            self.lin.add_module('lrelu' + str(i), nn.LeakyReLU(0.2, inplace=True))
            if i != len(self.layer_dims) - 2:
                self.lin.add_module('bn' + str(i), nn.BatchNorm1d(self.layer_dims[i+1]))
        
        # Output layers for discriminator
        self.d_out = nn.Sequential(
                nn.Linear(self.layer_dims[-1], 1, bias=False),
                nn.Sigmoid())
        
    def forward(self, x):
        out_conv = self.conv(x) 
        out_lin = self.lin(out_conv) # Needed for QNet
        validity = self.d_out(out_lin) # Usual discriminator output
        return validity, out_lin 
    

class ConvolutionalSharedDandQ(nn.Module):
    def __init__(self, model_config, data_config):
        super(ConvolutionalSharedDandQ, self).__init__()
        self.model_config = model_config

        self.channel_dims = model_config['channel_dims']
        self.layer_dims = model_config['layer_dims']

        self.conv = nn.Sequential()
        for i in range(len(self.channel_dims) - 1):
            self.conv.add_module('conv2d' + str(i), nn.Conv2d(
                    self.channel_dims[i], self.channel_dims[i+1], 4, stride=2,
                    bias=False))
            self.conv.add_module('lrelu' + str(i), nn.LeakyReLU(0.2, inplace=True))
            if i > 0:
                self.conv.add_module('bn' + str(i), nn.BatchNorm2d(self.channel_dims[i+1]))
        
        self.lin = nn.Sequential()
        self.lin.add_module('conv_to_lin', ConvToLin())
        for i in range(len(self.layer_dims) - 1):
            self.lin.add_module('lin' + str(i), nn.Linear(
                    self.layer_dims[i], self.layer_dims[i+1], bias=False))
            self.lin.add_module('lrelu' + str(i), nn.LeakyReLU(0.2, inplace=True))
            if i != len(self.layer_dims) - 2:
                self.lin.add_module('bn' + str(i), nn.BatchNorm1d(self.layer_dims[i+1]))
        
    def forward(self, x):
        out_conv = self.conv(x) 
        out_lin = self.lin(out_conv) # Needed for QNet
        return out_lin 


    
# --------------------------------- #
# --- Testing the architectures --- #
# --------------------------------- #
if __name__ == '__main__D':
    from torch.utils.data import Dataset
    import numpy as np

    class TrajDataset(Dataset):
        def __init__(self, data_filename, device=None):
            if not device:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = device
            self.data = torch.from_numpy(
                    np.load(data_filename)).float().to(self.device)
            self.num_samples = self.data.shape[0]
            
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            return self.data[idx]

    def full_noise(batch_size, config, batch_dis_classes=None):
        """
        Generates uninformed noise, structured discrete noise and 
        structured continuous noise.
        """
        z_dim = config['usual_noise_dim']
        dis_classes = config['structured_cat_dim']
        con_c_dim = config['structured_con_dim']
        device = 'cpu'
        
        # the usual uninformed noise
        z_noise = torch.empty((batch_size, z_dim), requires_grad=False, 
                              device=device).normal_() # b, x_dim
        
        # structured discrete code noise
        if batch_dis_classes is None:
            batch_dis_classes = np.random.randint(0, dis_classes, batch_size)
        dis_noise = np.zeros((batch_size, dis_classes)) 
        dis_noise[range(batch_size), batch_dis_classes] = 1.0 # bs, dis_classes
        dis_noise = torch.Tensor(dis_noise)
        
        # structured continuous code noise
        con_noise = torch.empty((batch_size, con_c_dim), requires_grad=False, 
                               device=device).uniform_(-1, 1)
        return z_noise, dis_noise, con_noise
    
    def con_noise(batch_size, config):
        """
        Generates uninformed noise, structured discrete noise and 
        structured continuous noise.
        """
        z_dim = config['usual_noise_dim']
        con_c_dim = config['structured_con_dim']
        device = 'cpu'
        
        # the usual uninformed noise
        z_noise = torch.empty((batch_size, z_dim), requires_grad=False, 
                              device=device).normal_() # b, x_dim

        # structured continuous code noise
        con_noise = torch.empty((batch_size, con_c_dim), requires_grad=False, 
                               device=device).uniform_(-1, 1)
        return z_noise, con_noise

    path_to_data = 'dataset/robot_trajectories/yumi_joint_pose.npy'
    data = TrajDataset(path_to_data)
    batch_size = 64
    test_batch = data[:batch_size].view(batch_size, -1)
    data_config = {
            'input_size': 553,
            'usual_noise_dim': 1,
            'structured_cat_dim': 0, 
            'structured_con_dim': 6,
            'total_noise': 7
            }
    noise_fn = full_noise if data_config['structured_cat_dim'] > 0 else con_noise        
    
    discriminator_config =  {
            'class_name': 'FullyConnectedDiscriminator',
            'layer_dims': [1000, 500, 250, 250, 50]
            }
    
    discriminator = FullyConnectedDiscriminator(discriminator_config, data_config)
    d_out = discriminator(test_batch) # tuple of shape ( (64, 1), (64, last_dim) )
    print(' *- d_out.shape ({0}, {1})'.format(d_out[0].shape, d_out[1].shape)) 
    
    generator_config = {
            'class_name': 'FullyConnectedGenerator',
            'layer_dims': [128, 256, 256, 512]
            }
    generator = FullyConnectedGNet(generator_config, data_config)
    noise = noise_fn(64, data_config)
    print(' *- noise.shape ', list(map(lambda x: x.shape, noise)))
    g_out = generator(noise)
    print(' *- g_out.shape ', g_out.shape)
    
    Qnet_config = {
            'class_name': 'QNet',
            'last_layer_dim': 50,
            }
    qnet = QNet(Qnet_config, data_config)
    q_out = qnet(d_out[1])
    print(' *- q_out.shape ', list(map(lambda x: x.shape, q_out)))


    # Test the Convolutional Models
    print('\n\nTesting Convolutional layers...\n')
    data_config = {
        'input_size': None,
        'usual_noise_dim': 62,
        'structured_cat_dim': 10, 
        'structured_con_dim': 2,
        'total_noise': 74
        }
    noise = full_noise(64, data_config)
    print(' *- noise.shape ', list(map(lambda x: x.shape, noise)))
    generator_config = {
        'class_name': 'ConvolutionalGenerator',
        'layer_dims': [1024, 7*7*128],
        'channel_dims': [128, 64, 1],
        'init_size': 7
    }
    generator = ConvolutionalGenerator(generator_config, data_config)
    g_out = generator(noise)
    print(' *- g_out.shape ', g_out.shape)
    
    discriminator_config =  {
            'class_name': 'ConvolutionalDiscriminator',
            'channel_dims': [1, 64, 128],
            'layer_dims': [128*5*5, 1024, 128]
            }
    
    discriminator = ConvolutionalDiscriminator(discriminator_config, data_config)
    d_out = discriminator(g_out)
    print(' *- d_out.shape ({0}, {1})'.format(d_out[0].shape, d_out[1].shape))  
    
    Qnet_config = {
            'class_name': 'QNet',
            'last_layer_dim': 128,
            }
    qnet = QNet(Qnet_config, data_config)
    q_out = qnet(d_out[1])
    print(' *- q_out.shape ', list(map(lambda x: x.shape, q_out)))
