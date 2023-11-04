import os
import pdb
import torch
import numpy as np
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint
import torch.nn.functional as F
from model.PLS3D import DS_conv,DeDS_conv,DRDB,DRDB_sequential
from torch.utils.checkpoint import checkpoint

MAX_NUM_STEPS = 1000 # Maximum number of steps for ODE solver

class ODEBlock(nn.Module):
    def __init__(self, odefunc, tol=1e-3, adjoint=False):
        """
        Code adapted from https://github.com/EmilienDupont/augmented-neural-odes

        Utility class that wraps odeint and odeint_adjoint.

        Args:
            odefunc (nn.Module): the module to be evaluated
            tol (float): tolerance for the ODE solver
            adjoint (bool): whether to use the adjoint method for gradient calculation
        """
        super(ODEBlock, self).__init__()
        self.adjoint = adjoint
        self.odefunc = odefunc
        self.tol = tol

    def forward(self, x, eval_times=None):
        # Forward pass corresponds to solving ODE, so reset number of function
        # evaluations counter
        self.odefunc.nfe = 0

        if eval_times is None:
            integration_time = torch.tensor([0, 1]).float()
        else:
            integration_time = eval_times.type_as(x)
            
        if self.adjoint:
            out = odeint_adjoint(self.odefunc, x, integration_time,
                                 rtol=self.tol, atol=self.tol, method='dopri5',
                                 options={'max_num_steps': MAX_NUM_STEPS})
        else:
            out = odeint(self.odefunc, x, integration_time,
                         rtol=self.tol, atol=self.tol, method='dopri5',
                         options={'max_num_steps': MAX_NUM_STEPS})

        if eval_times is None:
            return out[1]  # Return only final time
        else:
            return out[1]

    def trajectory(self, x, timesteps):
        integration_time = torch.linspace(0., 1., timesteps)
        return self.forward(x, eval_times=integration_time)
    
    
def get_nonlinearity(name):
    """Helper function to get non linearity module, choose from relu/softplus/swish/lrelu"""
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'softplus':
        return nn.Softplus()
    # elif name == 'swish':
    #     return Swish(inplace=True)
    elif name == 'lrelu':
        return nn.LeakyReLU()

class nmODE(nn.Module):
    def __init__(self):
        """
        """
        super(nmODE, self).__init__()
        self.nfe = 0  # Number of function evaluations
        self.gamma = None
    
    def fresh(self, gamma):
        self.gamma = gamma
    
    def forward(self, t, p):
        self.nfe += 1
        dpdt = -p + torch.pow(torch.sin(p + self.gamma), 2)
        return dpdt
    
class nmODEBlock(nn.Module):
    def __init__(self, non_linearity='relu', output_dim=64, tol=1e-3, adjoint=False,
                 eval_times=(0, 1)):
        super(nmODEBlock, self).__init__()
        self.eval_times = torch.tensor(eval_times).float()

        self.nmODE = nmODE()
        self.ode_block = ODEBlock(self.nmODE, tol=tol, adjoint=adjoint)
        self.non_linearity = get_nonlinearity(non_linearity)

    def forward(self, x):
        self.nmODE.fresh(x)
        x = self.ode_block(torch.zeros_like(x), self.eval_times)
        x = self.non_linearity(x)
        return x 

class NMODEPls_Net_3D_cp(torch.nn.Module):

    def __init__(self, n_channels, n_classes,g=8,pretrain=False):
        super(NMODEPls_Net_3D_cp, self).__init__()

        self.ds_conv1 = DS_conv(n_channels, 16, stride=2, padding=1, dilation=1)
        self.drdb1 = DRDB(17,g)
        self.ds_conv_cat_1 = DS_conv(17, 2 * n_classes)

        self.ds_conv2 = DS_conv(17, 64, stride=2, padding=1, dilation=1)
        #self.drdb2 = DRDB(65)
        self.drdb2 = nn.Sequential(
            DRDB(65,g),
            DRDB(65,g)
        )
        self.ds_conv_cat_2 = DS_conv(65, 2 * n_classes)

        self.ds_conv3 = DS_conv(65, 128, stride=2, padding=1, dilation=1)
        #self.drdb3 = DRDB(129)
        self.drdb3 = nn.Sequential(
            DRDB(129,g),
            DRDB(129,g),
            DRDB(129,g),
            DRDB(129,g)
        )

        self.D_ds_conv3 = DS_conv(129, 2 * n_classes, stride=1, padding=1, dilation=1)
        self.D_ds_conv2 = DS_conv(4 * n_classes, 2 * n_classes, stride=1, padding=1, dilation=1)
        self.D_ds_conv1 = DS_conv(4 * n_classes, 2 * n_classes, stride=1, padding=1, dilation=1)

        #self.down = nn.functional.interpolate()
        # 预训练为True时使输出channel为1
        self.odeblock = nmODEBlock()
        if pretrain:
            self.conv = nn.Conv3d(in_channels=2 * n_classes, out_channels=1, kernel_size=1, stride=1, padding=0)
        else:
            self.conv = nn.Conv3d(in_channels=2 * n_classes, out_channels=n_classes, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
    
    def encoder_1(self,x):
        x1 = self.ds_conv1(x)
        ir_1 = nn.functional.interpolate(x,size = [x1.size(2),x1.size(3),x1.size(4)])
        x1 = torch.cat([x1, ir_1], 1)
        x1 = self.drdb1(x1)
        cat_1 = self.ds_conv_cat_1(x1)
        return x1,cat_1,ir_1
    
    def encoder_2(self,x1,ir_1):
        x1 = self.ds_conv2(x1)
        ir_2 = nn.functional.interpolate(ir_1,[x1.size(2),x1.size(3),x1.size(4)])
        x1 = torch.cat([x1, ir_2], 1)
        x1 = self.drdb2(x1)
        cat_2 = self.ds_conv_cat_2(x1)
        return x1,cat_2,ir_2
    
    def encoder_3(self,x1,ir_2):
        x1 = self.ds_conv3(x1)
        ir_3 = nn.functional.interpolate(ir_2,[x1.size(2),x1.size(3),x1.size(4)])
        x1 = torch.cat([x1, ir_3], 1)
        x1 = self.drdb3(x1)
        return x1
    
    def decoder_3(self,x1,cat_2):
        
        x1 = self.D_ds_conv3(x1)
        x1 = nn.functional.interpolate(input=x1, size = [cat_2.size(2),cat_2.size(3),cat_2.size(4)], mode='trilinear',
                                    align_corners=False)
        x1 = torch.cat([x1, cat_2], 1)
        return x1
        
    def decoder_2(self,x1,cat_1):
        x1 = self.D_ds_conv2(x1)
        x1 = nn.functional.interpolate(input=x1, size = [cat_1.size(2),cat_1.size(3),cat_1.size(4)], mode='trilinear',
                                    align_corners=False)
        x1 = torch.cat([x1, cat_1], 1)
        return x1
    
    def decoder_1(self,x1,x):
        
        x1 = self.D_ds_conv1(x1)
        x1 = nn.functional.interpolate(input=x1, size = [x.size(2),x.size(3),x.size(4)], mode='trilinear',
                                    align_corners=False)
        
        return x1
    
    def forward(self,x):
        
        #size_x = [x.size(2),x.size(3),x.size(4)]
        x = x + torch.zeros(1, dtype=x.dtype, device=x.device, requires_grad=True)
        
        x1,cat_1,ir_1 = checkpoint(self.encoder_1,x)
        x1,cat_2,ir_2 = checkpoint(self.encoder_2,x1,ir_1)
        del ir_1
        x1 = checkpoint(self.encoder_3,x1,ir_2)
        del ir_2
        
        x1 = checkpoint(self.odeblock,x1)
        x1 = checkpoint(self.decoder_3,x1,cat_2)
        del cat_2
        x1 = checkpoint(self.decoder_2,x1,cat_1)
        del cat_1
        x1 = checkpoint(self.decoder_1,x1,x)
        del x
        
        # out
        x1 = self.conv(x1)
        #x1 = self.softmax(x1)
        #x1 = self.sigmoid(x1)
        
        return x1

class NMODEPls_Net_3D_cp_multitask(torch.nn.Module):

    def __init__(self, n_channels, n_classes,g=8, is_checkpoint = True):
        super(NMODEPls_Net_3D_cp_multitask, self).__init__()
        
        self.is_checkpoint = is_checkpoint
        self.ds_conv1 = DS_conv(n_channels, 16, stride=2, padding=1, dilation=1)
        self.drdb1 = DRDB(17,g)
        self.ds_conv_cat_1 = DS_conv(17, 2 * n_classes)

        self.ds_conv2 = DS_conv(17, 64, stride=2, padding=1, dilation=1)
        #self.drdb2 = DRDB(65)
        self.drdb2 = nn.Sequential(
            DRDB(65,g),
            DRDB(65,g)

        )
        self.ds_conv_cat_2 = DS_conv(65, 2 * n_classes)

        self.ds_conv3 = DS_conv(65, 128, stride=2, padding=1, dilation=1)
        #self.drdb3 = DRDB(129)
        self.drdb3 = nn.Sequential(
            DRDB(129,g),
            DRDB(129,g),
            DRDB(129,g),
            DRDB(129,g)
        )

        self.D_ds_conv3 = DS_conv(129, 2 * n_classes, stride=1, padding=1, dilation=1)
        self.D_ds_conv2 = DS_conv(4 * n_classes, 2 * n_classes, stride=1, padding=1, dilation=1)
        self.D_ds_conv1 = DS_conv(4 * n_classes, 2 * n_classes, stride=1, padding=1, dilation=1)

        #self.down = nn.functional.interpolate()
        self.conv = nn.Conv3d(in_channels=2 * n_classes, out_channels=n_classes, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv3d(in_channels=2 * n_classes, out_channels= 1, kernel_size=1, stride=1, padding=0)
        self.odeblock = nmODEBlock()
    def encoder_1(self,x):
        
        x1 = self.ds_conv1(x)
        
        ir_1 = nn.functional.interpolate(x,size = [x1.size(2),x1.size(3),x1.size(4)])
      
        
        x1 = torch.cat([x1, ir_1], 1)
        x1 = self.drdb1(x1)
        cat_1 = self.ds_conv_cat_1(x1)
        
        return x1,cat_1,ir_1
    
    def encoder_2(self,x1,ir_1):
        
        x1 = self.ds_conv2(x1)
        
        
        ir_2 = nn.functional.interpolate(ir_1,[x1.size(2),x1.size(3),x1.size(4)])
        
        
        x1 = torch.cat([x1, ir_2], 1)
        x1 = self.drdb2(x1)

        cat_2 = self.ds_conv_cat_2(x1)
        
        return x1,cat_2,ir_2
    
    def encoder_3(self,x1,ir_2):
        x1 = self.ds_conv3(x1)
        
        
        ir_3 = nn.functional.interpolate(ir_2,[x1.size(2),x1.size(3),x1.size(4)])
        
        
        x1 = torch.cat([x1, ir_3], 1)
        x1 = self.drdb3(x1)
        
        return x1
    
    def decoder_3(self,x1,cat_2):
        
        x1 = self.D_ds_conv3(x1)
        x1 = nn.functional.interpolate(input=x1, size = [cat_2.size(2),cat_2.size(3),cat_2.size(4)], mode='trilinear',
                                    align_corners=False)
        x1 = torch.cat([x1, cat_2], 1)
        return x1
        
    def decoder_2(self,x1,cat_1):
        
        
        x1 = self.D_ds_conv2(x1)
        x1 = nn.functional.interpolate(input=x1, size = [cat_1.size(2),cat_1.size(3),cat_1.size(4)], mode='trilinear',
                                    align_corners=False)
        x1 = torch.cat([x1, cat_1], 1)
        
        return x1
    
    def decoder_1(self,x1,x):
        
        x1 = self.D_ds_conv1(x1)
        x1 = nn.functional.interpolate(input=x1, size = [x.size(2),x.size(3),x.size(4)], mode='trilinear',
                                    align_corners=False)
        
        return x1
    
    def out_conv(self,x1):
        
        
        return self.conv(x1),torch.sigmoid(self.conv2(x1))
    
    def forward(self,x):
        
        #size_x = [x.size(2),x.size(3),x.size(4)]
        x = x + torch.zeros(1, dtype=x.dtype, device=x.device, requires_grad=True)
        
        if self.is_checkpoint:
            x1,cat_1,ir_1 = checkpoint(self.encoder_1,x)
        else:
            x1,cat_1,ir_1 = self.encoder_1(x)
        
        if self.is_checkpoint:
            x1,cat_2,ir_2 = checkpoint(self.encoder_2,x1,ir_1)
        else:
            x1,cat_2,ir_2 = self.encoder_2(x1,ir_1)
            
        del ir_1
        
        if self.is_checkpoint:
            x1 = checkpoint(self.encoder_3,x1,ir_2)
        else:
            x1 = self.encoder_3(x1,ir_2)
        del ir_2
        x1 = checkpoint(self.odeblock,x1)
        if self.is_checkpoint:
            x1 = checkpoint(self.decoder_3,x1,cat_2)
        else:
            x1 = self.decoder_3(x1,cat_2)
            
        del cat_2
        
        if self.is_checkpoint:
            x1 = checkpoint(self.decoder_2,x1,cat_1)
        else:
            x1 = self.decoder_2(x1,cat_1)
            
        del cat_1
        
        if self.is_checkpoint:
            x1 = checkpoint(self.decoder_1,x1,x)
        else:
            x1 = self.decoder_1(x1,x)
        del x
        
        # out
        if self.is_checkpoint: 
            out1,out2 = checkpoint(self.out_conv,x1)
        else:
            out1,out2 = self.out_conv(x1)
        #out1 = self.conv(x1)
        #out2 = self.conv2(x1)
        #x1 = self.softmax(x1)
        #x1 = self.sigmoid(x1)
        del x1
        
        return out1,out2
    