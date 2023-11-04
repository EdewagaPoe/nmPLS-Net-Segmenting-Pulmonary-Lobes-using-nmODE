import torch
import torch.nn  as nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

class Pls_Net_3D(torch.nn.Module):

    def __init__(self, n_channels, n_classes):
        super(Pls_Net_3D, self).__init__()

        self.ds_conv1 = DS_conv(n_channels, 16, stride=2, padding=1, dilation=1)
        self.drdb1 = DRDB(17,12)
        self.ds_conv_cat_1 = DS_conv(17, 2 * n_classes)

        self.ds_conv2 = DS_conv(17, 64, stride=2, padding=1, dilation=1)
        #self.drdb2 = DRDB(65)
        self.drdb2 = nn.Sequential(
            DRDB(65,12),
            DRDB(65,12)

        )
        self.ds_conv_cat_2 = DS_conv(65, 2 * n_classes)

        self.ds_conv3 = DS_conv(65, 128, stride=2, padding=1, dilation=1)
        #self.drdb3 = DRDB(129)
        self.drdb3 = nn.Sequential(
            DRDB(129,12),
            DRDB(129,12),
            DRDB(129,12),
            DRDB(129,12)
        )

        self.D_ds_conv3 = DS_conv(129, 2 * n_classes, stride=1, padding=1, dilation=1)
        self.D_ds_conv2 = DS_conv(4 * n_classes, 2 * n_classes, stride=1, padding=1, dilation=1)
        self.D_ds_conv1 = DS_conv(4 * n_classes, 2 * n_classes, stride=1, padding=1, dilation=1)

        #self.down = nn.functional.interpolate()
        self.conv = nn.Conv3d(in_channels=2 * n_classes, out_channels=n_classes, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        size_x = [x.size(2),x.size(3),x.size(4)]
        # encoder L1
        x1 = self.ds_conv1(x)
        
        ir_1 = nn.functional.interpolate(x,size = [x1.size(2),x1.size(3),x1.size(4)])
      
        
        x1 = torch.cat([x1, ir_1], 1)
        x1 = self.drdb1(x1)
        cat_1 = self.ds_conv_cat_1(x1)

        # encoder L2
        x1 = self.ds_conv2(x1)
        
        
        ir_2 = nn.functional.interpolate(ir_1,[x1.size(2),x1.size(3),x1.size(4)])
        
        
        x1 = torch.cat([x1, ir_2], 1)
        x1 = self.drdb2(x1)

        cat_2 = self.ds_conv_cat_2(x1)

        # encoder L3
        x1 = self.ds_conv3(x1)
        
        
        ir_3 = nn.functional.interpolate(ir_2,[x1.size(2),x1.size(3),x1.size(4)])
        
        
        x1 = torch.cat([x1, ir_3], 1)
        x1 = self.drdb3(x1)
        
        del x
        del ir_1
        del ir_2
        del ir_3

        # decoder L3
        x1 = self.D_ds_conv3(x1)
        x1 = nn.functional.interpolate(input=x1, size = [cat_2.size(2),cat_2.size(3),cat_2.size(4)], mode='trilinear',
                                    align_corners=False)
        
        # decoder L2
        x1 = torch.cat([x1, cat_2], 1)
        x1 = self.D_ds_conv2(x1)
        x1 = nn.functional.interpolate(input=x1, size = [cat_1.size(2),cat_1.size(3),cat_1.size(4)], mode='trilinear',
                                    align_corners=False)
        
        del cat_2
        
        # decoder L1
        x1 = torch.cat([x1, cat_1], 1)
        
        del cat_1
        
        x1 = self.D_ds_conv1(x1)
        x1 = nn.functional.interpolate(input=x1, size = size_x, mode='trilinear',
                                    align_corners=False)
        
        
        
        
        
        # out
        x1 = self.conv(x1)
        x1 = self.softmax(x1)
        
        return x1
    
    
class Pls_Net_3D_cp(torch.nn.Module):

    def __init__(self, n_channels, n_classes,g=8,pretrain=False):
        super(Pls_Net_3D_cp, self).__init__()

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
class Pls_Net_3D_cp(torch.nn.Module):

    def __init__(self, n_channels, n_classes,g=8,pretrain=False):
        super(Pls_Net_3D_cp, self).__init__()

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
        

class Pls_Net_3D_cp_multitask(torch.nn.Module):

    def __init__(self, n_channels, n_classes,g=8, is_checkpoint = True):
        super(Pls_Net_3D_cp_multitask, self).__init__()
        
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
    

class PPls_Net_3D_cp_multitask(torch.nn.Module):

    def __init__(self, n_channels, n_classes,g=8, is_checkpoint = True):
        super(PPls_Net_3D_cp_multitask, self).__init__()
        
        self.is_checkpoint = is_checkpoint
        self.ds_conv1 = DS_conv(n_channels, 16, stride=2, padding=1, dilation=1)
        self.drdb1 = DRDB(17,g)
        self.ds_conv_cat_1 = DS_conv(17, 2 * n_classes)

        self.ds_conv2 = DS_conv(17, 64, stride=2, padding=1, dilation=1)
        
        self.drdb2 = nn.Sequential(
            DRDB(65,g),
            DRDB(65,g)

        )
        self.ds_conv_cat_2 = DS_conv(65, 2 * n_classes)

        self.ds_conv3 = DS_conv(65, 128, stride=2, padding=1, dilation=1)
        
        self.drdb3 = nn.Sequential(
            DRDB(129,g),
            DRDB(129,g),
            DRDB(129,g),
            DRDB(129,g)
        )

        self.D_ds_conv3 = DS_conv(129, 2 * n_classes, stride=1, padding=1, dilation=1)
        self.D_ds_conv2 = DS_conv(4 * n_classes, 2 * n_classes, stride=1, padding=1, dilation=1)
        self.D_ds_conv1 = DS_conv(4 * n_classes, 2 * n_classes, stride=1, padding=1, dilation=1)
        
        self.De_ds_conv3 = DeDS_conv(in_channels = 24, out_channels = 12,kernel_size = 4, stride=4)
        self.De_ds_conv2 = DeDS_conv(in_channels = 24, out_channels = 12,kernel_size = 2, stride=2)
        
        self.out3 = nn.Conv3d(in_channels=2 * n_classes, out_channels=n_classes, kernel_size=1, stride=1, padding=0)
        self.out2 = nn.Conv3d(in_channels=2 * n_classes, out_channels=n_classes, kernel_size=1, stride=1, padding=0)
        
        self.out1 = nn.Conv3d(in_channels=2 * n_classes, out_channels=n_classes, kernel_size=1, stride=1, padding=0)
        self.out1m = nn.Conv3d(in_channels=2 * n_classes, out_channels= 1, kernel_size=1, stride=1, padding=0)
    
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
        s1 = self.De_ds_conv3(x1)
        return x1,s1
        
    def decoder_2(self,x1,cat_1):
        
        
        x1 = self.D_ds_conv2(x1)
        x1 = nn.functional.interpolate(input=x1, size = [cat_1.size(2),cat_1.size(3),cat_1.size(4)], mode='trilinear',
                                    align_corners=False)
        x1 = torch.cat([x1, cat_1], 1)
        s1 = self.De_ds_conv2(x1)
        return x1,s1
    
    def decoder_1(self,x1,x):
        
        x1 = self.D_ds_conv1(x1)
        x1 = nn.functional.interpolate(input=x1, size = [x.size(2),x.size(3),x.size(4)], mode='trilinear',
                                    align_corners=False)
        
        return x1
    
    def out_conv(self,x1):
        
        
        return self.out1(x1),torch.sigmoid(self.out1m(x1))
    
    def forward(self,x):
        
        size_x = [x.size(2),x.size(3),x.size(4)]
        x = x + torch.zeros(1, dtype=x.dtype, device=x.device, requires_grad=True)
        #print(x.shape)
        
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
        
        if self.is_checkpoint:
            x1,s3 = checkpoint(self.decoder_3,x1,cat_2)
            s3 = nn.functional.interpolate(input=s3, size =size_x , mode='trilinear',
                                    align_corners=False)
        else:
            x1,s3 = self.decoder_3(x1,cat_2)
            s3 = nn.functional.interpolate(input=s3, size =size_x , mode='trilinear',
                                    align_corners=False)
        #print(s3.shape)
            
        del cat_2
        
        if self.is_checkpoint:
            x1,s2 = checkpoint(self.decoder_2,x1,cat_1)
            s2 = nn.functional.interpolate(input=s2, size =size_x , mode='trilinear',
                                    align_corners=False )                             
        else:
            x1,s2 = self.decoder_2(x1,cat_1)
            s2 = nn.functional.interpolate(input=s2, size =size_x , mode='trilinear',
                                    align_corners=False )
        #print(s2.shape)    
        del cat_1
        
        if self.is_checkpoint:
            x1 = checkpoint(self.decoder_1,x1,x)
        else:
            x1 = self.decoder_1(x1,x)
        del x
        
        
        
        # out
        if self.is_checkpoint: 
            
            
            out3 = self.out3(s3)
            out2 = self.out2(s2 + s3)
            out1,out1m = checkpoint(self.out_conv,x1 + s3 + s2)
        else:
            out3 = self.out3(s3)
            out2 = self.out2(s2 + s3)
            out1,out1m = self.out_conv(x1 + s3 + s2)
        
        del x1,s3,s2
        
        return out1,out1m,out3,out2

    
class PPls_Net_3D_cp_multitask_v2(torch.nn.Module):

    def __init__(self, n_channels, n_classes,g=12, is_checkpoint = True):
        super(PPls_Net_3D_cp_multitask_v2, self).__init__()
        
        self.is_checkpoint = is_checkpoint
        self.ds_conv1 = DS_conv(n_channels, 16, stride=2, padding=1, dilation=1)
        self.drdb1 = DRDB(17,g)
        self.ds_conv_cat_1 = DS_conv(17, 2 * n_classes)

        self.ds_conv2 = DS_conv(17, 64, stride=2, padding=1, dilation=1)
        
        self.drdb2 = nn.Sequential(
            DRDB(65,g),
            DRDB(65,g)

        )
        self.ds_conv_cat_2 = DS_conv(65, 2 * n_classes)

        self.ds_conv3 = DS_conv(65, 128, stride=2, padding=1, dilation=1)
        
        self.drdb3 = nn.Sequential(
            DRDB(129,g),
            DRDB(129,g),
            DRDB(129,g),
            DRDB(129,g)
        )

        self.D_ds_conv3 = DS_conv(129, 2 * n_classes, stride=1, padding=1, dilation=1)
        self.D_ds_conv2 = DS_conv(4 * n_classes, 2 * n_classes, stride=1, padding=1, dilation=1)
        self.D_ds_conv1 = DS_conv(4 * n_classes, 2 * n_classes, stride=1, padding=1, dilation=1)
        
        self.De_ds_conv3 = DeDS_conv(in_channels = 24, out_channels = 12,kernel_size = 4, stride=4)
        self.De_ds_conv2 = DeDS_conv(in_channels = 24, out_channels = 12,kernel_size = 2, stride=2)
        
        self.out3 = nn.Conv3d(in_channels=2 * n_classes, out_channels=n_classes, kernel_size=1, stride=1, padding=0)
        self.out2 = nn.Conv3d(in_channels=2 * n_classes, out_channels=n_classes, kernel_size=1, stride=1, padding=0)
        
        self.out1 = nn.Conv3d(in_channels=2 * n_classes, out_channels=n_classes, kernel_size=1, stride=1, padding=0)
        self.out1m = nn.Conv3d(in_channels=2 * n_classes, out_channels= 1, kernel_size=1, stride=1, padding=0)
    
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
        s1 = self.De_ds_conv3(x1)
        return x1,s1
        
    def decoder_2(self,x1,cat_1):
        
        
        x1 = self.D_ds_conv2(x1)
        x1 = nn.functional.interpolate(input=x1, size = [cat_1.size(2),cat_1.size(3),cat_1.size(4)], mode='trilinear',
                                    align_corners=False)
        x1 = torch.cat([x1, cat_1], 1)
        s1 = self.De_ds_conv2(x1)
        return x1,s1
    
    def decoder_1(self,x1,x):
        
        x1 = self.D_ds_conv1(x1)
        x1 = nn.functional.interpolate(input=x1, size = [x.size(2),x.size(3),x.size(4)], mode='trilinear',
                                    align_corners=False)
        
        return x1
    
    def out_conv(self,x1):
        
        
        return self.out1(x1),torch.sigmoid(self.out1m(x1))
    
    def forward(self,x):
        
        size_x = [x.size(2),x.size(3),x.size(4)]
        x = x + torch.zeros(1, dtype=x.dtype, device=x.device, requires_grad=True)
        #print(x.shape)
        
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
        
        if self.is_checkpoint:
            x1,s3 = checkpoint(self.decoder_3,x1,cat_2)
            s3 = nn.functional.interpolate(input=s3, size =size_x , mode='trilinear',
                                    align_corners=False)
        else:
            x1,s3 = self.decoder_3(x1,cat_2)
            s3 = nn.functional.interpolate(input=s3, size =size_x , mode='trilinear',
                                    align_corners=False)
        #print(s3.shape)
            
        del cat_2
        
        if self.is_checkpoint:
            x1,s2 = checkpoint(self.decoder_2,x1,cat_1)
            s2 = nn.functional.interpolate(input=s2, size =size_x , mode='trilinear',
                                    align_corners=False )                             
        else:
            x1,s2 = self.decoder_2(x1,cat_1)
            s2 = nn.functional.interpolate(input=s2, size =size_x , mode='trilinear',
                                    align_corners=False )
        #print(s2.shape)    
        del cat_1
        
        if self.is_checkpoint:
            x1 = checkpoint(self.decoder_1,x1,x)
        else:
            x1 = self.decoder_1(x1,x)
        del x
        
        
        
        # out
        if self.is_checkpoint: 
            
            if self.training:
            
                out3 = checkpoint(self.out3,s3)
                out2 = checkpoint(self.out2,s2)
                out1,out1m = checkpoint(self.out_conv,x1)
            else:
                out1,out1m = checkpoint(self.out_conv,x1)
                return out1,out1m
        else:
            if self.training:
                out3 = self.out3(s3)
                out2 = self.out2(s2)
                out1,out1m = self.out_conv(x1)
            else:
                out1,out1m = self.out_conv(x1)
                return out1,out1m
        
        del x1,s3,s2
        
        return out1,out1m,out3,out2    
    
class Pls_Net_3D_parallel_2(torch.nn.Module):

    def __init__(self, n_channels, n_classes):
        super(Pls_Net_3D_parallel_2, self).__init__()

        self.ds_conv1 = DS_conv(n_channels, 16, stride=2, padding=1, dilation=1,device='cuda:0')
        self.drdb1 = DRDB(17,12,device = 'cuda:0')
        self.ds_conv_cat_1 = DS_conv(17, 2 * n_classes,device='cuda:0')

        self.ds_conv2 = DS_conv(17, 64, stride=2, padding=1, dilation=1,device='cuda:0')
        #self.drdb2 = DRDB(65)
        self.drdb2 = nn.Sequential(
            DRDB(65,12,device = 'cuda:0'),
            DRDB(65,12,device = 'cuda:0')

        ).to('cuda:0')
        self.ds_conv_cat_2 = DS_conv(65, 2 * n_classes,device='cuda:0')

        self.ds_conv3 = DS_conv(65, 128, stride=2, padding=1, dilation=1,device='cuda:1')
        self.ds_conv3.cuda(1)
#         self.ds_conv3 = nn.Sequential(
#             nn.Conv3d(in_channels=65, out_channels=65, kernel_size=3, stride=2,
#                                padding=1, dilation=1,groups=65),
#             nn.Conv3d(in_channels=65, out_channels=128, kernel_size=1, stride=1,
#                                    padding=0),
#             nn.BatchNorm3d(128),
#             nn.ReLU(inplace=False)
            
#         ).to('cuda:1')
        
        self.drdb3 = nn.Sequential(
            DRDB_sequential(129,12,device = 'cuda:1').to('cuda:1'),
            DRDB_sequential(129,12,device = 'cuda:1').to('cuda:1'),
            DRDB_sequential(129,12,device = 'cuda:1').to('cuda:1'),
            DRDB_sequential(129,12,device = 'cuda:1').to('cuda:1')
        ).to('cuda:1')

        self.D_ds_conv3 = DS_conv(129, 2 * n_classes, stride=1, padding=1, dilation=1,device='cuda:1').to('cuda:1')
        self.D_ds_conv2 = DS_conv(4 * n_classes, 2 * n_classes, stride=1, padding=1, dilation=1,device='cuda:1').to('cuda:1')
        self.D_ds_conv1 = DS_conv(4 * n_classes, 2 * n_classes, stride=1, padding=1, dilation=1,device='cuda:1').to('cuda:1')

        #self.down = nn.functional.interpolate()
        self.conv = nn.Conv3d(in_channels=2 * n_classes, out_channels=n_classes, kernel_size=1, stride=1, padding=0).to('cuda:1')
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
    
    
    
    def forward(self,x):
        
        #size_x = [x.size(2),x.size(3),x.size(4)]
        x = x + torch.zeros(1, dtype=x.dtype, device=x.device, requires_grad=True)
        
        #encoder 1
        x1 = self.ds_conv1(x)
        
        ir_1 = nn.functional.interpolate(x,size = [x1.size(2),x1.size(3),x1.size(4)])
      
        
        x1 = torch.cat([x1, ir_1], 1)
        x1 = self.drdb1(x1)
        cat_1 = self.ds_conv_cat_1(x1)
        
        #encoder 2
        x1 = self.ds_conv2(x1)
        
        
        ir_2 = nn.functional.interpolate(ir_1,[x1.size(2),x1.size(3),x1.size(4)])
        
        
        x1 = torch.cat([x1, ir_2], 1)
        x1 = self.drdb2(x1)

        cat_2 = self.ds_conv_cat_2(x1)
        
        
        del ir_1
        #encoder 3
        x1 = self.ds_conv3(x1.cuda(1))
        
        
        ir_3 = nn.functional.interpolate(ir_2.to('cuda:1'),[x1.size(2),x1.size(3),x1.size(4)])
        
        
        x1 = torch.cat([x1, ir_3], 1)
        x1 = self.drdb3(x1)
        
        del ir_2
        
        #decoder 3
        
        x1 = self.D_ds_conv3(x1)
        x1 = nn.functional.interpolate(input=x1, size = [cat_2.size(2),cat_2.size(3),cat_2.size(4)], mode='trilinear',
                                    align_corners=False)
        x1 = torch.cat([x1, cat_2.to('cuda:1')], 1)
        del cat_2
        #decoder 2
        x1 = self.D_ds_conv2(x1)
        x1 = nn.functional.interpolate(input=x1, size = [cat_1.size(2),cat_1.size(3),cat_1.size(4)], mode='trilinear',
                                    align_corners=False)
        x1 = torch.cat([x1, cat_1.to('cuda:1')], 1)
        del cat_1
        
        #decoder 1
        x1 = self.D_ds_conv1(x1)
        x1 = nn.functional.interpolate(input=x1, size = [x.size(2),x.size(3),x.size(4)], mode='trilinear',
                                    align_corners=False)
        del x
        
        # out
        x1 = self.conv(x1.to('cuda:1'))
        #x1 = self.softmax(x1)
        #x1 = self.sigmoid(x1)
        
        return x1        
                
class Pls_Net_3D_parallel(torch.nn.Module):

    def __init__(self, n_channels, n_classes):
        super(Pls_Net_3D_parallel, self).__init__()

        self.ds_conv1 = DS_conv(n_channels, 16, stride=2, padding=1, dilation=1,device='cuda:0').cuda(0)
        self.drdb1 = DRDB(17,12).cuda(0)
        self.drdb1 = DRDB(17,12).cuda(0)
        self.ds_conv_cat_1 = DS_conv(17, 2 * n_classes,device='cuda:0').cuda(0)

        self.ds_conv2 = DS_conv(17, 64, stride=2, padding=1, dilation=1,device='cuda:0').cuda(0)
        #self.drdb2 = DRDB(65)
        self.drdb2 = nn.Sequential(
            DRDB(65,12),
            DRDB(65,12)

        ).to('cuda:0')
        self.ds_conv_cat_2 = DS_conv(65, 2 * n_classes,device='cuda:0').cuda(0)

        self.ds_conv3 = DS_conv(65, 128, stride=2, padding=1, dilation=1,device='cuda:1').cuda(1)
        
        self.drdb3 = nn.Sequential(
            DRDB(129,12,device = 'cuda:1'),
            DRDB(129,12,device = 'cuda:1'),
            DRDB(129,12,device = 'cuda:1'),
            DRDB(129,12,device = 'cuda:1')
        ).to('cuda:1')

        self.D_ds_conv3 = DS_conv(129, 2 * n_classes, stride=1, padding=1, dilation=1,device='cuda:1').to('cuda:1')
        self.D_ds_conv2 = DS_conv(4 * n_classes, 2 * n_classes, stride=1, padding=1, dilation=1,device='cuda:1').to('cuda:1')
        self.D_ds_conv1 = DS_conv(4 * n_classes, 2 * n_classes, stride=1, padding=1, dilation=1,device='cuda:1').to('cuda:1')

        #self.down = nn.functional.interpolate()
        self.conv = nn.Conv3d(in_channels=2 * n_classes, out_channels=n_classes, kernel_size=1, stride=1, padding=0).to('cuda:1')
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
        x1 = self.ds_conv3(x1.cuda(1))
        
        
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
        
        x1,cat_1,ir_1 = self.encoder_1(x)
        x1,cat_2,ir_2 = self.encoder_2(x1,ir_1)
        del ir_1
        x1 = self.encoder_3(x1,ir_2)
        del ir_2
        
        
        x1 = self.decoder_3(x1,cat_2)
        del cat_2
        x1 = self.decoder_2(x1,cat_1)
        del cat_1
        x1 = self.decoder_1(x1,x)
        del x
        
        # out
        x1 = self.conv(x1.to('cuda:1'))
        #x1 = self.softmax(x1)
        #x1 = self.sigmoid(x1)
        
        return x1        
    
    
    

#3d DS空洞卷积
class DS_conv(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, padding=1, dilation=1):
        super(DS_conv, self).__init__()
        
        
        
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride,
                               padding=padding, dilation=dilation,groups=in_channels)
        self.pointwise = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                   padding=0)
        self.bn_relu = nn.Sequential(
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False)
        )
        
        

    def forward(self, x):

        x = self.conv1(x)
        #x = self.conv1(x)
        x = self.pointwise(x)
        x = self.bn_relu(x)

        return x
#3d DS空洞反卷积
class DeDS_conv(nn.Module):

    def __init__(self, in_channels, out_channels,kernel_size = 2, stride=2, padding=0, dilation=1):
        super(DeDS_conv, self).__init__()
        
        
        
        self.conv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride,
                               padding=padding, dilation=dilation,groups=in_channels)
        self.pointwise = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                   padding=0)
        self.bn_relu = nn.Sequential(
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False)
        )
        
        

    def forward(self, x):

        x = self.conv1(x)
        #x = self.conv1(x)
        x = self.pointwise(x)
        x = self.bn_relu(x)

        return x
#PLS-Net里面提出的基础模块
class DRDB(nn.Module):

    def __init__(self, in_channels,g):
        super(DRDB, self).__init__()

        self.ds_con1 = DS_conv(in_channels, g, stride=1, padding=1, dilation=1)
        self.ds_con2 = DS_conv(in_channels+g*1, g, stride=1, padding=2, dilation=2)
        self.ds_con3 = DS_conv(in_channels+g*2, g, stride=1, padding=3, dilation=3)
        self.ds_con4 = DS_conv(in_channels+g*3, g, stride=1, padding=4, dilation=4)
        self.conv1 = nn.Conv3d(in_channels=in_channels+g*4, out_channels=in_channels, kernel_size=1, stride=1,
                               padding=0)

        self.bn_relu = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=False)
        )
        
        

    def forward(self, x):
        x1 = self.ds_con1(x)     # 1*
        x1 = torch.cat([x, x1], 1) # 2*
        x2 = self.ds_con2(x1)       # 1*
        x2 = torch.cat([x2, x1], 1) # 3*
        x3 = self.ds_con3(x2)       # 1*
        x3 = torch.cat([x3, x2], 1) # 4*
        
        del x1
        del x2
        
        x4 = self.ds_con4(x3)       # 1*
        x4 = torch.cat([x4, x3], 1) # 5*
        
        del x3
        
        x4 = self.conv1(x4)
        x4 = self.bn_relu(x4)
        x4 = x4+x
        
        del x

        return x4
    
    
class DRDB_sequential(nn.Module):

    def __init__(self, in_channels,g,device = 'cuda:0'):
        super(DRDB_sequential, self).__init__()

        
        
        self.ds_con1 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2,
                               padding=1, dilation=1,groups=in_channels),
            nn.Conv3d(in_channels=in_channels, out_channels=g, kernel_size=1, stride=1,
                                   padding=0),
            nn.BatchNorm3d(g),
            nn.ReLU(inplace=False)
            
        ).to(device)
        
        
        
        self.ds_con2 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels+g*1, out_channels=in_channels+g*1, kernel_size=3, stride=2,
                               padding=1, dilation=1,groups=in_channels+g*1),
            nn.Conv3d(in_channels=in_channels+g*1, out_channels=g, kernel_size=1, stride=1,
                                   padding=0),
            nn.BatchNorm3d(g),
            nn.ReLU(inplace=False)
            
        ).to(device)
        
        
        self.ds_con3 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels+g*2, out_channels=in_channels+g*2, kernel_size=3, stride=2,
                               padding=1, dilation=1,groups=in_channels+g*2),
            nn.Conv3d(in_channels=in_channels+g*2, out_channels=g, kernel_size=1, stride=1,
                                   padding=0),
            nn.BatchNorm3d(g),
            nn.ReLU(inplace=False)
            
        ).to(device)
        
        self.ds_con4 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels+g*3, out_channels=in_channels+g*3, kernel_size=3, stride=2,
                               padding=1, dilation=1,groups=in_channels+g*3),
            nn.Conv3d(in_channels=in_channels+g*3, out_channels=g, kernel_size=1, stride=1,
                                   padding=0),
            nn.BatchNorm3d(g),
            nn.ReLU(inplace=False)
            
        ).to(device)
        self.conv1 = nn.Conv3d(in_channels=in_channels+g*4, out_channels=in_channels, kernel_size=1, stride=1,
                               padding=0).to(device)

        self.bn_relu = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=False)
        ).to(device)
        
        self.device = device

    def forward(self, x):
        x1 = self.ds_con1(x.to(self.device))     # 1*
        x1 = torch.cat([x, x1], 1) # 2*
        x2 = self.ds_con2(x1)       # 1*
        x2 = torch.cat([x2, x1], 1) # 3*
        x3 = self.ds_con3(x2)       # 1*
        x3 = torch.cat([x3, x2], 1) # 4*
        
        del x1
        del x2
        
        x4 = self.ds_con4(x3)       # 1*
        x4 = torch.cat([x4, x3], 1) # 5*
        
        del x3
        
        x4 = self.conv1(x4)
        x4 = self.bn_relu(x4)
        x4 = x4 + x
        
        del x

        return x4    

class attention3d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature):
        super(attention3d, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        if in_planes != 3:
            hidden_planes = int(in_planes * ratios)+1
        else:
            hidden_planes = K
        # self.fc1 = nn.Conv3d(in_planes, hidden_planes, 1, bias=False)
        self.fc1 = DS_conv(in_planes, hidden_planes, 1)
        # self.fc2 = nn.Conv3d(hidden_planes, K, 1, bias=False)
        self.fc2 = DS_conv(hidden_planes, K, 1)
        self.temperature = temperature

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x / self.temperature, 1)

  
class Dynamic_conv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4, temperature=34):
        super(Dynamic_conv3d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention3d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None


        #TODO 初始化
        # nn.init.kaiming_uniform_(self.weight, )

    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):#将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        softmax_attention = self.attention(x)
        batch_size, in_planes, depth, height, width = x.size()
        x = x.view(1, -1, depth, height, width)# 变化成一个维度进行组卷积
        weight = self.weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size*self.out_planes, self.in_planes//self.groups, self.kernel_size, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv3d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv3d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-3), output.size(-2), output.size(-1))
        return output
