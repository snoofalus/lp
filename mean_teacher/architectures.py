# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
# Changes were made by 
# Authors: A. Iscen, G. Tolias, Y. Avrithis, O. Chum. 2018.


import sys
import math
import itertools

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable, Function
from torch.nn.utils import weight_norm

from .utils import export, parameter_count

import pdb


#cnns
#----------------------------------------------------------------------------------------------------
@export
def cifar_cnn(pretrained=False, **kwargs):
    assert not pretrained
    model = CNN(**kwargs)
    return model

@export
def s1s2glcm_CNN16(pretrained=False, **kwargs):
    assert not pretrained
    model = CNN16(c_in=180, **kwargs)
    return model

@export
def s1s2glcm_CNN8(pretrained=False, **kwargs):
    assert not pretrained
    model = CNN8(c_in=180, **kwargs)
    return model

@export
def s1_CNN8(pretrained=False, **kwargs):
    assert not pretrained
    model = CNN8(c_in=2, **kwargs)
    return model

@export
def s2_CNN8(pretrained=False, **kwargs):
    assert not pretrained
    model = CNN8(c_in=13, **kwargs)
    return model

@export
def glcm_CNN8(pretrained=False, **kwargs):
    assert not pretrained
    model = CNN8(c_in=165, **kwargs)
    return model


#resnets
#----------------------------------------------------------------------------------------------------

@export
def s1s2glcm_RES(pretrained=False, **kwargs):
    assert not pretrained
    model = WideResNet(c_in=180, **kwargs)
    return model

@export
def s1s2glcm_RES(pretrained=False, **kwargs):
    assert not pretrained
    model = WideResNet(c_in=180, **kwargs)
    return model


#conv nets, 16 and 8 layered
#----------------------------------------------------------------------------------------------------
class CNN(nn.Module):
    """
    CNN from Mean Teacher paper
    """
    
    def __init__(self, num_classes=10, isL2 = False, double_output = False):
        super(CNN, self).__init__()

        self.isL2 = isL2
        self.double_output = double_output

        self.activation = nn.LeakyReLU(0.1)
        self.conv1a = weight_norm(nn.Conv2d(3, 128, 3, padding=1))
        self.bn1a = nn.BatchNorm2d(128)
        self.conv1b = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1b = nn.BatchNorm2d(128)
        self.conv1c = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1c = nn.BatchNorm2d(128)
        self.mp1 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop1  = nn.Dropout(0.5)
        
        self.conv2a = weight_norm(nn.Conv2d(128, 256, 3, padding=1))
        self.bn2a = nn.BatchNorm2d(256)
        self.conv2b = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2b = nn.BatchNorm2d(256)
        self.conv2c = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2c = nn.BatchNorm2d(256)
        self.mp2 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop2  = nn.Dropout(0.5)
        
        self.conv3a = weight_norm(nn.Conv2d(256, 512, 3, padding=0))
        self.bn3a = nn.BatchNorm2d(512)
        self.conv3b = weight_norm(nn.Conv2d(512, 256, 1, padding=0))
        self.bn3b = nn.BatchNorm2d(256)
        self.conv3c = weight_norm(nn.Conv2d(256, 128, 1, padding=0))
        self.bn3c = nn.BatchNorm2d(128)
        self.ap3 = nn.AvgPool2d(6, stride=2, padding=0)
        
        self.fc1 =  weight_norm(nn.Linear(128, num_classes))

        if self.double_output:
            self.fc2 =  weight_norm(nn.Linear(128, num_classes))
    
    def forward(self, x, debug=False):

        x = self.activation(self.bn1a(self.conv1a(x)))
        x = self.activation(self.bn1b(self.conv1b(x)))
        x = self.activation(self.bn1c(self.conv1c(x)))
        x = self.mp1(x)
        x = self.drop1(x)
        
        x = self.activation(self.bn2a(self.conv2a(x)))
        x = self.activation(self.bn2b(self.conv2b(x)))
        x = self.activation(self.bn2c(self.conv2c(x)))
        x = self.mp2(x)
        x = self.drop2(x)
        
        x = self.activation(self.bn3a(self.conv3a(x)))
        x = self.activation(self.bn3b(self.conv3b(x)))
        x = self.activation(self.bn3c(self.conv3c(x)))
        x = self.ap3(x)
 
        x = x.view(-1, 128)
        if self.isL2:
            x = F.normalize(x)

        if self.double_output:
            return self.fc1(x), self.fc2(x), x
        else:
            return self.fc1(x), x
            
class CNN16(nn.Module):
    """
    CNN from Mean Teacher paper
    """
    
    def __init__(self, c_in, num_classes=6, isL2 = False, double_output = False):
        super(CNN16, self).__init__()

        self.isL2 = isL2
        self.double_output = double_output

        self.activation = nn.LeakyReLU(0.1)
        #Conv2d(C_in,C_out,kernel-size)
        self.conv1a = weight_norm(nn.Conv2d(c_in, 128, 3, padding=1))
        self.bn1a = nn.BatchNorm2d(128)
        self.conv1b = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1b = nn.BatchNorm2d(128)
        self.conv1c = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1c = nn.BatchNorm2d(128)
        self.mp1 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop1  = nn.Dropout(0.5)
        
        self.conv2a = weight_norm(nn.Conv2d(128, 256, 3, padding=1))
        self.bn2a = nn.BatchNorm2d(256)
        self.conv2b = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2b = nn.BatchNorm2d(256)
        self.conv2c = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2c = nn.BatchNorm2d(256)
        self.mp2 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop2  = nn.Dropout(0.5)
        
        self.conv3a = weight_norm(nn.Conv2d(256, 512, 3, padding=0))
        self.bn3a = nn.BatchNorm2d(512)
        self.conv3b = weight_norm(nn.Conv2d(512, 256, 1, padding=0))
        self.bn3b = nn.BatchNorm2d(256)
        self.conv3c = weight_norm(nn.Conv2d(256, 128, 1, padding=0))
        self.bn3c = nn.BatchNorm2d(128)
        self.ap3 = nn.AvgPool2d(6, stride=2, padding=0)
        
        self.fc1 =  weight_norm(nn.Linear(128, num_classes))

        if self.double_output:
            self.fc2 =  weight_norm(nn.Linear(128, num_classes))
    
    def forward(self, x, debug=False):

        x = self.activation(self.bn1a(self.conv1a(x)))
        x = self.activation(self.bn1b(self.conv1b(x)))
        x = self.activation(self.bn1c(self.conv1c(x)))
        x = self.mp1(x)
        x = self.drop1(x)
        
        x = self.activation(self.bn2a(self.conv2a(x)))
        x = self.activation(self.bn2b(self.conv2b(x)))
        x = self.activation(self.bn2c(self.conv2c(x)))
        x = self.mp2(x)
        x = self.drop2(x)
        
        x = self.activation(self.bn3a(self.conv3a(x)))
        x = self.activation(self.bn3b(self.conv3b(x)))
        x = self.activation(self.bn3c(self.conv3c(x)))
        x = self.ap3(x)
 
        x = x.view(-1, 128)
        if self.isL2:
            x = F.normalize(x)

        if self.double_output:
            return self.fc1(x), self.fc2(x), x
        else:
            return self.fc1(x), x

class CNN8(nn.Module):
    """
    CNN from Mean Teacher paper
    """
    
    def __init__(self, c_in, num_classes=6, isL2 = False, double_output = False):
        super(CNN8, self).__init__()

        self.isL2 = isL2
        self.double_output = double_output

        self.activation = nn.LeakyReLU(0.1)
        #Conv2d(C_in,C_out,kernel-size)
        self.conv1a = weight_norm(nn.Conv2d(c_in, 128, 3, padding=1))
        self.bn1a = nn.BatchNorm2d(128)
        self.conv1b = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1b = nn.BatchNorm2d(128)
        self.mp1 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop1  = nn.Dropout(0.5)
        
        self.conv2a = weight_norm(nn.Conv2d(128, 256, 3, padding=1))
        self.bn2a = nn.BatchNorm2d(256)
        self.conv2b = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2b = nn.BatchNorm2d(256)
        self.mp2 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop2  = nn.Dropout(0.5)
        
        self.conv3a = weight_norm(nn.Conv2d(256, 256, 3, padding=0))
        self.bn3a = nn.BatchNorm2d(256)
        self.conv3b = weight_norm(nn.Conv2d(256, 128, 1, padding=0))
        self.bn3b = nn.BatchNorm2d(128)
        self.ap3 = nn.AvgPool2d(6, stride=2, padding=0)
        
        self.fc1 =  weight_norm(nn.Linear(128, num_classes))

        if self.double_output:
            self.fc2 =  weight_norm(nn.Linear(128, num_classes))
    
    def forward(self, x, debug=False):

        x = self.activation(self.bn1a(self.conv1a(x)))
        x = self.activation(self.bn1b(self.conv1b(x)))
        x = self.mp1(x)
        x = self.drop1(x)
        
        x = self.activation(self.bn2a(self.conv2a(x)))
        x = self.activation(self.bn2b(self.conv2b(x)))
        x = self.mp2(x)
        x = self.drop2(x)
        
        x = self.activation(self.bn3a(self.conv3a(x)))
        x = self.activation(self.bn3b(self.conv3b(x)))
        x = self.ap3(x)
 
        x = x.view(-1, 128)
        if self.isL2:
            x = F.normalize(x)

        if self.double_output:
            return self.fc1(x), self.fc2(x), x
        else:
            return self.fc1(x), x

#wide resnet 28-2
#----------------------------------------------------------------------------------------------------
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual
    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, activate_before_residual))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, c_in, num_classes, depth=28, widen_factor=2, dropRate=0.0, isL2 = False, double_output = False):

        super(WideResNet, self).__init__()
        self.isL2=isL2
        self.double_output = double_output
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        #self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
        self.conv1 = nn.Conv2d(c_in, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc1 = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

        if self.double_output:
            self.fc2 = weight_norm(nn.Linear(nChannels[3], num_classes))

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)

        if self.isL2:
            out = F.normalize(out)

        if self.double_output:
            return self.fc1(out), self.fc2(out), out
        else: 
            return self.fc1(out), out
