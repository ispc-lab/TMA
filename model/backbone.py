import torch
import torch.nn as nn
import torch.nn.functional as F


class Resblock(nn.Module):
    def __init__(self, inchannel, outchannel, norm='BN', stride=1):
        super(Resblock, self).__init__()
        self.conv1 = nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outchannel, outchannel, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        if norm == 'BN':
            self.norm1 = nn.BatchNorm2d(outchannel)
            self.norm2 = nn.BatchNorm2d(outchannel)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(outchannel)

        elif norm =='IN':
            self.norm1 = nn.InstanceNorm2d(outchannel)
            self.norm2 = nn.InstanceNorm2d(outchannel)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(outchannel)
        
        elif norm == 'None':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()
        
        if stride == 1:
            self.downsample = None
            
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride),
                self.norm3
                )
        
    def forward(self, x):
        y = x
        
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)

class ExtractorF(nn.Module):
    def __init__(self, input_channel=15, outchannel=128, norm='IN'):
        super(ExtractorF, self).__init__()
        self.norm = norm
        if self.norm == 'BN':
            self.norm1 = nn.BatchNorm2d(32)           
        elif self.norm == 'IN':
            self.norm1 = nn.InstanceNorm2d(32)       
        elif self.norm == 'NONE':
            self.norm1 = nn.Sequential()
        
        self.conv1 = nn.Conv2d(input_channel, 32, kernel_size=7, padding=3, stride=2)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 32
        self.layer1 = self._make_layer(32,  stride=1)
        self.layer2 = self._make_layer(64, stride=2)
        self.layer3 = self._make_layer(96, stride=2)

        self.conv2 = nn.Conv2d(96, outchannel, kernel_size=1, padding=0)

    def _make_layer(self, dim, stride=1):
        layer1 = Resblock(self.in_planes, dim, self.norm, stride=stride)
        layer2 = Resblock(dim, dim, self.norm, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)    
                
    def forward(self,x):
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            num_group = len(x)
            x = torch.cat(x, dim=0)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)
        if is_list:
            x = x.chunk(num_group, dim=0)
        return x#Spatial size: 1/8

class ExtractorC(nn.Module):
    def __init__(self, input_channel=5, outchannel=128, norm='IN'):
        super(ExtractorC, self).__init__()
        self.norm = norm
        if self.norm == 'BN':
            self.norm1 = nn.BatchNorm2d(64)           
        elif self.norm == 'IN':
            self.norm1 = nn.InstanceNorm2d(64)       
        elif self.norm == 'None':
            self.norm1 = nn.Sequential()
        
        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, padding=3, stride=2)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64,  stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)

        self.conv2 = nn.Conv2d(128, outchannel, kernel_size=1, padding=0)

    def _make_layer(self, dim, stride=1):
        layer1 = Resblock(self.in_planes, dim, self.norm, stride=stride)
        layer2 = Resblock(dim, dim, self.norm, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)    
                
    def forward(self,x):             
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)
        return x
