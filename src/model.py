import torch
import torch.nn as nn
import math
from torchvision.models.mobilenetv2 import InvertedResidual
from torchvision.ops import Conv2dNormActivation as ConvBNReLU



class MobileNetV2_CIFAR10(nn.Module):
    """
    MobileNetV2 adapted for CIFAR-10 (32x32 images).
    Modified strides: 1st, 2nd bottlenecks all use stride=1.
    """
    def __init__(self, num_classes=10, width_mult=1.0, dropout=0.2):
        super(MobileNetV2_CIFAR10, self).__init__()
        
        # MobileNetV2 configuration: [expansion, output_channels, num_blocks, stride]
        # Modified: First 3 stages use stride=1 for CIFAR-10
        inverted_residual_setting = [
            [1, 16, 1, 1],   # 1st bottleneck: stride=1 (unchanged)
            [6, 24, 2, 1],   # 2nd bottleneck: stride=1 (changed from 2)  
            [6, 32, 3, 2],   # Remaining kept same
            [6, 64, 4, 2],   
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        # Building first layer
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * max(1.0, width_mult))
        
        # First convolution - stride 1 for CIFAR-10
        features = [ConvBNReLU(
            in_channels=3, 
            out_channels=input_channel, 
            kernel_size=3,
            stride=1,
            padding=1,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.ReLU6
        )]
        
        # Building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(
                    inp=input_channel, 
                    oup=output_channel, 
                    stride=stride, 
                    expand_ratio=t,
                    norm_layer=nn.BatchNorm2d
                ))
                input_channel = output_channel
        
        # Building last several layers using ConvBNReLU
        features.append(ConvBNReLU(
            in_channels=input_channel, 
            out_channels=self.last_channel, 
            kernel_size=1,
            stride=1,
            padding=0,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.ReLU6
        ))
        
        # Making it nn.Sequential
        self.features = nn.Sequential(*features)
        
        # Building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.last_channel, num_classes),
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
