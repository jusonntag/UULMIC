'''
Pytorch implementation of EEGNet from the official githup repo from the paper:
- https://iopscience.iop.org/article/10.1088/1741-2552/aace8c
- https://github.com/vlawhern/arl-eegmodels / https://arxiv.org/pdf/1611.08024

'''

import torch
from torch import nn
import torch.nn.functional as F
from src.core.domain.config import EEGNetConfig


class KerasBasicDropuout2D(nn.Module):
    def __init__(self, p):
        super().__init__()
        
        # Keras’ SpatialDropout1D in PyTorch:
        x = x.permute(0, 2, 1)   # convert to [batch, channels, time]
        x = F.dropout2d(x, p, training=self.training)
        x = x.permute(0, 2, 1)   # back to [batch, time, channels]


    def forward(self):
        pass

class ConstrainedConv2d(nn.Conv2d):
    """L2Norm constrained Conv2D block"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.conv2d(
            input = x,
            weight = self.weight.clamp(max=1, min = -1),
            bias = self.bias,
            stride = self.stride,
            padding = self.padding,
            dilation = self.dilation,
            groups = self.groups,
            )
        return x


class DepthwiseConv2D(nn.Module):
    def __init__(self, in_channels: int, depth_multiplier: int, n_channels: int, bias: bool = False):
        super().__init__()
        # Total output channels F2 = in_channels (F1) * depth_multiplier (D)
        self.depth_conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels = in_channels * depth_multiplier,
            kernel_size = (n_channels, 1),
            groups = in_channels,
            stride = 1,
            padding = 0,
            bias = bias,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth_conv(x)
        return x


class SeparableConv2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.sepconv = nn.Conv2d(
                in_channels = out_channels,
                out_channels = out_channels,
                kernel_size = (1,1),
                stride = 1,
                padding = 'same',
                bias = bias,
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sepconv(x)
        return x
    

class Conv2D(nn.Module):
    def __init__(self, out_channels: int, kern_length: int, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels = 1,
            out_channels = out_channels,
            kernel_size = (1, kern_length),
            stride = 1,
            padding = 'same',
            bias = bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class EEGNet(nn.Module):
    def __init__(self, config: EEGNetConfig):
        super().__init__()
        self.config = config

        if config.dropoutType == 'spatial':
            # Pytoch dropout2d == keras spatialdropout2d
            dropout_layer = nn.Dropout2d(config.dropoutRate) 
        elif config.dropoutType == 'base':
            dropout_layer = nn.Dropout(config.dropoutRate)
        else:
            raise ValueError("dropoutType must be 'spatial' or 'base'.")
        
        # block 1
        self.block1 = nn.Sequential(
            Conv2D(out_channels=config.F1, kern_length=config.kernLength, bias=config.bias),
            nn.BatchNorm2d(config.F1),
            DepthwiseConv2D(in_channels=config.F1, depth_multiplier=config.D, n_channels=config.channels, bias=config.bias),
            nn.BatchNorm2d(config.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            dropout_layer,
        )
        
        # block 2
        self.block2 = nn.Sequential(
            SeparableConv2D(in_channels=config.F2, out_channels=config.F2, bias=config.bias),
            nn.BatchNorm2d(config.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            dropout_layer,
        )

        if config.output_activation == "log_softmax":
            activation = nn.LogSoftmax(dim=1)
        elif config.output_activation == "softmax":
            activation = nn.Softmax(dim=1)
        elif config.output_activation == "sigmoid":
            activation = nn.Sigmoid()
        else: # default linear
            activation = nn.Identity()

        # classification block
        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features = self._get_num_inputs(),
                out_features = config.n_classes,
            ),
            activation,
        )

    def _get_num_inputs(self):
        """Mocks input shape for last linear layer dynamically."""
        with torch.no_grad():
            dummy_data = torch.zeros(1, 1, self.config.channels, self.config.samples)

            dummy_data = self.block1(dummy_data)
            dummy_data = self.block2(dummy_data)

        return self.config.F2 * dummy_data.shape[3]


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Shape of input data should be (Batch, Channels, Samples)."""
        assert x.ndim == 3, f"Input data must be 3D (Batch, Channels, Samples), but got {x.ndim}D"
        # Add channel dimension
        x = x.unsqueeze(1)
        # block 1
        x = self.block1(x)

        # block 2
        x = self.block2(x)

        # classification
        x = self.classify(x)
        return x