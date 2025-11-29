# Copyright (c) Phigent Robotics. All rights reserved.
# ---------------------------------------------
#  Modified by Seokha Moon
# ---------------------------------------------

import torch
from torch import nn
from mmcv.cnn import build_conv_layer, build_upsample_layer
from mmdet.models import BACKBONES

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x, spatial_out = self.spatial_attention(x)
        return x, spatial_out


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.fc1 = nn.Conv3d(channels, channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv3d(channels // reduction, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out), out


@BACKBONES.register_module()
class RefineNet(nn.Module):
    def __init__(self, channels, internal_channels, use_occupied_head=False):
        super(RefineNet, self).__init__()
        self.conv1 = nn.Conv3d(channels, internal_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm3d(internal_channels)
        
        self.conv2 = nn.Conv3d(internal_channels, internal_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(internal_channels)
        
        self.conv3 = nn.Conv3d(internal_channels, channels, kernel_size=1)
        self.bn3 = nn.BatchNorm3d(channels)
        
        self.cbam = CBAM(channels)
        self.relu = nn.ReLU()

        self.use_occupied_head = use_occupied_head
        if self.use_occupied_head:
            deconv_cfg = dict(type='deconv3d', bias=False)
            conv3d_cfg = dict(type='Conv3d', bias=False)
            up_sample = build_upsample_layer(deconv_cfg, 1, internal_channels, kernel_size=2, stride=2)
            self.up_block = nn.Sequential(up_sample,
                                          nn.BatchNorm3d(internal_channels),
                                          nn.ReLU(inplace=True))
            self.pred_occ_mask = build_conv_layer(
                            conv3d_cfg,
                            in_channels=internal_channels,
                            out_channels=1,
                            kernel_size=1,
                            stride=1,
                            padding=0)

    def forward(self, x):
        pred_occ_mask = None
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        out, spatial_out = self.cbam(out)
        if self.use_occupied_head:
            up_out = self.up_block(spatial_out)
            pred_occ_mask = self.pred_occ_mask(up_out).sigmoid()
        
        out += identity
        out = self.relu(out)
        return out, pred_occ_mask

__all__ = ["RefineNet"]

