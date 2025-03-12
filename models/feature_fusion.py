# models/feature_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel Attention
        self.channel_avg = nn.AdaptiveAvgPool2d(1)
        self.channel_max = nn.AdaptiveMaxPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.channel_sigmoid = nn.Sigmoid()
        
        # Spatial Attention
        self.spatial_conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.spatial_sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Channel attention
        avg_out = self.channel_fc(self.channel_avg(x))
        max_out = self.channel_fc(self.channel_max(x))
        ca = self.channel_sigmoid(avg_out + max_out)
        x = x * ca
        
        # Spatial attention
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_spatial, max_spatial], dim=1)
        sa = self.spatial_sigmoid(self.spatial_conv(spatial))
        x = x * sa
        
        return x

class FeatureFusion(nn.Module):
    def __init__(self, semantic_channels=256, low_channels=256, out_channels=256):
        super().__init__()
        self.conv_reduce = nn.Conv2d(semantic_channels + low_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(out_channels)
        
    def forward(self, semantic_features, low_features):
        # 上采样语义特征至低级特征分辨率
        semantic_features = F.interpolate(semantic_features, size=low_features.shape[2:], mode='bilinear', align_corners=False)
        # 特征拼接
        fused = torch.cat([semantic_features, low_features], dim=1)
        fused = self.conv_reduce(fused)
        fused = self.bn(fused)
        fused = self.relu(fused)
        fused = self.cbam(fused)
        return fused