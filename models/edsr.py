# models/edsr.py
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out + residual

class EDSR(nn.Module):
    def __init__(self, in_channels=256, out_channels=3, num_blocks=32, scale_factor=4):
        super().__init__()
        self.entry = nn.Conv2d(in_channels, 64, 3, padding=1, bias=False)
        self.bn_entry = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # 残差块
        self.blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_blocks)])
        self.middle = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.bn_middle = nn.BatchNorm2d(64)
        
        # 上采样（4x 分两次 2x）
        upsample_layers = []
        for _ in range(2):  # 4x = 2x * 2x
            upsample_layers.extend([
                nn.Conv2d(64, 64 * 4, 3, padding=1, bias=False),
                nn.BatchNorm2d(64 * 4),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True)
            ])
        self.upsample = nn.Sequential(*upsample_layers)
        
        self.exit = nn.Conv2d(64, out_channels, 3, padding=1)
        
    def forward(self, x):
        out = self.entry(x)
        out = self.bn_entry(out)
        out = self.relu(out)
        
        residual = out
        out = self.blocks(out)
        out = self.middle(out)
        out = self.bn_middle(out)
        out = out + residual
        
        out = self.upsample(out)
        out = self.exit(out)
        return torch.clamp(out, 0, 1)  # 输出范围 [0, 1]