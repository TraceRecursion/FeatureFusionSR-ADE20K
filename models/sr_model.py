# models/sr_model.py
import torch
import torch.nn as nn
from .segformer import SegFormerFeatureExtractor
from .feature_fusion import FeatureFusion
from .edsr import EDSR

class FeatureFusionSR(nn.Module):
    def __init__(self):
        super().__init__()
        self.segformer = SegFormerFeatureExtractor()
        # 低级特征提取器
        self.low_feature_extractor = nn.Sequential(
            nn.Conv2d(3, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.fusion = FeatureFusion(semantic_channels=256, low_channels=256, out_channels=256)
        self.reconstruction = EDSR(in_channels=256, out_channels=3)
        
    def forward(self, lr_img):
        # 提取语义特征
        semantic_features = self.segformer(lr_img)
        # 提取低级特征
        low_features = self.low_feature_extractor(lr_img)
        # 特征融合
        fused_features = self.fusion(semantic_features, low_features)
        # 重建高分辨率图像
        hr_img = self.reconstruction(fused_features)
        return hr_img