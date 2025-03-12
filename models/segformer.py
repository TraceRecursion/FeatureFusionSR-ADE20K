# models/segformer.py
import torch
import torch.nn as nn
from timm import create_model

class SegFormerFeatureExtractor(nn.Module):
    def __init__(self, model_name="nvidia/segformer-b5-finetuned-ade-640-640", pretrained=True):
        super().__init__()
        self.segformer = create_model(model_name, pretrained=pretrained, features_only=True)
        
    def forward(self, x):
        # 获取多尺度特征，返回最后一层
        features = self.segformer(x)
        return features[-1]  # 形状例如 [B, 256, H/4, W/4]