# models/segformer.py
import torch
import torch.nn as nn
from timm import create_model

class SegFormer(nn.Module):
    def __init__(self, model_name="nvidia/segformer-b5-finetuned-ade-640-640", num_classes=150, pretrained=True):
        super().__init__()
        self.segformer = create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        
    def forward(self, x):
        logits = self.segformer(x)
        return logits  # [B, num_classes, H, W]