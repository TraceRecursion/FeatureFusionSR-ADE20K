# models/segformer.py
import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation

class SegFormer(nn.Module):
    def __init__(self, model_name="nvidia/segformer-b5-finetuned-ade-640-640", num_classes=150, pretrained=True):
        super().__init__()
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True  # 允许调整类别数
        ) if pretrained else SegformerForSemanticSegmentation.from_config(num_classes=num_classes)
        
    def forward(self, x):
        outputs = self.segformer(x)
        return outputs.logits  # [B, num_classes, H/4, W/4]，需要上采样