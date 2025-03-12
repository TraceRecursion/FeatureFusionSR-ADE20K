# test_segformer.py
import torch
import torch.nn.functional as F
from data.ade20k_loader import get_dataloader
from models.segformer import SegFormer
from utils.metrics import calculate_iou
from config import Config
import numpy as np

def test_segformer():
    # 数据加载（使用验证集）
    val_loader = get_dataloader(
        Config.VAL_DIR, 
        batch_size=Config.BATCH_SIZE, 
        size=Config.IMAGE_SIZE, 
        shuffle=False, 
        mode='seg'
    )
    
    # 模型加载
    model = SegFormer(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    model.eval()
    
    # IoU 统计
    class_ious = np.zeros(Config.NUM_CLASSES)
    class_counts = np.zeros(Config.NUM_CLASSES)
    total_samples = len(val_loader.dataset)
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images, targets = images.to(Config.DEVICE), targets.to(Config.DEVICE)
            outputs = model(images)
            preds = F.interpolate(outputs, size=targets.shape[1:], mode='bilinear', align_corners=False)
            preds = torch.argmax(preds, dim=1)  # [B, H, W]
            
            # 计算 IoU
            ious = calculate_iou(preds, targets, Config.NUM_CLASSES)
            for cls, iou in enumerate(ious):
                if not np.isnan(iou):
                    class_ious[cls] += iou
                    class_counts[cls] += 1
            
            processed_samples = (batch_idx + 1) * Config.BATCH_SIZE
            print(f"Processed {min(processed_samples, total_samples)}/{total_samples} samples")
    
    # 计算平均 IoU
    mean_iou = 0
    valid_classes = 0
    for cls in range(Config.NUM_CLASSES):
        if class_counts[cls] > 0:
            avg_iou = class_ious[cls] / class_counts[cls]
            print(f"Class {cls}: IoU = {avg_iou:.4f}")
            mean_iou += avg_iou
            valid_classes += 1
    
    mean_iou /= valid_classes
    print(f"Mean IoU (mIoU): {mean_iou:.4f}")

if __name__ == "__main__":
    test_segformer()