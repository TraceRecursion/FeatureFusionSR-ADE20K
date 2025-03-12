# test_segformer.py
import torch
import torch.nn.functional as F
from data.ade20k_loader import get_dataloader
from models.segformer import SegFormer
from utils.metrics import calculate_iou
from config import Config
import numpy as np
from tqdm import tqdm

def test_segformer():
    val_loader = get_dataloader(
        Config.VAL_DIR,
        batch_size=Config.BATCH_SIZE,
        size=Config.IMAGE_SIZE,
        shuffle=False,
        mode='seg'
    )
    
    model = SegFormer(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    model.eval()
    
    class_ious = np.zeros(Config.NUM_CLASSES)
    class_counts = np.zeros(Config.NUM_CLASSES)
    total_samples = len(val_loader.dataset)
    
    with torch.no_grad():
        # 使用 tqdm 包装 val_loader，显示进度条
        with tqdm(total=total_samples, desc="Testing SegFormer", unit="sample") as pbar:
            for batch_idx, (images, targets) in enumerate(val_loader):
                images, targets = images.to(Config.DEVICE), targets.to(Config.DEVICE)
                outputs = model(images)
                preds = F.interpolate(outputs, size=targets.shape[1:], mode='bilinear', align_corners=False)
                preds = torch.argmax(preds, dim=1)
                
                ious = calculate_iou(preds, targets, Config.NUM_CLASSES)
                for cls, iou in enumerate(ious):
                    if not np.isnan(iou):
                        class_ious[cls] += iou
                        class_counts[cls] += 1
                
                # 更新进度条
                processed_samples = min((batch_idx + 1) * Config.BATCH_SIZE, total_samples)
                pbar.update(Config.BATCH_SIZE)
                pbar.set_postfix({"Processed": f"{processed_samples}/{total_samples}"})
    
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