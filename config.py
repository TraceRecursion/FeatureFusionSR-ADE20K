# config.py
import torch

class Config:
    # 数据集路径（需替换为实际路径）
    DATA_ROOT = "/Users/sydg/Documents/数据集/ADE20K_2021_17_01"
    TRAIN_DIR = f"{DATA_ROOT}/images/ADE/training"
    VAL_DIR = f"{DATA_ROOT}/images/ADE/validation"
    
    # 模型参数
    SCALE_FACTOR = 4  # 4x 超分辨率
    IMAGE_SIZE = 512  # 输入图像尺寸
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    LR = 0.0002
    WEIGHT_DECAY = 0.05
    
    # 设备
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"