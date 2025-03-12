# config.py
import torch
import os
from pathlib import Path

class Config:
    USER_HOME = os.path.expanduser("~")
    
    DATA_ROOT = os.environ.get(
        "ADE20K_DATASET_PATH", 
        os.path.join(USER_HOME, "Documents", "数据集", "ADE20K_2021_17_01")
    )
    TRAIN_DIR = os.path.join(DATA_ROOT, "images", "ADE", "training")
    VAL_DIR = os.path.join(DATA_ROOT, "images", "ADE", "validation")
    
    # 模型参数
    SCALE_FACTOR = 4
    IMAGE_SIZE = 512
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    LR = 0.0002
    WEIGHT_DECAY = 0.05
    NUM_CLASSES = 150  # ADE20K 有 150 个类别
    
    # 设备
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    @classmethod
    def set_data_root(cls, new_path):
        """允许在运行时更改数据集根路径"""
        cls.DATA_ROOT = new_path
        cls.TRAIN_DIR = os.path.join(cls.DATA_ROOT, "images", "ADE", "training")
        cls.VAL_DIR = os.path.join(cls.DATA_ROOT, "images", "ADE", "validation")