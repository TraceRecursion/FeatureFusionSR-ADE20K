# data/ade20k_loader.py
import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ADE20KDataset(Dataset):
    def __init__(self, image_dir, scale_factor=4, size=512):
        self.image_dir = image_dir
        self.scale_factor = scale_factor
        self.size = size
        
        # 递归获取所有图像文件
        self.image_files = []
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    self.image_files.append(os.path.join(root, file))
        self.image_files = sorted(self.image_files)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        hr_img = cv2.imread(img_path)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
        hr_img = cv2.resize(hr_img, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        
        # 生成低分辨率 (LR) 图像
        lr_size = self.size // self.scale_factor
        lr_img = cv2.resize(hr_img, (lr_size, lr_size), interpolation=cv2.INTER_AREA)
        
        # 转换为张量
        hr_img = self.transform(hr_img)
        lr_img = self.transform(lr_img)
        
        return lr_img, hr_img

def get_dataloader(image_dir, batch_size, scale_factor=4, size=512, shuffle=True):
    dataset = ADE20KDataset(image_dir, scale_factor, size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)