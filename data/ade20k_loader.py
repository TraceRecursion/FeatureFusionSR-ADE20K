# data/ade20k_loader.py
import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ADE20KDataset(Dataset):
    def __init__(self, data_dir, scale_factor=4, size=512, mode='sr'):
        self.data_dir = data_dir
        self.scale_factor = scale_factor
        self.size = size
        self.mode = mode  # 'sr' for super-resolution, 'seg' for segmentation
        
        # 递归获取所有图像文件（仅 .jpg 文件）
        self.image_files = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.jpg'):
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
        
        if self.mode == 'sr':
            lr_size = self.size // self.scale_factor
            lr_img = cv2.resize(hr_img, (lr_size, lr_size), interpolation=cv2.INTER_AREA)
            hr_img = self.transform(hr_img)
            lr_img = self.transform(lr_img)
            return lr_img, hr_img
        elif self.mode == 'seg':
            ann_path = img_path.replace('.jpg', '_seg.png')
            if not os.path.exists(ann_path):
                raise FileNotFoundError(f"Segmentation file not found: {ann_path}")
            ann_img = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)
            ann_img = cv2.resize(ann_img, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
            hr_img = self.transform(hr_img)
            ann_img = torch.from_numpy(ann_img).long()  # 类别索引从 0 到 149
            return hr_img, ann_img
        else:
            raise ValueError("Mode must be 'sr' or 'seg'")

def get_dataloader(data_dir, batch_size, scale_factor=4, size=512, shuffle=True, mode='sr'):
    dataset = ADE20KDataset(data_dir, scale_factor, size, mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)