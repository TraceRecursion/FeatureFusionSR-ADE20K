# train.py
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data.ade20k_loader import get_dataloader
from models.sr_model import FeatureFusionSR
from utils.metrics import calculate_psnr, calculate_ssim
from utils.training_utils import save_checkpoint
from config import Config
from tqdm import tqdm

def train():
    train_loader = get_dataloader(Config.TRAIN_DIR, Config.BATCH_SIZE)
    val_loader = get_dataloader(Config.VAL_DIR, Config.BATCH_SIZE, shuffle=False)
    
    model = FeatureFusionSR().to(Config.DEVICE)
    optimizer = AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    criterion = nn.L1Loss()
    
    best_psnr = 0
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        train_loss = 0
        
        # 训练进度条
        with tqdm(total=len(train_loader.dataset), desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - Training", unit="sample") as pbar:
            for batch_idx, (lr_imgs, hr_imgs) in enumerate(train_loader):
                lr_imgs, hr_imgs = lr_imgs.to(Config.DEVICE), hr_imgs.to(Config.DEVICE)
                optimizer.zero_grad()
                sr_imgs = model(lr_imgs)
                loss = criterion(sr_imgs, hr_imgs)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
                pbar.update(Config.BATCH_SIZE)
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}, Train Loss: {avg_train_loss:.4f}")
        
        model.eval()
        val_psnr, val_ssim = 0, 0
        val_loss = 0
        
        # 验证进度条
        with torch.no_grad():
            with tqdm(total=len(val_loader.dataset), desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - Validation", unit="sample") as pbar:
                for lr_imgs, hr_imgs in enumerate(val_loader):
                    lr_imgs, hr_imgs = lr_imgs.to(Config.DEVICE), hr_imgs.to(Config.DEVICE)
                    sr_imgs = model(lr_imgs)
                    loss = criterion(sr_imgs, hr_imgs)
                    val_loss += loss.item()
                    val_psnr += calculate_psnr(sr_imgs, hr_imgs)
                    val_ssim += calculate_ssim(sr_imgs, hr_imgs)
                    
                    pbar.update(Config.BATCH_SIZE)
                    pbar.set_postfix({"Val Loss": f"{loss.item():.4f}"})
        
        val_loss /= len(val_loader)
        val_psnr /= len(val_loader)
        val_ssim /= len(val_loader)
        print(f"Val Loss: {val_loss:.4f}, Val PSNR: {val_psnr:.2f}, Val SSIM: {val_ssim:.4f}")
        
        scheduler.step(val_loss)
        
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint(model, optimizer, epoch, "best_model.pth")

if __name__ == "__main__":
    train()