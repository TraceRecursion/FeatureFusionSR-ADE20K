# test.py
import torch
from data.ade20k_loader import get_dataloader
from models.sr_model import FeatureFusionSR
from utils.metrics import calculate_psnr, calculate_ssim
from config import Config
from tqdm import tqdm

def test(model_path, test_loader):
    model = FeatureFusionSR().to(Config.DEVICE)
    checkpoint = torch.load(model_path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_psnr, test_ssim = 0, 0
    with torch.no_grad():
        # 测试进度条
        with tqdm(total=len(test_loader.dataset), desc="Testing SR Model", unit="sample") as pbar:
            for batch_idx, (lr_imgs, hr_imgs) in enumerate(test_loader):
                lr_imgs, hr_imgs = lr_imgs.to(Config.DEVICE), hr_imgs.to(Config.DEVICE)
                sr_imgs = model(lr_imgs)
                test_psnr += calculate_psnr(sr_imgs, hr_imgs)
                test_ssim += calculate_ssim(sr_imgs, hr_imgs)
                
                pbar.update(Config.BATCH_SIZE)
                pbar.set_postfix({"PSNR": f"{test_psnr/(batch_idx+1):.2f}", "SSIM": f"{test_ssim/(batch_idx+1):.4f}"})
    
    test_psnr /= len(test_loader)
    test_ssim /= len(test_loader)
    print(f"Final Test PSNR: {test_psnr:.2f}, Test SSIM: {test_ssim:.4f}")

if __name__ == "__main__":
    test_loader = get_dataloader(Config.VAL_DIR, Config.BATCH_SIZE, shuffle=False)
    test("best_model.pth", test_loader)