# utils/metrics.py
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def calculate_psnr(sr, hr, max_val=1.0):
    """
    计算 PSNR
    sr: 超分辨率图像 (Tensor, [B, C, H, W])
    hr: 高分辨率图像 (Tensor, [B, C, H, W])
    """
    sr = sr.cpu().detach().numpy()
    hr = hr.cpu().detach().numpy()
    batch_size = sr.shape[0]
    psnr = 0
    for i in range(batch_size):
        sr_img = np.transpose(sr[i], (1, 2, 0))  # [C, H, W] -> [H, W, C]
        hr_img = np.transpose(hr[i], (1, 2, 0))
        psnr += peak_signal_noise_ratio(hr_img, sr_img, data_range=max_val)
    return psnr / batch_size

def calculate_ssim(sr, hr):
    """
    计算 SSIM
    sr: 超分辨率图像 (Tensor, [B, C, H, W])
    hr: 高分辨率图像 (Tensor, [B, C, H, W])
    """
    sr = sr.cpu().detach().numpy()
    hr = hr.cpu().detach().numpy()
    batch_size = sr.shape[0]
    ssim = 0
    for i in range(batch_size):
        sr_img = np.transpose(sr[i], (1, 2, 0))  # [C, H, W] -> [H, W, C]
        hr_img = np.transpose(hr[i], (1, 2, 0))
        ssim += structural_similarity(hr_img, sr_img, channel_axis=2, data_range=1.0)
    return ssim / batch_size