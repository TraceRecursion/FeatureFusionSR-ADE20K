# utils/metrics.py
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def calculate_psnr(sr, hr, max_val=1.0):
    sr = sr.cpu().detach().numpy()
    hr = hr.cpu().detach().numpy()
    batch_size = sr.shape[0]
    psnr = 0
    for i in range(batch_size):
        sr_img = np.transpose(sr[i], (1, 2, 0))
        hr_img = np.transpose(hr[i], (1, 2, 0))
        psnr += peak_signal_noise_ratio(hr_img, sr_img, data_range=max_val)
    return psnr / batch_size

def calculate_ssim(sr, hr):
    sr = sr.cpu().detach().numpy()
    hr = hr.cpu().detach().numpy()
    batch_size = sr.shape[0]
    ssim = 0
    for i in range(batch_size):
        sr_img = np.transpose(sr[i], (1, 2, 0))
        hr_img = np.transpose(hr[i], (1, 2, 0))
        ssim += structural_similarity(hr_img, sr_img, channel_axis=2, data_range=1.0)
    return ssim / batch_size

def calculate_iou(pred, target, num_classes):
    pred = pred.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    ious = []
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        if union == 0:
            iou = float('nan')
        else:
            iou = intersection / union
        ious.append(iou)
    return ious