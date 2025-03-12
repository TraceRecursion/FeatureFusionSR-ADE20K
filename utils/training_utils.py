# utils/training_utils.py
import torch

def save_checkpoint(model, optimizer, epoch, path):
    """保存模型检查点"""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(state, path)
    print(f"Checkpoint saved at {path}")

def load_checkpoint(model, optimizer, path):
    """加载模型检查点"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {path} at epoch {epoch}")
    return epoch