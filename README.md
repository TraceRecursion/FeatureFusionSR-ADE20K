# FeatureFusionSR-ADE20K

一个基于PyTorch实现的特征融合驱动的图像超分辨率方法，适用于ADE20K数据集。该项目作为本科毕业论文设计，利用SegFormer-B5模型进行语义特征提取，并集成多尺度低级特征，从低分辨率（LR）输入重建高分辨率（HR）图像。该方法增强了像素级细节恢复和语义一致性，在复杂场景中实现了卓越的性能。

## 特点
- **语义特征提取**：利用在ADE20K上预训练的SegFormer-B5进行像素级语义分割。
- **特征融合**：使用轻量级融合模块结合CBAM，将语义特征与多尺度低级特征融合。
- **超分辨率**：采用增强的基于EDSR的网络和PixelShuffle上采样进行4倍HR重建。
- **模块化设计**：干净、可扩展的代码库，适合实验和进一步研究。

## 动机
传统的超分辨率方法由于其对全局特征的关注，在处理语义丰富的图像时表现不佳。该项目通过将SegFormer-B5的语义先验与低级细节相结合，提供了一种高质量图像重建的新方法。

## 数据集
- **ADE20K**：一个包含150个类别的语义分割数据集，提供多样的室内和室外场景，用于稳健的训练和评估。

## 要求
- Python 3.8+
- PyTorch 2.0+
- torchvision
- timm
- numpy
- opencv-python

## 项目架构

```shell
FeatureFusionSR-ADE20K/
├── data/
│   ├── ade20k_loader.py       # 数据加载和预处理
│   └── README.md             # 数据集下载和准备说明
│
├── models/
│   ├── segformer.py          # SegFormer-B5 特征提取模块
│   ├── feature_fusion.py     # 特征融合模块（含 CBAM）
│   ├── edsr.py               # Enhanced EDSR 重建网络
│   └── sr_model.py           # 完整超分辨率模型
│
├── utils/
│   ├── metrics.py            # PSNR 和 SSIM 计算
│   └── training_utils.py     # 训练辅助函数
│
├── train.py                  # 训练脚本
├── test.py                   # 测试脚本
├── config.py                 # 配置文件
├── requirements.txt          # 依赖文件
└── README.md                 # 项目介绍
```

## 入门指南
1. 克隆仓库：
   ```bash
   git clone https://github.com/[YourGitHubUsername]/FeatureFusionSR-ADE20K.git
   cd FeatureFusionSR-ADE20K
   ```

2. 安装依赖：
    ```bash
    pip install -r requirements.txt
    ```

3. 下载ADE20K数据集并进行预处理（参见data/README.md）。

## 许可证
MIT许可证
