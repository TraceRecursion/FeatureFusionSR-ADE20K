# ADE20K Dataset Preparation

1. **下载 ADE20K 数据集**：
   - 访问 [MIT CSAIL ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/)
   - 下载 `ADE20K_2021_17_01` 版本（例如 `ADE20K_2021_17_01.zip`）
2. **解压数据集**：
   - 解压至 `/Users/sydg/Documents/数据集/ADE20K_2021_17_01`，确保目录结构如下：
     ```
     /Users/sydg/Documents/数据集/ADE20K_2021_17_01/
     ├── images/
     │   └── ADE/
     │       ├── training/
     │       └── validation/
     ├── index_ade20k.mat
     ├── index_ade20k.pkl
     └── objects.txt
     ```
3. **配置路径**：
   - 在 `config.py` 中更新 `DATA_ROOT` 为实际路径，例如 `/home/sydg/ADE20K_2021_17_01`。