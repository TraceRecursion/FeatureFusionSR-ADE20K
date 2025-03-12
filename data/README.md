# ADE20K Dataset Preparation

1. **下载 ADE20K 数据集**：
   - 访问 [MIT CSAIL ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/)
   - 下载数据集文件（例如 `ADEChallengeData2016.zip`）
2. **解压数据集**：
   - 解压至 `/path/to/ADE20K`，确保目录结构如下：
     ```
     /path/to/ADE20K/
     ├── images/
     │   ├── training/
     │   └── validation/
     ├── annotations/
     │   ├── training/
     │   └── validation/
     ```
3. **配置路径**：
   - 在 `config.py` 中更新 `DATA_ROOT` 为实际路径，例如 `/home/user/ADE20K`。