# ADE20K Dataset Preparation

1. **下载 ADE20K 数据集**：
   - 访问 [MIT CSAIL ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/)
   - 下载 `ADE20K_2021_17_01` 版本（例如 `ADE20K_2021_17_01.zip`）
2. **解压数据集**：
   - 默认情况下，系统会在当前用户的 `~/Documents/数据集/ADE20K_2021_17_01` 目录下查找数据集
   - 确保目录结构如下：
     ```
     ~/Documents/数据集/ADE20K_2021_17_01/
     ├── images/
     │   └── ADE/
     │       ├── training/
     │       └── validation/
     ├── index_ade20k.mat
     ├── index_ade20k.pkl
     └── objects.txt
     ```
3. **配置数据集路径**：
   如果您的数据集不在默认位置，有以下方法指定路径：
   
   a) **环境变量**：设置 `ADE20K_DATASET_PATH` 环境变量
      ```bash
      # Linux/macOS
      export ADE20K_DATASET_PATH="/path/to/your/ADE20K_2021_17_01"
      
      # Windows
      set ADE20K_DATASET_PATH=C:\path\to\your\ADE20K_2021_17_01
      ```
      
   b) **代码方式**：在运行时使用 `Config.set_data_root()` 方法
      ```python
      from config import Config
      Config.set_data_root("/path/to/your/ADE20K_2021_17_01")
      ```