# 深度学习推荐系统代码调试指南

## 1. 项目概述

本项目实现了一个基于深度学习的推荐系统，核心是DeepFM模型，用于电影和杂志推荐任务。项目支持两个数据集：MovieLens-1M和Amazon杂志订阅数据集，并提供了多种优化器实现，包括一种基于Count-Sketch技术的内存优化版Adam优化器。

## 2. 文件结构及功能

### 数据目录 (data/)
- **MovieLens 数据集**:
  - `ratings.dat`：用户对电影的评分数据
  - `users.dat`：用户信息，包括性别、年龄等
  - `movies.dat`：电影信息，包括标题、类型等
- **Amazon杂志订阅数据集**:
  - `Magazine_Subscriptions.csv`：亚马逊杂志订阅数据

### 源代码目录 (src/)

#### 主程序 (src/main.py)
- 实现了整个训练流程的`Trainer`类
- 提供了对MovieLens和Amazon两种数据集的训练与评估方法
- 支持多种优化器选择和模型配置
- 包含主函数入口和参数配置

#### 数据处理模块 (src/data_processing/)
- `movie_lens_processor.py`：处理MovieLens数据集
  - 加载数据、特征工程、数据分割和自定义数据集实现
  - 实现了`CustomDataset`类，支持正负样本对的生成
- `amazon_magazine_processor.py`：处理Amazon杂志订阅数据集
  - 提供类似的数据处理功能，但针对不同的数据格式

#### 模型模块 (src/model/)
- `deepfm.py`：实现了DeepFM模型
  - 包含`FM`类实现因子分解机部分
  - 包含`DeepFM`类整合FM和深度学习网络
  - 支持不同数据格式的输入处理（独热编码和索引形式）

#### 优化器模块 (src/optimizers/)
- `base_optimizers.py`：实现了基础优化器
  - `CustomSGD`：随机梯度下降
  - `CustomRMSprop`：RMSprop优化器
  - `CustomAdam`：Adam优化器
- `count_sketch_optimizer.py`：实现了基于Count-Sketch的优化器
  - `CountSketch`类：实现哈希压缩与恢复
  - `AdamCountSketch`类：将Count-Sketch技术应用于Adam优化器，减少内存使用

## 3. 如何使用

### 环境准备
确保安装以下依赖：
```bash
pip install torch numpy pandas tqdm scikit-learn
```

### 运行步骤

1. **准备数据**
   - 确保数据文件位于`data/`目录下
   - MovieLens数据包括：`ratings.dat`, `users.dat`, `movies.dat`
   - Amazon杂志数据为：`Magazine_Subscriptions.csv`

2. **执行训练**
   ```bash
   python src/main.py --dataset movielens
   ```
   或
   ```bash
   python src/main.py --dataset amazon
   ```

3. **参数配置**
   - 可在`main.py`的`main`函数中修改训练参数：
     ```python
     config = {
         'batch_size': 1024,
         'embedding_size': 16, 
         'hidden_dims': [64, 32, 16],
         'dropout_deep': [0.5, 0.5, 0.5],
         'dropout_fm': [0.0, 0.0],
         'lr': 0.001,
         'epochs': 30,
         'optimizer': 'adam',  # 'sgd', 'rmsprop', 'adam', 'adam_countsketch'
         'sketch_dim': 1000,   # for adam_countsketch
         'data_path': 'data',
         'dataset': dataset_name,
     }
     ```