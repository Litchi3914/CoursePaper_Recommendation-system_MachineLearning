import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F

# 将AmazonDataset定义在模块级别而不是方法内部
class AmazonDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, feature_dims):
        self.X = X
        self.y = y
        self.feature_dims = feature_dims
        
        # 计算特征偏移量
        self.feature_offsets = {}
        offset = 0
        for name, dim in feature_dims.items():
            self.feature_offsets[name] = offset
            offset += dim
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x_row = self.X.iloc[idx]
        
        # 创建一个特征向量，存储4个特征而不是整个特征空间
        features = torch.zeros(4)
        
        # 确保索引在有效范围内
        user_id = min(int(x_row['user_id']), self.feature_dims['user_dim'] - 1)
        item_id = min(int(x_row['item_id']), self.feature_dims['item_dim'] - 1)
        hour = min(int(x_row['hour']), self.feature_dims['hour_dim'] - 1)
        day = min(int(x_row['day_of_week']), self.feature_dims['day_dim'] - 1)
        
        # 用户ID
        features[0] = user_id
        
        # 物品ID - 需要加上用户偏移量
        features[1] = item_id + self.feature_dims['user_dim']
        
        # 小时 - 需要加上用户和物品偏移量
        features[2] = hour + self.feature_dims['user_dim'] + self.feature_dims['item_dim']
        
        # 星期几 - 需要加上用户、物品和小时偏移量
        features[3] = day + self.feature_dims['user_dim'] + self.feature_dims['item_dim'] + self.feature_dims['hour_dim']
        
        return features, torch.tensor(self.y.iloc[idx], dtype=torch.float)

class AmazonMagazineProcessor:
    def __init__(self, data_path, min_item_freq=5):
        self.data_path = data_path
        self.min_item_freq = min_item_freq
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
    def load_data(self):
        """加载Amazon Magazine数据集"""
        ratings = pd.read_csv(
            f"{self.data_path}/Magazine_Subscriptions.csv",
            header=None,
            names=['item_id', 'user_id', 'rating', 'timestamp']
        )
        return ratings
    
    def filter_items(self, ratings):
        """过滤低频物品"""
        # 统计每个物品的出现次数
        item_counts = ratings['item_id'].value_counts()
        
        # 保留出现次数>=min_item_freq的物品
        valid_items = item_counts[item_counts >= self.min_item_freq].index
        
        # 过滤数据
        filtered_ratings = ratings[ratings['item_id'].isin(valid_items)].copy()  # 使用.copy()避免SettingWithCopyWarning
        
        print(f"原始数据: {len(ratings)}条交互")
        print(f"过滤后数据: {len(filtered_ratings)}条交互")
        print(f"用户数: {filtered_ratings['user_id'].nunique()}")
        print(f"杂志数: {filtered_ratings['item_id'].nunique()}")
        
        return filtered_ratings
    
    def preprocess(self, ratings):
        """数据预处理"""
        # 1. 过滤低频物品
        ratings = self.filter_items(ratings)
        
        # 2. 将评分转换为二值标签（评分>=4为正样本）
        ratings['label'] = (ratings['rating'] >= 4).astype(int)
        
        # 3. 编码用户ID和杂志ID
        ratings['user_id'] = self.user_encoder.fit_transform(ratings['user_id'])
        ratings['item_id'] = self.item_encoder.fit_transform(ratings['item_id'])
        
        # 4. 添加时间特征
        ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
        ratings['hour'] = ratings['timestamp'].dt.hour
        ratings['day_of_week'] = ratings['timestamp'].dt.dayofweek
        
        # 5. 选择最终特征
        features = ['user_id', 'item_id', 'hour', 'day_of_week']
        target = 'label'
        
        return ratings[features], ratings[target]
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1):
        """划分数据集"""
        # 首先划分出测试集
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y)
        
        # 从剩余数据中划分出验证集
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted, 
            random_state=42, stratify=y_train_val)
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def get_feature_dims(self):
        """获取特征维度"""
        return {
            'user_dim': len(self.user_encoder.classes_),
            'item_dim': len(self.item_encoder.classes_),
            'hour_dim': 24,  # 0-23小时
            'day_dim': 7     # 0-6 一周七天
        }
    
    def process(self):
        """完整的数据处理流程"""
        # 加载数据
        ratings = self.load_data()
        
        # 预处理
        X, y = self.preprocess(ratings)
        
        # 划分数据集
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.split_data(X, y)
        
        # 构建特征维度字典
        feature_dims = self.get_feature_dims()
        
        # 创建数据集
        train_dataset = AmazonDataset(X_train, y_train, feature_dims)
        val_dataset = AmazonDataset(X_val, y_val, feature_dims)
        test_dataset = AmazonDataset(X_test, y_test, feature_dims)
        
        print(f"训练集样本数: {len(train_dataset)}")
        print(f"验证集样本数: {len(val_dataset)}")
        print(f"测试集样本数: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset, feature_dims 