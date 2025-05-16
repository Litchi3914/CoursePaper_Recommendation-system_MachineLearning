import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, feature_dims):
        self.X = X
        self.y = y
        self.feature_dims = feature_dims
        
        # 计算每个特征的起始索引
        self.feature_offsets = {}
        offset = 0
        for name, dim in feature_dims.items():
            self.feature_offsets[name] = offset
            offset += dim
        
        # 打印数据集初始化信息
        print(f"\n数据集特征维度: {sum(feature_dims.values())}")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # 获取当前样本的特征和标签
        features = self.X.iloc[idx]
        movie_id = self.y.iloc[idx]['movie_id']
        label = self.y.iloc[idx]['label']
        
        # 构建特征向量，使用float32类型
        feature_vector = torch.zeros(sum(self.feature_dims.values()), dtype=torch.float32)
        # 用户ID
        feature_vector[self.feature_offsets['user_id'] + int(features['user_id'])] = 1
        # 年龄分桶
        feature_vector[self.feature_offsets['age_bucket'] + int(features['age_bucket'])] = 1
        # 性别
        feature_vector[self.feature_offsets['gender'] + int(features['gender'])] = 1
        # 电影ID
        feature_vector[self.feature_offsets['movie_id'] + movie_id] = 1
        
        # 如果是正样本，返回正样本特征和目标
        if label == 1:
            return {
                'pos_features': feature_vector,
                'pos_target': torch.tensor(1.0, dtype=torch.float32),
                'neg_features': torch.zeros_like(feature_vector),
                'neg_target': torch.tensor(0.0, dtype=torch.float32)
            }
        # 如果是负样本，返回负样本特征和目标
        else:
            return {
                'pos_features': torch.zeros_like(feature_vector),
                'pos_target': torch.tensor(1.0, dtype=torch.float32),
                'neg_features': feature_vector,
                'neg_target': torch.tensor(0.0, dtype=torch.float32)
            }

class MovieLensProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.feature_dims = None  # 将在预处理后设置
        
    def load_data(self):
        """加载MovieLens-1M数据集"""
        # 加载评分数据
        ratings = pd.read_csv(f"{self.data_path}/ratings.dat", 
                            sep='::', 
                            names=['user_id', 'movie_id', 'rating', 'timestamp'],
                            engine='python')
        
        # 加载用户数据
        users = pd.read_csv(f"{self.data_path}/users.dat",
                          sep='::',
                          names=['user_id', 'gender', 'age', 'occupation', 'zip_code'],
                          engine='python')
        
        # 加载电影数据
        movies = pd.read_csv(f"{self.data_path}/movies.dat",
                           sep='::',
                           names=['movie_id', 'title', 'genres'],
                           engine='python',
                           encoding='latin-1')
        
        return ratings, users, movies
    
    def preprocess(self, ratings, users, movies):
        """数据预处理"""
        # 1. 处理评分数据
        # 将评分转换为二值标签（评分>=3为正样本）
        ratings['label'] = (ratings['rating'] >= 3).astype(int)
        
        # 2. 处理用户特征
        # 年龄分桶（3个区间，减少维度）
        users['age_bucket'] = pd.qcut(users['age'], q=3, labels=False, duplicates='drop')
        
        # 性别编码
        users['gender'] = users['gender'].map({'M': 1, 'F': 0})
        
        # 3. 处理电影特征
        # 提取电影年份
        movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)
        movies['year_bucket'] = pd.qcut(movies['year'], q=3, labels=False, duplicates='drop')
        
        # 4. 合并特征
        data = ratings.merge(users[['user_id', 'age_bucket', 'gender']], on='user_id')
        data = data.merge(movies[['movie_id', 'year_bucket']], on='movie_id')
        
        # 5. 编码用户ID和电影ID
        data['user_id'] = self.user_encoder.fit_transform(data['user_id'])
        data['movie_id'] = self.item_encoder.fit_transform(data['movie_id'])
        
        # 打印编码后的唯一值数量
        print(f"用户数量: {len(self.user_encoder.classes_)}")
        print(f"电影数量: {len(self.item_encoder.classes_)}")
        print(f"总样本数: {len(data)}")
        
        # 6. 添加时间特征
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
        data['hour'] = data['timestamp'].dt.hour // 4  # 将24小时分为6个区间，减少维度
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        
        # 7. 为每个用户生成负样本
        # 首先获取每个用户已评分的电影
        user_items = data.groupby('user_id')['movie_id'].apply(set).to_dict()
        
        # 为每个用户生成固定数量的负样本
        negative_samples = []
        all_items = set(range(len(self.item_encoder.classes_)))
        
        # 对每个用户生成负样本
        for user_id, rated_items in user_items.items():
            # 获取用户未评分的电影
            negative_items = list(all_items - rated_items)
            if negative_items:
                # 为每个用户生成最多2个负样本，减少负样本数量
                n_negative = min(2, len(negative_items))
                neg_items = np.random.choice(negative_items, n_negative, replace=False)
                
                # 获取用户特征
                user_data = data[data['user_id'] == user_id].iloc[0]
                
                # 生成负样本
                for neg_movie in neg_items:
                    negative_samples.append({
                        'user_id': user_id,
                        'movie_id': neg_movie,
                        'label': 0,
                        'age_bucket': user_data['age_bucket'],
                        'gender': user_data['gender'],
                        'hour': user_data['hour'],
                        'day_of_week': user_data['day_of_week']
                    })
        
        # 将负样本添加到数据集中
        negative_df = pd.DataFrame(negative_samples)
        data = pd.concat([data, negative_df], ignore_index=True)
        
        # 8. 选择最终特征
        features = ['user_id', 'age_bucket', 'gender']
        target = 'label'
        
        # 确保所有特征都是数值类型
        for feature in features:
            data[feature] = data[feature].astype(float)
        
        # 将特征标准化
        for feature in features[1:]:  # 跳过 user_id
            data[feature] = (data[feature] - data[feature].mean()) / data[feature].std()
        
        # 设置特征维度
        self.feature_dims = {
            'user_id': len(self.user_encoder.classes_),
            'age_bucket': 3,  # 减少年龄分桶数
            'gender': 2,      # 性别编码数
            'movie_id': len(self.item_encoder.classes_),
            'hour': 6,        # 减少小时数（4小时一个区间）
            'day_of_week': 7  # 星期数
        }
        
        # 打印特征维度信息
        print("\n特征维度信息:")
        for name, dim in self.feature_dims.items():
            print(f"{name}: {dim}")
        print(f"总维度: {sum(self.feature_dims.values())}")
        
        return data[features], data[['movie_id', 'label', 'hour', 'day_of_week']]
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1):
        """划分数据集"""
        # 首先划分出测试集
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42)
        
        # 从剩余数据中划分出验证集
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted, 
            random_state=42)
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def get_feature_dims(self):
        """获取特征维度"""
        if self.feature_dims is None:
            raise ValueError("请先调用 preprocess 方法处理数据")
        return self.feature_dims
    
    def process(self):
        """完整的数据处理流程"""
        # 加载数据
        ratings, users, movies = self.load_data()
        
        # 预处理
        X, y = self.preprocess(ratings, users, movies)
        
        # 划分数据集
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.split_data(X, y)
        
        # 获取特征维度
        feature_dims = self.get_feature_dims()
        
        # 转换为自定义数据集
        train_dataset = CustomDataset(X_train, y_train, feature_dims)
        val_dataset = CustomDataset(X_val, y_val, feature_dims)
        test_dataset = CustomDataset(X_test, y_test, feature_dims)
        
        return train_dataset, val_dataset, test_dataset, feature_dims 