import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from contextlib import nullcontext

from src.data_processing.movie_lens_processor import MovieLensProcessor
from src.data_processing.amazon_magazine_processor import AmazonMagazineProcessor
from src.model.deepfm import DeepFM
from src.optimizers.base_optimizers import BaseOptimizer, CustomSGD, CustomRMSprop, CustomAdam
from src.optimizers.count_sketch_optimizer import AdamCountSketch

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 初始化混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        
    def load_data(self):
        """加载数据集"""
        if self.config['dataset'] == 'movielens':
            processor = MovieLensProcessor(self.config['data_path'])
        else:  # amazon
            processor = AmazonMagazineProcessor(self.config['data_path'])
            
        train, val, test, feature_dims = processor.process()
        return train, val, test, feature_dims
    
    def create_model(self, feature_dims):
        """创建模型"""
        model = DeepFM(
            feature_dims=feature_dims,
            embedding_size=self.config['embedding_size'],
            hidden_dims=self.config['hidden_dims'],
            dropout_deep=self.config['dropout_deep'],
            dropout_fm=self.config['dropout_fm']
        ).to(self.device)
        return model
    
    def create_optimizer(self, model):
        """创建优化器"""
        if self.config['optimizer'] == 'sgd':
            return CustomSGD(model.parameters(), lr=self.config['lr'])
        elif self.config['optimizer'] == 'rmsprop':
            return CustomRMSprop(model.parameters(), lr=self.config['lr'])
        elif self.config['optimizer'] == 'adam':
            return CustomAdam(model.parameters(), lr=self.config['lr'])
        elif self.config['optimizer'] == 'adam_countsketch':
            return AdamCountSketch(model.parameters(), lr=self.config['lr'], 
                                 sketch_dim=self.config['sketch_dim'])
        else:
            raise ValueError(f"Unsupported optimizer: {self.config['optimizer']}")
    
    def calculate_metrics(self, predictions, targets, k=10):
        """计算评估指标"""
        # 将预测值和目标值转换为numpy数组
        preds = predictions.cpu().numpy()
        targets = targets.cpu().numpy()
        
        # 将预测值和目标值分成正样本和负样本
        pos_preds = preds[::2]  # 偶数索引是正样本
        neg_preds = preds[1::2]  # 奇数索引是负样本
        pos_targets = targets[::2]
        neg_targets = targets[1::2]
        
        # 计算HR@K
        # 对每个用户，检查正样本的预测值是否大于负样本
        hr = np.mean(pos_preds > neg_preds)
        
        # 计算NDCG@K
        def dcg_at_k(r, k):
            r = np.asarray(r)[:k]
            if r.size:
                return np.sum((2**r - 1) / np.log2(np.arange(2, r.size + 2)))
            return 0.
        
        def ndcg_at_k(r, k):
            dcg_max = dcg_at_k(sorted(r, reverse=True), k)
            if not dcg_max:
                return 0.
            return dcg_at_k(r, k) / dcg_max
        
        # 对每个用户计算NDCG@K
        ndcg = np.mean([ndcg_at_k([1, 0], k) if pos_preds[i] > neg_preds[i] else ndcg_at_k([0, 1], k) 
                        for i in range(len(pos_preds))])
        
        return {'HR@10': hr, 'NDCG@10': ndcg}
    
    def train_epoch_movielens(self, model, optimizer, train_loader, criterion):
        """MovieLens数据集专用的训练一个epoch的方法"""
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc='Training MovieLens'):
            pos_features = batch['pos_features'].to(self.device)
            neg_features = batch['neg_features'].to(self.device)
            pos_target = batch['pos_target'].to(self.device)
            neg_target = batch['neg_target'].to(self.device)
            
            with torch.cuda.amp.autocast() if torch.cuda.is_available() else nullcontext():
                pos_pred = model(pos_features)
                neg_pred = model(neg_features)
                pos_loss = criterion(pos_pred, pos_target.unsqueeze(1))
                neg_loss = criterion(neg_pred, neg_target.unsqueeze(1))
                loss = pos_loss + neg_loss
                
            optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def train_epoch_amazon(self, model, optimizer, train_loader, criterion):
        """Amazon Magazine数据集专用的训练一个epoch的方法"""
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc='Training Amazon'):
            features, target = batch
            features = features.to(self.device)
            target = target.to(self.device)
            
            with torch.cuda.amp.autocast() if torch.cuda.is_available() else nullcontext():
                pred = model(features)
                loss = criterion(pred, target.unsqueeze(1))
                
            optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def evaluate_movielens(self, model, data_loader, criterion):
        """MovieLens数据集专用的评估方法"""
        model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Evaluating MovieLens'):
                pos_features = batch['pos_features'].to(self.device)
                neg_features = batch['neg_features'].to(self.device)
                pos_target = batch['pos_target'].to(self.device)
                neg_target = batch['neg_target'].to(self.device)
                
                pos_pred = model(pos_features)
                neg_pred = model(neg_features)
                pos_loss = criterion(pos_pred, pos_target.unsqueeze(1))
                neg_loss = criterion(neg_pred, neg_target.unsqueeze(1))
                loss = pos_loss + neg_loss
                
                all_predictions.extend([pos_pred, neg_pred])
                all_targets.extend([pos_target, neg_target])
                total_loss += loss.item()
                
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        metrics = self.calculate_metrics(all_predictions, all_targets)
        metrics['loss'] = total_loss / len(data_loader)
        return metrics
    
    def evaluate_amazon(self, model, data_loader, criterion):
        """Amazon Magazine数据集专用的评估方法"""
        model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Evaluating Amazon'):
                features, target = batch
                features = features.to(self.device)
                target = target.to(self.device)
                
                pred = model(features)
                loss = criterion(pred, target.unsqueeze(1))
                
                all_predictions.append(pred)
                all_targets.append(target)
                total_loss += loss.item()
                
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        metrics = self.calculate_metrics(all_predictions, all_targets)
        metrics['loss'] = total_loss / len(data_loader)
        return metrics
    
    def train_movielens(self):
        """MovieLens数据集专用的完整训练流程"""
        # 加载数据
        train, val, test, feature_dims = self.load_data()
        
        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(
            train, 
            batch_size=self.config['batch_size'], 
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
            prefetch_factor=2
        )
        val_loader = torch.utils.data.DataLoader(
            val, 
            batch_size=self.config['batch_size'] * 2,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True
        )
        test_loader = torch.utils.data.DataLoader(
            test, 
            batch_size=self.config['batch_size'] * 2,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True
        )
        
        # 创建模型和优化器
        model = self.create_model(feature_dims)
        optimizer = self.create_optimizer(model)
        criterion = nn.BCEWithLogitsLoss()
        
        # 训练循环
        best_val_metrics = {'HR@10': 0, 'NDCG@10': 0}
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            
            # 训练
            train_loss = self.train_epoch_movielens(model, optimizer, train_loader, criterion)
            print(f"Train Loss: {train_loss:.4f}")
            
            # 验证
            val_metrics = self.evaluate_movielens(model, val_loader, criterion)
            print(f"Validation Metrics: {val_metrics}")
            
            # 保存最佳模型
            if val_metrics['HR@10'] > best_val_metrics['HR@10']:
                best_val_metrics = val_metrics
                torch.save(model.state_dict(), 
                          os.path.join(self.config['save_path'], 'best_model_movielens.pth'))
        
        # 在测试集上评估最佳模型
        model.load_state_dict(torch.load(
            os.path.join(self.config['save_path'], 'best_model_movielens.pth')))
        test_metrics = self.evaluate_movielens(model, test_loader, criterion)
        print(f"\nTest Metrics: {test_metrics}")
        
        return test_metrics
    
    def train_amazon(self):
        """Amazon Magazine数据集专用的完整训练流程"""
        # 加载数据
        train, val, test, feature_dims = self.load_data()
        
        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(
            train, 
            batch_size=self.config['batch_size'], 
            shuffle=True,
            num_workers=0,  # 避免多进程问题
            pin_memory=torch.cuda.is_available()
        )
        val_loader = torch.utils.data.DataLoader(
            val, 
            batch_size=self.config['batch_size'] * 2,
            num_workers=0,  # 避免多进程问题
            pin_memory=torch.cuda.is_available()
        )
        test_loader = torch.utils.data.DataLoader(
            test, 
            batch_size=self.config['batch_size'] * 2,
            num_workers=0,  # 避免多进程问题
            pin_memory=torch.cuda.is_available()
        )
        
        # 创建模型和优化器
        model = self.create_model(feature_dims)
        optimizer = self.create_optimizer(model)
        criterion = nn.BCEWithLogitsLoss()
        
        # 训练循环
        best_val_metrics = {'HR@10': 0, 'NDCG@10': 0}
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            
            # 训练
            train_loss = self.train_epoch_amazon(model, optimizer, train_loader, criterion)
            print(f"Train Loss: {train_loss:.4f}")
            
            # 验证
            val_metrics = self.evaluate_amazon(model, val_loader, criterion)
            print(f"Validation Metrics: {val_metrics}")
            
            # 保存最佳模型
            if val_metrics['HR@10'] > best_val_metrics['HR@10']:
                best_val_metrics = val_metrics
                torch.save(model.state_dict(), 
                          os.path.join(self.config['save_path'], 'best_model_amazon.pth'))
        
        # 在测试集上评估最佳模型
        model.load_state_dict(torch.load(
            os.path.join(self.config['save_path'], 'best_model_amazon.pth')))
        test_metrics = self.evaluate_amazon(model, test_loader, criterion)
        print(f"\nTest Metrics: {test_metrics}")
        
        return test_metrics
    
    def train(self):
        """根据配置选择对应的训练流程"""
        if self.config['dataset'] == 'movielens':
            return self.train_movielens()
        else:  # amazon
            return self.train_amazon()

def main(dataset_name=None):
    # 配置参数
    config = {
        'dataset': dataset_name or 'movielens',  # 可以通过参数指定数据集
        'data_path': 'data',  # 修改为当前目录下的data目录
        'save_path': 'results',  # 修改为当前目录下的results目录
        'embedding_size': 4,  # 减小embedding维度
        'hidden_dims': [128, 128, 128],  # 减小隐藏层维度
        'dropout_deep': [0.3, 0.3, 0.3],  # 减小dropout
        'dropout_fm': [0.3, 0.3],  # 减小dropout
        'batch_size': 2048,  # 增加batch size
        'epochs': 50,
        'lr': 0.001,
        'optimizer': 'adam_countsketch',
        'sketch_dim': 1000}
    
    print(f"训练数据集: {config['dataset']}")
    
    # 创建保存目录
    os.makedirs(config['save_path'], exist_ok=True)
    
    # 训练模型
    trainer = Trainer(config)
    metrics = trainer.train()
    
    # 保存结果
    results = {
        'dataset': config['dataset'],
        'optimizer': config['optimizer'],
        'metrics': metrics,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv(os.path.join(config['save_path'], 'results.csv'), 
                     mode='a', header=not os.path.exists(
                         os.path.join(config['save_path'], 'results.csv')))

if __name__ == '__main__':
    # 可以在这里指定要使用的数据集
    # main()  # 使用默认的movielens数据集
    main('amazon_magazine')  # 使用amazon_magazine数据集 