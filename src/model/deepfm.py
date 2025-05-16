import torch
import torch.nn as nn
import torch.nn.functional as F

class FM(nn.Module):
    def __init__(self, feature_dims, embedding_size):
        super(FM, self).__init__()
        self.feature_dims = feature_dims
        self.embedding_size = embedding_size
        
        # 计算每个特征的起始索引
        self.feature_offsets = {}
        offset = 0
        for name, dim in feature_dims.items():
            self.feature_offsets[name] = offset
            offset += dim
        
        # 一阶特征
        self.linear = nn.Linear(sum(feature_dims.values()), 1)
        
        # 二阶特征
        self.v = nn.Parameter(torch.randn(sum(feature_dims.values()), embedding_size))
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.v)
    
    def forward(self, x):
        # 确保输入维度正确
        batch_size = x.size(0)
        
        # 判断输入是独热编码还是索引形式
        if x.size(1) == sum(self.feature_dims.values()):
            # 独热编码形式 - MovieLens
            x = x.float()
            
            # 一阶特征
            linear_part = self.linear(x)
            
            # 二阶特征交互
            # 计算加权平均嵌入
            embeddings = []
            for name, dim in self.feature_dims.items():
                start_idx = self.feature_offsets[name]
                feature = x[:, start_idx:start_idx + dim]
                embedding = torch.matmul(feature, self.v[start_idx:start_idx + dim])
                embeddings.append(embedding)
            
            # 计算二阶交互
            square_of_sum = torch.pow(torch.sum(torch.stack(embeddings), dim=0), 2)
            sum_of_square = torch.sum(torch.stack([torch.pow(emb, 2) for emb in embeddings]), dim=0)
            interaction_part = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)
            
            return linear_part + interaction_part
        else:
            # 索引形式 - Amazon
            # 将索引转换为独热编码的等效操作
            feature_indices = x.long()  # 确保索引是整数类型
            
            # 一阶特征（使用gather操作从linear.weight中获取对应权重）
            linear_weights = self.linear.weight.squeeze()
            linear_bias = self.linear.bias
            linear_part = torch.zeros(batch_size, 1, device=x.device)
            
            for i in range(feature_indices.size(1)):
                indices = feature_indices[:, i]
                linear_part += linear_weights[indices].unsqueeze(1)
            
            linear_part += linear_bias
            
            # 二阶特征交互（使用embedding look-up）
            embeddings = []
            for i in range(feature_indices.size(1)):
                indices = feature_indices[:, i]
                embedding = self.v[indices]  # [batch_size, embedding_size]
                embeddings.append(embedding)
            
            # 计算二阶交互
            embeddings_sum = torch.stack(embeddings, dim=0).sum(dim=0)  # [batch_size, embedding_size]
            square_of_sum = torch.pow(embeddings_sum, 2)  # [batch_size, embedding_size]
            
            embeddings_square = torch.stack([torch.pow(emb, 2) for emb in embeddings], dim=0)  # [num_fields, batch_size, embedding_size]
            sum_of_square = embeddings_square.sum(dim=0)  # [batch_size, embedding_size]
            
            interaction_part = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)
            
            return linear_part + interaction_part

class DeepFM(nn.Module):
    def __init__(self, feature_dims, embedding_size, hidden_dims, dropout_deep, dropout_fm):
        super(DeepFM, self).__init__()
        self.feature_dims = feature_dims
        self.embedding_size = embedding_size
        
        # 计算每个特征的起始索引
        self.feature_offsets = {}
        offset = 0
        for name, dim in feature_dims.items():
            self.feature_offsets[name] = offset
            offset += dim
        
        # 打印模型初始化信息
        print("\n模型初始化信息:")
        print(f"特征维度: {feature_dims}")
        print(f"总维度: {sum(feature_dims.values())}")
        
        # FM部分
        self.fm = FM(feature_dims, embedding_size)
        
        # Deep部分
        self.embedding = nn.Embedding(sum(feature_dims.values()), embedding_size)
        
        # 计算deep部分的输入维度
        deep_input_dim = embedding_size * len(feature_dims)
        
        # Deep网络
        self.deep_layers = nn.ModuleList()
        input_dim = deep_input_dim
        for hidden_dim in hidden_dims:
            self.deep_layers.append(nn.Linear(input_dim, hidden_dim))
            self.deep_layers.append(nn.ReLU())
            self.deep_layers.append(nn.Dropout(dropout_deep[0]))
            input_dim = hidden_dim
        
        # 输出层
        self.output = nn.Linear(hidden_dims[-1] + 1, 1)  # +1 是因为FM部分的输出是1维的
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        for layer in self.deep_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
    
    def forward(self, x):
        # 确保输入维度正确
        batch_size = x.size(0)
        
        # 判断输入是独热编码还是索引形式
        is_amazon_format = (x.size(1) == 4)  # Amazon格式只有4个特征索引
        
        if not is_amazon_format and x.size(1) != sum(self.feature_dims.values()):
            raise ValueError(f"输入特征维度不匹配: 期望 {sum(self.feature_dims.values())} 或 4, 实际 {x.size(1)}")
        
        # FM部分
        fm_output = self.fm(x)
        
        # Deep部分
        if is_amazon_format:
            # Amazon格式 - 索引形式
            # 获取所有特征的嵌入
            embeddings = []
            for i in range(x.size(1)):
                indices = x[:, i].long()  # 转为长整型用于索引
                embedding = self.embedding(indices)  # [batch_size, embedding_size]
                embeddings.append(embedding)
            
            deep_input = torch.cat(embeddings, dim=1)
        else:
            # MovieLens格式 - 独热编码形式
            embeddings = []
            for name, dim in self.feature_dims.items():
                start_idx = self.feature_offsets[name]
                feature = x[:, start_idx:start_idx + dim]
                # 计算加权平均嵌入
                embedding = torch.matmul(feature, self.embedding.weight[start_idx:start_idx + dim])
                embeddings.append(embedding)
            
            deep_input = torch.cat(embeddings, dim=1)
        
        # 通过deep网络
        deep_output = deep_input
        for layer in self.deep_layers:
            deep_output = layer(deep_output)
        
        # 合并FM和Deep的输出
        concat_input = torch.cat([fm_output, deep_output], dim=1)
        output = self.output(concat_input)
        
        # 使用sigmoid确保输出在[0,1]范围内
        return torch.sigmoid(output)
    
    def predict(self, x):
        return self.forward(x) 