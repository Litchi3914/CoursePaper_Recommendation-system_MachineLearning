import torch
import torch.nn as nn
import numpy as np
import math
from collections import defaultdict

class CountSketch:
    def __init__(self, d, m):
        """
        d: 原始向量维度
        m: 压缩后的维度
        """
        self.d = d
        self.m = m
        self.h = np.random.randint(0, m, size=d)  # 哈希函数
        self.s = np.random.choice([-1, 1], size=d)  # 符号函数
        
    def sketch(self, x):
        """
        将向量x压缩到m维
        """
        result = np.zeros(self.m)
        for i in range(self.d):
            result[self.h[i]] += self.s[i] * x[i]
        return result
    
    def unsketch(self, y):
        """
        从压缩向量y恢复原始向量
        """
        result = np.zeros(self.d)
        for i in range(self.d):
            result[i] = self.s[i] * y[self.h[i]]
        return result

class AdamCountSketch(nn.Module):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, sketch_dim=1000):
        super(AdamCountSketch, self).__init__()
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.sketch_dim = sketch_dim
        
        # 为每个参数创建CountSketch
        self.sketches = {}
        for p in self.params:
            if p.requires_grad:
                self.sketches[p] = CountSketch(p.numel(), sketch_dim)
        
        # 初始化状态
        self.state = defaultdict(dict)
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
            
        for p in self.params:
            if p.grad is None:
                continue
                
            grad = p.grad.data
            if grad.is_sparse:
                raise RuntimeError('AdamCountSketch does not support sparse gradients')
                
            state = self.state[p]
            
            # 初始化状态
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                if self.amsgrad:
                    state['max_exp_avg_sq'] = torch.zeros_like(p.data)
            
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            if self.amsgrad:
                max_exp_avg_sq = state['max_exp_avg_sq']
            beta1, beta2 = self.betas
            
            state['step'] += 1
            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']
            
            if self.weight_decay != 0:
                grad = grad.add(p.data, alpha=self.weight_decay)
            
            # 使用CountSketch压缩梯度
            grad_np = grad.cpu().numpy().flatten()
            sketch = self.sketches[p].sketch(grad_np)
            grad_sketched = torch.from_numpy(sketch).to(grad.device)
            
            # 将压缩后的梯度恢复为原始形状
            grad_restored = torch.from_numpy(self.sketches[p].unsketch(sketch)).to(grad.device)
            grad_restored = grad_restored.view_as(grad)
            
            # 更新一阶矩估计
            exp_avg.mul_(beta1).add_(grad_restored, alpha=1 - beta1)
            
            # 更新二阶矩估计
            exp_avg_sq.mul_(beta2).addcmul_(grad_restored, grad_restored, value=1 - beta2)
            
            if self.amsgrad:
                torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(self.eps)
            else:
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(self.eps)
            
            step_size = self.lr / bias_correction1
            
            # 更新参数
            p.data.addcdiv_(exp_avg, denom, value=-step_size)
            
        return loss 