"""
强化学习优化器 (Reinforcement Learning Optimizer)
用于动态调整模型架构和超参数，提升毒性预测性能

作者: AI助手
日期: 2024-06-11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import random
import json
from collections import deque, defaultdict
import math


class ArchitectureSearchSpace:
    """神经架构搜索空间定义"""
    
    def __init__(self):
        # 定义可搜索的架构组件
        self.search_space = {
            'encoder_layers': [2, 3, 4, 5, 6],
            'hidden_dims': [128, 256, 512, 1024, 2048],
            'attention_heads': [4, 8, 12, 16],
            'dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            'activation_function': ['relu', 'gelu', 'swish', 'leaky_relu'],
            'pooling_method': ['mean', 'max', 'attention', 'set2set'],
            'normalization': ['batch_norm', 'layer_norm', 'group_norm', 'none'],
            'optimizer_type': ['adam', 'adamw', 'sgd', 'rmsprop'],
            'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
            'weight_decay': [0.0, 1e-5, 1e-4, 1e-3, 1e-2],
            'warmup_steps': [0, 100, 500, 1000, 2000],
            'scheduler_type': ['cosine', 'linear', 'exponential', 'step']
        }
        
        # 架构约束
        self.constraints = {
            'max_parameters': 50e6,  # 最大参数数量
            'max_memory_gb': 16,     # 最大内存使用
            'min_accuracy': 0.7      # 最小准确率要求
        }
    
    def sample_architecture(self) -> Dict[str, Any]:
        """随机采样一个架构配置"""
        config = {}
        for param_name, choices in self.search_space.items():
            config[param_name] = random.choice(choices)
        return config
    
    def mutate_architecture(self, config: Dict[str, Any], mutation_rate: float = 0.3) -> Dict[str, Any]:
        """对现有架构进行变异"""
        new_config = config.copy()
        for param_name, choices in self.search_space.items():
            if random.random() < mutation_rate:
                new_config[param_name] = random.choice(choices)
        return new_config


class PolicyNetwork(nn.Module):
    """策略网络，用于选择架构参数"""
    
    def __init__(self, 
                 state_dim: int = 128,
                 action_space_sizes: Dict[str, int] = None,
                 hidden_dims: List[int] = [256, 256, 128]):
        super().__init__()
        self.state_dim = state_dim
        
        # 默认动作空间大小
        if action_space_sizes is None:
            action_space_sizes = {
                'encoder_layers': 5,
                'hidden_dims': 5,
                'attention_heads': 4,
                'dropout_rate': 6,
                'activation_function': 4,
                'pooling_method': 4,
                'normalization': 4,
                'optimizer_type': 4,
                'learning_rate': 6,
                'weight_decay': 5,
                'warmup_steps': 5,
                'scheduler_type': 4
            }
        
        self.action_space_sizes = action_space_sizes
        
        # 共享特征提取层
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # 每个参数的策略头
        self.policy_heads = nn.ModuleDict()
        for param_name, action_size in action_space_sizes.items():
            self.policy_heads[param_name] = nn.Sequential(
                nn.Linear(prev_dim, action_size),
                nn.Softmax(dim=-1)
            )
    
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: 当前状态 [batch_size, state_dim]
            
        Returns:
            action_probs: 各参数的动作概率分布
        """
        features = self.feature_extractor(state)
        
        action_probs = {}
        for param_name, head in self.policy_heads.items():
            action_probs[param_name] = head(features)
        
        return action_probs
    
    def sample_actions(self, state: torch.Tensor) -> Tuple[Dict[str, int], Dict[str, torch.Tensor]]:
        """采样动作"""
        action_probs = self.forward(state)
        
        actions = {}
        log_probs = {}
        
        for param_name, probs in action_probs.items():
            # 采样动作
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            actions[param_name] = action.item()
            log_probs[param_name] = log_prob
        
        return actions, log_probs


class ValueNetwork(nn.Module):
    """价值网络，估计状态价值"""
    
    def __init__(self, 
                 state_dim: int = 128,
                 hidden_dims: List[int] = [256, 256, 128]):
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.value_network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 当前状态 [batch_size, state_dim]
            
        Returns:
            value: 状态价值 [batch_size, 1]
        """
        return self.value_network(state)


class StateEncoder:
    """状态编码器，将当前架构和性能编码为状态向量"""
    
    def __init__(self, state_dim: int = 128):
        self.state_dim = state_dim
        self.search_space = ArchitectureSearchSpace()
        
        # 为每个参数创建编码维度
        self.param_encodings = {}
        current_dim = 0
        
        for param_name, choices in self.search_space.search_space.items():
            if isinstance(choices[0], (int, float)):
                # 数值型参数，使用归一化
                self.param_encodings[param_name] = {
                    'type': 'numerical',
                    'min_val': min(choices),
                    'max_val': max(choices),
                    'dim': 1,
                    'start_idx': current_dim
                }
                current_dim += 1
            else:
                # 类别型参数，使用one-hot编码
                self.param_encodings[param_name] = {
                    'type': 'categorical',
                    'choices': choices,
                    'dim': len(choices),
                    'start_idx': current_dim
                }
                current_dim += len(choices)
        
        # 性能指标维度
        self.performance_dim = 5  # accuracy, f1, auc, loss, validation_loss
        current_dim += self.performance_dim
        
        # 历史信息维度
        self.history_dim = 10
        current_dim += self.history_dim
        
        # 确保总维度不超过state_dim
        self.total_encoding_dim = current_dim
        if current_dim > state_dim:
            raise ValueError(f"编码维度({current_dim})超过状态维度({state_dim})")
    
    def encode_architecture(self, config: Dict[str, Any]) -> np.ndarray:
        """编码架构配置"""
        encoding = np.zeros(self.total_encoding_dim)
        
        for param_name, value in config.items():
            if param_name not in self.param_encodings:
                continue
                
            param_info = self.param_encodings[param_name]
            start_idx = param_info['start_idx']
            
            if param_info['type'] == 'numerical':
                # 归一化数值
                min_val = param_info['min_val']
                max_val = param_info['max_val']
                normalized_val = (value - min_val) / (max_val - min_val)
                encoding[start_idx] = normalized_val
            else:
                # One-hot编码
                try:
                    choice_idx = param_info['choices'].index(value)
                    encoding[start_idx + choice_idx] = 1.0
                except ValueError:
                    # 如果值不在选择列表中，随机选择一个
                    choice_idx = 0
                    encoding[start_idx + choice_idx] = 1.0
        
        return encoding
    
    def encode_performance(self, metrics: Dict[str, float]) -> np.ndarray:
        """编码性能指标"""
        encoding = np.zeros(self.performance_dim)
        
        # 获取并归一化性能指标
        encoding[0] = metrics.get('accuracy', 0.0)
        encoding[1] = metrics.get('f1_score', 0.0)
        encoding[2] = metrics.get('auc', 0.0)
        encoding[3] = min(metrics.get('loss', 1.0), 1.0)  # 截断loss
        encoding[4] = min(metrics.get('val_loss', 1.0), 1.0)
        
        return encoding
    
    def encode_history(self, history: List[float]) -> np.ndarray:
        """编码历史性能"""
        encoding = np.zeros(self.history_dim)
        
        if len(history) > 0:
            # 取最近的历史记录
            recent_history = history[-self.history_dim:]
            for i, val in enumerate(recent_history):
                encoding[i] = val
        
        return encoding
    
    def encode_state(self, 
                    config: Dict[str, Any],
                    metrics: Dict[str, float],
                    history: List[float] = None) -> np.ndarray:
        """编码完整状态"""
        if history is None:
            history = []
            
        arch_encoding = self.encode_architecture(config)
        perf_encoding = self.encode_performance(metrics)
        hist_encoding = self.encode_history(history)
        
        # 拼接所有编码
        full_encoding = np.concatenate([arch_encoding, perf_encoding, hist_encoding])
        
        # 填充或截断到指定维度
        if len(full_encoding) < self.state_dim:
            padded_encoding = np.zeros(self.state_dim)
            padded_encoding[:len(full_encoding)] = full_encoding
            return padded_encoding
        else:
            return full_encoding[:self.state_dim]


class RLOptimizer:
    """强化学习优化器主类"""
    
    def __init__(self,
                 state_dim: int = 128,
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 clip_ratio: float = 0.2,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5):
        
        self.state_dim = state_dim
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        # 搜索空间和状态编码器
        self.search_space = ArchitectureSearchSpace()
        self.state_encoder = StateEncoder(state_dim)
        
        # 策略网络和价值网络
        action_space_sizes = {param: len(choices) 
                            for param, choices in self.search_space.search_space.items()}
        
        self.policy_net = PolicyNetwork(state_dim, action_space_sizes)
        self.value_net = ValueNetwork(state_dim)
        
        # 优化器
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=learning_rate)
        
        # 经验缓冲
        self.experience_buffer = []
        self.performance_history = []
        
        # 最佳配置记录
        self.best_config = None
        self.best_performance = -float('inf')
        
    def get_action_indices_to_config(self, action_indices: Dict[str, int]) -> Dict[str, Any]:
        """将动作索引转换为配置"""
        config = {}
        for param_name, action_idx in action_indices.items():
            if param_name in self.search_space.search_space:
                choices = self.search_space.search_space[param_name]
                config[param_name] = choices[action_idx]
        return config
    
    def select_architecture(self, current_metrics: Dict[str, float] = None) -> Dict[str, Any]:
        """选择下一个要评估的架构"""
        if current_metrics is None:
            current_metrics = {'accuracy': 0.0, 'f1_score': 0.0, 'auc': 0.0, 'loss': 1.0, 'val_loss': 1.0}
        
        # 编码当前状态
        if self.best_config is None:
            # 第一次选择，随机选择
            return self.search_space.sample_architecture()
        
        state_vector = self.state_encoder.encode_state(
            self.best_config, current_metrics, self.performance_history
        )
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
        
        # 使用策略网络选择动作
        with torch.no_grad():
            action_indices, _ = self.policy_net.sample_actions(state_tensor)
        
        # 转换为配置
        config = self.get_action_indices_to_config(action_indices)
        
        return config
    
    def update_performance(self, config: Dict[str, Any], metrics: Dict[str, float]):
        """更新性能记录"""
        # 计算奖励（可以根据具体需求调整）
        reward = self._calculate_reward(metrics)
        
        # 更新历史记录
        self.performance_history.append(reward)
        
        # 更新最佳配置
        if reward > self.best_performance:
            self.best_performance = reward
            self.best_config = config.copy()
        
        # 存储经验
        if len(self.experience_buffer) > 0:
            # 为上一个经验添加奖励
            self.experience_buffer[-1]['reward'] = reward
            self.experience_buffer[-1]['next_state'] = self.state_encoder.encode_state(
                config, metrics, self.performance_history
            )
        
        # 添加当前经验（下次会补充reward和next_state）
        current_state = self.state_encoder.encode_state(
            config, metrics, self.performance_history[:-1] if len(self.performance_history) > 1 else []
        )
        
        self.experience_buffer.append({
            'state': current_state,
            'action_indices': None,  # 在下次选择时会设置
            'log_probs': None,
            'reward': None,
            'next_state': None,
            'done': False
        })
    
    def _calculate_reward(self, metrics: Dict[str, float]) -> float:
        """计算奖励函数"""
        # 多目标奖励：准确率、F1分数、AUC，惩罚高损失
        accuracy = metrics.get('accuracy', 0.0)
        f1_score = metrics.get('f1_score', 0.0)
        auc = metrics.get('auc', 0.0)
        loss = metrics.get('loss', 1.0)
        
        # 加权组合
        reward = 0.4 * accuracy + 0.3 * f1_score + 0.3 * auc - 0.1 * loss
        
        # 奖励改进
        if len(self.performance_history) > 1:
            improvement = reward - self.performance_history[-2]
            reward += 0.1 * improvement
        
        return reward
    
    def train_policy(self, batch_size: int = 32, num_epochs: int = 10):
        """训练策略网络"""
        if len(self.experience_buffer) < batch_size:
            return
        
        # 计算优势函数
        self._compute_advantages()
        
        # 训练循环
        for epoch in range(num_epochs):
            # 随机采样batch
            batch_indices = random.sample(range(len(self.experience_buffer) - 1), 
                                        min(batch_size, len(self.experience_buffer) - 1))
            
            states = torch.FloatTensor([self.experience_buffer[i]['state'] for i in batch_indices])
            advantages = torch.FloatTensor([self.experience_buffer[i]['advantage'] for i in batch_indices])
            returns = torch.FloatTensor([self.experience_buffer[i]['return'] for i in batch_indices])
            
            # 旧的log概率（用于PPO裁剪）
            old_log_probs = {}
            for param_name in self.search_space.search_space.keys():
                old_log_probs[param_name] = torch.FloatTensor([
                    self.experience_buffer[i]['log_probs'][param_name].item() 
                    for i in batch_indices
                ])
            
            # 当前策略的输出
            action_probs = self.policy_net(states)
            values = self.value_net(states).squeeze()
            
            # 计算策略损失（PPO）
            policy_loss = 0
            entropy_loss = 0
            
            for param_name, probs in action_probs.items():
                # 重新计算log概率
                action_indices = [self.experience_buffer[i]['action_indices'][param_name] 
                                for i in batch_indices]
                new_log_probs = torch.log(probs[range(len(action_indices)), action_indices])
                
                # PPO裁剪
                ratio = torch.exp(new_log_probs - old_log_probs[param_name])
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                
                policy_loss += -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
                
                # 熵正则化
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                entropy_loss += entropy.mean()
            
            # 价值损失
            value_loss = F.mse_loss(values, returns)
            
            # 总损失
            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
            
            # 反向传播
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()
            self.policy_optimizer.step()
            self.value_optimizer.step()
    
    def _compute_advantages(self):
        """计算优势函数（GAE）"""
        if len(self.experience_buffer) < 2:
            return
        
        # 计算returns和advantages
        for i in range(len(self.experience_buffer) - 1):
            exp = self.experience_buffer[i]
            next_exp = self.experience_buffer[i + 1]
            
            # 计算return
            reward = exp['reward'] if exp['reward'] is not None else 0.0
            next_value = 0.0
            
            if next_exp['next_state'] is not None:
                next_state_tensor = torch.FloatTensor(next_exp['next_state']).unsqueeze(0)
                with torch.no_grad():
                    next_value = self.value_net(next_state_tensor).item()
            
            exp['return'] = reward + self.gamma * next_value
            
            # 计算advantage（简化版，可以使用GAE改进）
            current_state_tensor = torch.FloatTensor(exp['state']).unsqueeze(0)
            with torch.no_grad():
                current_value = self.value_net(current_state_tensor).item()
            
            exp['advantage'] = exp['return'] - current_value
    
    def save_checkpoint(self, filepath: str):
        """保存检查点"""
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'best_config': self.best_config,
            'best_performance': self.best_performance,
            'performance_history': self.performance_history
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.best_config = checkpoint['best_config']
        self.best_performance = checkpoint['best_performance']
        self.performance_history = checkpoint['performance_history']


# 使用示例
if __name__ == "__main__":
    # 创建强化学习优化器
    rl_optimizer = RLOptimizer(state_dim=128)
    
    # 模拟优化过程
    for iteration in range(10):
        # 选择架构
        config = rl_optimizer.select_architecture()
        print(f"迭代 {iteration + 1}, 选择的配置: {config}")
        
        # 模拟训练和评估
        simulated_metrics = {
            'accuracy': random.uniform(0.7, 0.95),
            'f1_score': random.uniform(0.65, 0.92),
            'auc': random.uniform(0.75, 0.98),
            'loss': random.uniform(0.1, 0.5),
            'val_loss': random.uniform(0.15, 0.6)
        }
        
        # 更新性能
        rl_optimizer.update_performance(config, simulated_metrics)
        
        # 训练策略（每5次迭代）
        if (iteration + 1) % 5 == 0:
            rl_optimizer.train_policy()
        
        print(f"性能指标: {simulated_metrics}")
        print(f"当前最佳性能: {rl_optimizer.best_performance:.4f}")
        print("-" * 50)
    
    print(f"最终最佳配置: {rl_optimizer.best_config}")
    print(f"最终最佳性能: {rl_optimizer.best_performance:.4f}") 