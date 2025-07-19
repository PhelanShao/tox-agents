import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        
    def forward(self, 
                features: torch.Tensor, 
                labels: torch.Tensor,
                positive_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = features.shape[0]
        device = features.device
        
        features = F.normalize(features, dim=1)
        
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        if positive_mask is None:
            positive_mask = self._create_positive_mask(labels)
        
        mask = torch.eye(batch_size, device=device).bool()
        similarity_matrix.masked_fill_(mask, float('-inf'))
        positive_mask.masked_fill_(mask, False)
        
        exp_sim = torch.exp(similarity_matrix)
        sum_exp_sim = exp_sim.sum(dim=1, keepdim=True)
        
        pos_sim = similarity_matrix * positive_mask.float()
        pos_exp_sim = torch.exp(pos_sim) * positive_mask.float()
        
        num_positives = positive_mask.sum(dim=1)
        
        log_prob = pos_sim - torch.log(sum_exp_sim + 1e-8)
        loss = -(log_prob * positive_mask.float()).sum(dim=1) / (num_positives + 1e-8)
        
        valid_samples = num_positives > 0
        if valid_samples.sum() > 0:
            loss = loss[valid_samples]
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    def _create_positive_mask(self, labels: torch.Tensor, threshold: float = 0.8) -> torch.Tensor:
        batch_size = labels.shape[0]
        device = labels.device
        
        labels_norm = F.normalize(labels.float(), dim=1)
        label_similarity = torch.matmul(labels_norm, labels_norm.T)
        
        positive_mask = label_similarity > threshold
        
        return positive_mask


class TripletLoss(nn.Module):
    def __init__(self, margin: float = 1.0, p: int = 2, reduction: str = 'mean'):
        super().__init__()
        self.margin = margin
        self.p = p
        self.reduction = reduction
        
    def forward(self,
                anchor: torch.Tensor,
                positive: torch.Tensor, 
                negative: torch.Tensor) -> torch.Tensor:
        pos_dist = F.pairwise_distance(anchor, positive, p=self.p)
        neg_dist = F.pairwise_distance(anchor, negative, p=self.p)
        
        loss = F.relu(pos_dist - neg_dist + self.margin)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        batch_size = features.shape[0]
        
        features = F.normalize(features, dim=1)
        
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T), self.temperature
        )
        
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        labels_normalized = F.normalize(labels.float(), dim=1)
        mask = torch.matmul(labels_normalized, labels_normalized.T) > 0.8
        
        mask = mask.fill_diagonal_(False)
        
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        
        valid_samples = mask.sum(1) > 0
        if valid_samples.sum() > 0:
            loss = loss[valid_samples].mean()
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            
        return loss


class HardNegativeMiner(nn.Module):
    def __init__(self, negative_ratio: float = 0.5, distance_threshold: float = 0.1):
        super().__init__()
        self.negative_ratio = negative_ratio
        self.distance_threshold = distance_threshold
        
    def mine_hard_negatives(self,
                           features: torch.Tensor,
                           labels: torch.Tensor,
                           num_negatives: int) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = features.shape[0]
        device = features.device
        
        features_norm = F.normalize(features, dim=1)
        distance_matrix = torch.cdist(features_norm, features_norm, p=2)
        
        labels_norm = F.normalize(labels.float(), dim=1)
        label_similarity = torch.matmul(labels_norm, labels_norm.T)
        
        negative_mask = label_similarity < 0.3
        hard_mask = distance_matrix < self.distance_threshold
        
        hard_negative_mask = negative_mask & hard_mask
        
        hard_negative_mask.fill_diagonal_(False)
        
        hard_scores = (1.0 / (distance_matrix + 1e-8)) * (1.0 - label_similarity) * hard_negative_mask.float()
        
        hard_negative_indices = []
        hard_negative_scores = []
        
        for i in range(batch_size):
            scores_i = hard_scores[i]
            if scores_i.sum() > 0:
                _, top_indices = torch.topk(scores_i, min(num_negatives, (scores_i > 0).sum().item()))
                hard_negative_indices.append(top_indices)
                hard_negative_scores.append(scores_i[top_indices])
            else:
                random_indices = torch.randperm(batch_size, device=device)[:num_negatives]
                random_indices = random_indices[random_indices != i]
                hard_negative_indices.append(random_indices)
                hard_negative_scores.append(torch.zeros(len(random_indices), device=device))
        
        return hard_negative_indices, hard_negative_scores


class ContrastiveToxicityLoss(nn.Module):
    def __init__(self,
                 temperature: float = 0.1,
                 margin: float = 1.0,
                 alpha: float = 0.5,
                 beta: float = 0.3,
                 gamma: float = 0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta  
        self.gamma = gamma
        
        self.infonce_loss = InfoNCELoss(temperature=temperature)
        self.triplet_loss = TripletLoss(margin=margin)
        self.supcon_loss = SupConLoss(temperature=temperature)
        
        self.hard_negative_miner = HardNegativeMiner()
        
    def forward(self,
                features: torch.Tensor,
                labels: torch.Tensor,
                return_components: bool = False) -> torch.Tensor:
        batch_size = features.shape[0]
        device = features.device
        
        infonce_loss = self.infonce_loss(features, labels)
        
        supcon_loss = self.supcon_loss(features, labels)
        
        triplet_loss = torch.tensor(0.0, device=device)
        if batch_size >= 3:
            hard_neg_indices, _ = self.hard_negative_miner.mine_hard_negatives(
                features, labels, num_negatives=1
            )
            
            anchors = []
            positives = []
            negatives = []
            
            labels_norm = F.normalize(labels.float(), dim=1)
            similarity_matrix = torch.matmul(labels_norm, labels_norm.T)
            
            for i in range(batch_size):
                sim_scores = similarity_matrix[i]
                sim_scores[i] = -1
                pos_idx = sim_scores.argmax()
                
                if sim_scores[pos_idx] > 0.5:
                    anchors.append(features[i])
                    positives.append(features[pos_idx])
                    
                    if len(hard_neg_indices[i]) > 0:
                        neg_idx = hard_neg_indices[i][0]
                        negatives.append(features[neg_idx])
                    else:
                        neg_idx = torch.randint(0, batch_size, (1,), device=device)[0]
                        while neg_idx == i or neg_idx == pos_idx:
                            neg_idx = torch.randint(0, batch_size, (1,), device=device)[0]
                        negatives.append(features[neg_idx])
            
            if len(anchors) > 0:
                anchor_tensor = torch.stack(anchors)
                positive_tensor = torch.stack(positives)
                negative_tensor = torch.stack(negatives)
                triplet_loss = self.triplet_loss(anchor_tensor, positive_tensor, negative_tensor)
        
        total_loss = (self.alpha * infonce_loss + 
                     self.beta * triplet_loss + 
                     self.gamma * supcon_loss)
        
        if return_components:
            return {
                'total_loss': total_loss,
                'infonce_loss': infonce_loss,
                'triplet_loss': triplet_loss,
                'supcon_loss': supcon_loss
            }
        
        return total_loss


class ContrastiveDataAugmentation:
    @staticmethod
    def molecular_augmentation(features: torch.Tensor, 
                             noise_level: float = 0.1) -> torch.Tensor:
        noise = torch.randn_like(features) * noise_level
        augmented = features + noise
        return augmented
    
    @staticmethod
    def create_positive_pairs(batch_features: torch.Tensor,
                            batch_labels: torch.Tensor,
                            similarity_threshold: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = batch_features.shape[0]
        
        labels_norm = F.normalize(batch_labels.float(), dim=1)
        similarity_matrix = torch.matmul(labels_norm, labels_norm.T)
        
        positive_pairs = []
        pair_indices = []
        
        for i in range(batch_size):
            similar_indices = torch.where(similarity_matrix[i] > similarity_threshold)[0]
            similar_indices = similar_indices[similar_indices != i]
            
            if len(similar_indices) > 0:
                for j in similar_indices:
                    positive_pairs.append((batch_features[i], batch_features[j]))
                    pair_indices.append((i, j.item()))
        
        if len(positive_pairs) > 0:
            pairs_tensor = torch.stack([torch.stack(pair) for pair in positive_pairs])
            return pairs_tensor, pair_indices
        else:
            return torch.empty(0, 2, batch_features.shape[1]), []


if __name__ == "__main__":
    contrastive_loss = ContrastiveToxicityLoss(
        temperature=0.1,
        margin=1.0,
        alpha=0.5,
        beta=0.3,
        gamma=0.2
    )
    
    batch_size = 8
    feature_dim = 512
    num_tasks = 26
    
    features = torch.randn(batch_size, feature_dim)
    labels = torch.rand(batch_size, num_tasks)
    
    try:
        loss_dict = contrastive_loss(features, labels, return_components=True)
        
        print("Contrastive Loss Components:")
        for key, value in loss_dict.items():
            print(f"{key}: {value.item():.4f}")
        
        print(f"\nContrastive loss function created successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()