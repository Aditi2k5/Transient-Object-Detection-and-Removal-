"""
TADD-GS Loss Functions - Complete Implementation
Based on your original research framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalContrastiveLoss(nn.Module):
    """
    Self-supervised contrastive learning for distractor detection
    
    Based on your framework:
    - Contrastive temporal loss that pulls together static Gaussians (low motion variance)
    - Pushes apart distractors (high motion variance) from static points
    - Uses SimCLR-inspired contrastive learning
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features_t, features_t1, motion_variance, static_threshold=0.1, dynamic_threshold=0.3):
        """
        Args:
            features_t: Features at time t [N, D]
            features_t1: Features at time t+1 [N, D]
            motion_variance: Motion variance for each Gaussian [N]
            static_threshold: Below this = static
            dynamic_threshold: Above this = distractor
        
        Returns:
            Contrastive loss value
        """
        # Separate static and dynamic Gaussians
        static_mask = motion_variance < static_threshold
        dynamic_mask = motion_variance > dynamic_threshold
        
        if static_mask.sum() < 2 or dynamic_mask.sum() < 1:
            # Not enough samples for contrastive learning
            return torch.tensor(0.0, device=features_t.device)
        
        # Normalize features
        features_t = F.normalize(features_t, dim=-1)
        features_t1 = F.normalize(features_t1, dim=-1)
        
        # === Positive Pairs: Same static Gaussian across consecutive frames ===
        # These should have high similarity (pulled together)
        static_feat_t = features_t[static_mask]
        static_feat_t1 = features_t1[static_mask]
        
        positive_sim = (static_feat_t * static_feat_t1).sum(dim=-1) / self.temperature
        positive_loss = -torch.log(torch.sigmoid(positive_sim)).mean()
        
        # === Negative Pairs: Static vs Dynamic Gaussians ===
        # These should have low similarity (pushed apart)
        static_feat = features_t[static_mask]
        dynamic_feat = features_t[dynamic_mask]
        
        # Compute all pairwise similarities
        neg_sim = torch.matmul(static_feat, dynamic_feat.T) / self.temperature
        
        # InfoNCE-style negative loss
        negative_loss = torch.logsumexp(neg_sim, dim=-1).mean()
        
        # Combined loss
        loss = positive_loss + negative_loss
        
        return loss


class DistractorRegularizationLoss(nn.Module):
    """
    Regularization term that penalizes Gaussians with persistent motion
    
    Based on your framework:
    - Penalizes Gaussians exhibiting persistent motion
    - Encourages sparse distractor detection
    - Effectively "fades" them out of static reconstruction
    """
    def __init__(self, sparsity_weight=1.0):
        super().__init__()
        self.sparsity_weight = sparsity_weight
    
    def forward(self, motion_variance, distractor_threshold=0.3):
        """
        Args:
            motion_variance: Motion variance for each Gaussian [N]
            distractor_threshold: Threshold for considering a Gaussian as distractor
        
        Returns:
            Regularization loss
        """
        # L1 sparsity on motion variance (encourages most to be 0)
        sparsity_loss = motion_variance.abs().mean()
        
        # Extra penalty for high-variance points (push them to be classified clearly)
        distractor_mask = motion_variance > distractor_threshold
        if distractor_mask.sum() > 0:
            distractor_penalty = (motion_variance[distractor_mask] ** 2).mean()
        else:
            distractor_penalty = torch.tensor(0.0, device=motion_variance.device)
        
        loss = self.sparsity_weight * sparsity_loss + distractor_penalty
        
        return loss


def temporal_contrastive_loss(features_t, features_t1, motion_variance=None, temperature=0.07):
    """
    Wrapper function for backward compatibility
    
    Args:
        features_t: Features at time t [N, D] or same as features_t1 for fallback
        features_t1: Features at time t+1 [N, D]
        motion_variance: Optional motion variance [N]
        temperature: Temperature for contrastive loss
    
    Returns:
        Contrastive loss value
    """
    if motion_variance is None:
        # Fallback: simple feature similarity loss
        features_t = F.normalize(features_t, dim=-1)
        features_t1 = F.normalize(features_t1, dim=-1)
        loss = 1.0 - (features_t * features_t1).sum(dim=-1).mean()
        return loss
    
    # Use proper contrastive learning
    criterion = TemporalContrastiveLoss(temperature=temperature)
    return criterion(features_t, features_t1, motion_variance)


def distractor_regularization_loss(motion_variance, threshold=0.3, sparsity_weight=1.0):
    """
    Wrapper function for distractor regularization
    
    Args:
        motion_variance: Motion variance for each Gaussian [N]
        threshold: Distractor threshold
        sparsity_weight: Weight for sparsity term
    
    Returns:
        Regularization loss value
    """
    criterion = DistractorRegularizationLoss(sparsity_weight=sparsity_weight)
    return criterion(motion_variance, distractor_threshold=threshold)


# For backward compatibility, keep the old function names
__all__ = [
    'TemporalContrastiveLoss',
    'DistractorRegularizationLoss', 
    'temporal_contrastive_loss',
    'distractor_regularization_loss'
]