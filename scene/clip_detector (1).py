"""
CLIP Distractor Detector - FIXED VERSION
Handles dynamic Gaussian count changes during densification
"""

import torch
import torch.nn.functional as F
import clip
from typing import Optional, Tuple

class CLIPDistractorDetector:
    """Semantic distractor detection with dynamic Gaussian tracking"""
    
    def __init__(self, device="cuda", clip_model="ViT-B/32"):
        self.device = device
        
        print(f"[CLIP] Loading {clip_model}...")
        self.clip_model, self.clip_preprocess = clip.load(clip_model, device=device)
        self.clip_model.eval()
        print("[CLIP] ✅ Model loaded successfully")
        
        # Comprehensive prompts
        self.distractor_texts = [
            "a photo of a person",
            "a photo of people",
            "a photo of a human",
            "a photo of pedestrians",
            "a photo of hands",
            "a photo of a balloon",
            "a photo of a bottle",
            "a photo of something moving",
            "a photo of a temporary object"
        ]
        
        self.static_texts = [
            "a photo of a building",
            "a photo of architecture",
            "a photo of a wall",
            "a photo of furniture",
            "a photo of a statue",
            "a photo of the ground"
        ]
        
        # Encode prompts
        with torch.no_grad():
            dist_tokens = clip.tokenize(self.distractor_texts).to(device)
            static_tokens = clip.tokenize(self.static_texts).to(device)
            
            self.dist_features = self.clip_model.encode_text(dist_tokens)
            self.static_features = self.clip_model.encode_text(static_tokens)
            
            self.dist_features /= self.dist_features.norm(dim=-1, keepdim=True)
            self.static_features /= self.static_features.norm(dim=-1, keepdim=True)
        
        # Tracking
        self.view_counts = None
        self.gaussian_scores = None
        self.num_updates = 0
        
        print(f"[CLIP] Initialized with {len(self.distractor_texts)} distractor prompts")
    
    def compute_image_score(self, image_tensor: torch.Tensor) -> Tuple[float, float]:
        """Compute CLIP scores for image"""
        try:
            with torch.no_grad():
                img = image_tensor.unsqueeze(0)
                img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
                img = torch.clamp(img, 0, 1)
                
                img_features = self.clip_model.encode_image(img)
                img_features /= img_features.norm(dim=-1, keepdim=True)
                
                dist_sim = (img_features @ self.dist_features.T).max(dim=1)[0]
                static_sim = (img_features @ self.static_features.T).max(dim=1)[0]
                
                dist_score = ((dist_sim + 1) / 2).item()
                static_score = ((static_sim + 1) / 2).item()
                
            return dist_score, static_score
        except Exception as e:
            print(f"[CLIP] Warning: {e}")
            return 0.5, 0.5
    
    def _sync_size(self, num_gaussians: int):
        """Ensure tracking arrays match current Gaussian count"""
        if self.view_counts is None:
            self.view_counts = torch.zeros(num_gaussians, device=self.device)
            self.gaussian_scores = torch.zeros(num_gaussians, device=self.device)
            return
        
        current_size = self.view_counts.shape[0]
        
        if current_size != num_gaussians:
            new_counts = torch.zeros(num_gaussians, device=self.device)
            new_scores = torch.zeros(num_gaussians, device=self.device)
            
            # Copy old values (up to min size)
            copy_size = min(current_size, num_gaussians)
            new_counts[:copy_size] = self.view_counts[:copy_size]
            new_scores[:copy_size] = self.gaussian_scores[:copy_size]
            
            self.view_counts = new_counts
            self.gaussian_scores = new_scores
    
    def update_tracking(self, gaussians, visibility_filter: torch.Tensor, 
                       image_tensor: torch.Tensor, iteration: int):
        """Update tracking"""
        num_gaussians = gaussians.get_xyz.shape[0]
        
        # CRITICAL: Sync size BEFORE any operations
        self._sync_size(num_gaussians)
        
        # Update view counts
        self.view_counts[visibility_filter] += 1
        
        # Compute CLIP score
        dist_score, static_score = self.compute_image_score(image_tensor)
        
        # Update Gaussian scores
        if dist_score > 0.5:
            alpha = 0.1
            self.gaussian_scores[visibility_filter] += alpha * (dist_score - 0.5)
        
        self.num_updates += 1
        
        if self.num_updates % 100 == 0:
            avg_score = self.gaussian_scores[self.view_counts > 0].mean().item()
            print(f"[CLIP] Update {self.num_updates}: Avg score: {avg_score:.4f}")
    
    # EMERGENCY FIX - Replace in clip_detector.py

    def get_distractor_mask(self, gaussians, iteration: int, 
                          score_threshold: float = 0.015,  # CHANGED FROM 0.3!
                          min_views: int = 10) -> torch.Tensor:
        """Get pruning mask - WITH CORRECTED THRESHOLD"""
        num_gaussians = gaussians.get_xyz.shape[0]
        self._sync_size(num_gaussians)
        
        if self.gaussian_scores is None:
            return torch.zeros(num_gaussians, dtype=torch.bool, device=self.device)
        
        # Normalize scores
        normalized_scores = torch.zeros(num_gaussians, device=self.device)
        valid_mask = self.view_counts > 0
        
        if valid_mask.sum() > 0:
            normalized_scores[valid_mask] = self.gaussian_scores[valid_mask] / self.view_counts[valid_mask]
        
        opacities = gaussians.get_opacity.squeeze()
        if opacities.shape[0] != num_gaussians:
            return torch.zeros(num_gaussians, dtype=torch.bool, device=self.device)
        
        # ADJUSTED CRITERIA
        high_score = normalized_scores > score_threshold  # 0.015 instead of 0.3
        low_views = self.view_counts < min_views
        low_opacity = opacities < 0.1
        
        prune_mask = high_score | (low_views & low_opacity)
        
        return prune_mask
    
    def get_opacity_penalty(self, gaussians, lambda_penalty: float = 0.01) -> torch.Tensor:
        """Opacity regularization - WITH SIZE SAFETY"""
        num_gaussians = gaussians.get_xyz.shape[0]
        
        # Sync size
        self._sync_size(num_gaussians)
        
        if self.gaussian_scores is None:
            return torch.tensor(0.0, device=self.device)
        
        # Normalize
        normalized_scores = torch.zeros(num_gaussians, device=self.device)
        valid_mask = self.view_counts > 0
        
        if valid_mask.sum() > 0:
            normalized_scores[valid_mask] = self.gaussian_scores[valid_mask] / self.view_counts[valid_mask]
        
        # Get opacity
        opacities = gaussians.get_opacity.squeeze()
        
        # SIZE CHECK
        if opacities.shape[0] != num_gaussians:
            print(f"[CLIP] WARNING: Opacity size mismatch in penalty!")
            return torch.tensor(0.0, device=self.device)
        
        penalty = (normalized_scores * opacities).mean()
        return lambda_penalty * penalty
    
    def get_statistics(self) -> dict:
        """Get stats"""
        if self.gaussian_scores is None:
            return {}
        
        valid_mask = self.view_counts > 0
        if valid_mask.sum() == 0:
            return {}
        
        normalized_scores = self.gaussian_scores[valid_mask] / self.view_counts[valid_mask]
        
        return {
            'num_tracked': valid_mask.sum().item(),
            'avg_score': normalized_scores.mean().item(),
            'high_score_count': (normalized_scores > 0.3).sum().item(),
            'avg_views': self.view_counts[valid_mask].mean().item()
        }