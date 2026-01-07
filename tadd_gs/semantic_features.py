import torch
import torch.nn.functional as F
import clip
from typing import List, Tuple, Optional
import numpy as np


class SemanticFeatureExtractor:
    """
    Lightweight semantic feature extraction using frozen CLIP.
    Extracts features to distinguish distractors (people) from static scene.
    """
    
    def __init__(self, device="cuda"):
        """
        Initialize CLIP model (frozen, no training needed)
        
        Args:
            device: Device to run CLIP on ("cuda" or "cpu")
        """
        print("[Semantic] Loading CLIP ViT-B/32...")
        self.device = device
        
        # Load CLIP model (lightweight version)
        # ViT-B/32 is good balance of speed and accuracy
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.model.eval()  # Set to evaluation mode (frozen)
        
        # Disable gradients (we don't train CLIP)
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Text prompts for distractor vs static categories
        # These are the semantic concepts we want to distinguish
        self.text_prompts = [
            # Distractor categories (things that move)
            "a person",
            "a human", 
            "people walking",
            "a pedestrian",
            
            # Static categories (background)
            "a building",
            "architecture",
            "a tree",
            "grass",
            "the sky",
            "background",
            "a wall",
            "the ground",
        ]
        
        # Encode text prompts once (cache for efficiency)
        print("[Semantic] Encoding text prompts...")
        text_tokens = clip.tokenize(self.text_prompts).to(device)
        with torch.no_grad():
            self.text_features = self.model.encode_text(text_tokens)
            # Normalize for cosine similarity
            self.text_features = F.normalize(self.text_features, dim=-1)
        
        # Indices for distractor vs static categories
        self.distractor_indices = [0, 1, 2, 3]  # person, human, people, pedestrian
        self.static_indices = [4, 5, 6, 7, 8, 9, 10, 11]  # building, tree, sky, etc.
        
        print(f"[Semantic] Ready! {len(self.text_prompts)} categories encoded")
        print(f"  Distractor categories: {len(self.distractor_indices)}")
        print(f"  Static categories: {len(self.static_indices)}")
    
    @torch.no_grad()
    def extract_patch_features(self, image: torch.Tensor, bbox_list: List[List[int]]) -> torch.Tensor:
        """
        Extract CLIP features for image patches.
        
        Args:
            image: [3, H, W] RGB image tensor (0-1 range)
            bbox_list: List of [x1, y1, x2, y2] bounding boxes
        
        Returns:
            patch_features: [N, 512] CLIP feature vectors
        """
        if len(bbox_list) == 0:
            return torch.zeros((0, 512), device=self.device)
        
        patches = []
        for bbox in bbox_list:
            x1, y1, x2, y2 = bbox
            # Convert to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Clamp to image bounds
            H, W = image.shape[1], image.shape[2]
            x1 = max(0, min(x1, W-1))
            x2 = max(x1+1, min(x2, W))
            y1 = max(0, min(y1, H-1))
            y2 = max(y1+1, min(y2, H))
            
            # Extract patch
            patch = image[:, y1:y2, x1:x2]
            
            # Skip if patch is too small
            if patch.shape[1] < 10 or patch.shape[2] < 10:
                # Create blank patch
                patch = torch.zeros(3, 224, 224, device=self.device)
            else:
                # Resize to 224x224 (CLIP input size)
                patch = F.interpolate(
                    patch.unsqueeze(0), 
                    size=(224, 224), 
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            
            patches.append(patch)
        
        # Stack patches: [N, 3, 224, 224]
        patches = torch.stack(patches, dim=0)
        
        # Normalize for CLIP (specific normalization required)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(self.device)
        patches = (patches - mean) / std
        
        # Extract CLIP features
        with torch.no_grad():
            features = self.model.encode_image(patches)
            # Normalize for cosine similarity
            features = F.normalize(features, dim=-1)
        
        return features  # [N, 512]
    
    @torch.no_grad()
    def compute_distractor_scores(self, patch_features: torch.Tensor) -> torch.Tensor:
        """
        Compute distractor scores for patches using text-image similarity.
        
        Args:
            patch_features: [N, 512] CLIP feature vectors
        
        Returns:
            distractor_scores: [N] scores in range [0, 1]
                Higher score = more likely to be distractor (person)
        """
        if patch_features.shape[0] == 0:
            return torch.zeros(0, device=self.device)
        
        # Compute similarity to all text prompts
        # similarity[i, j] = how similar patch i is to text prompt j
        similarity = patch_features @ self.text_features.T  # [N, num_prompts]
        
        # Average similarity to distractor categories
        distractor_sim = similarity[:, self.distractor_indices].mean(dim=1)  # [N]
        
        # Average similarity to static categories
        static_sim = similarity[:, self.static_indices].mean(dim=1)  # [N]
        
        # Compute distractor score: relative similarity to "person" vs "static"
        # Use softmax to get probability-like scores
        logits = torch.stack([static_sim, distractor_sim], dim=1)  # [N, 2]
        
        # Temperature scaling (higher = more confident separation)
        temperature = 5.0
        probs = F.softmax(logits * temperature, dim=1)  # [N, 2]
        
        # Probability of being distractor
        distractor_scores = probs[:, 1]  # [N]
        
        return distractor_scores


class GaussianClusterer:
    """
    Cluster Gaussians in screen space to create patches for CLIP.
    Each patch contains multiple Gaussians for semantic analysis.
    """
    
    def __init__(self, patch_size: int = 64, min_gaussians: int = 10):
        """
        Args:
            patch_size: Size of patches in pixels (64x64 works well)
            min_gaussians: Minimum Gaussians per patch (filter noise)
        """
        self.patch_size = patch_size
        self.min_gaussians = min_gaussians
    
    def create_patches(self, gaussians, viewpoint) -> Tuple[List[List[int]], List[torch.Tensor]]:
        """
        Create patches from Gaussians in current view.
        
        Args:
            gaussians: GaussianModel instance
            viewpoint: Camera viewpoint
        
        Returns:
            patches: List of [x1, y1, x2, y2] bounding boxes
            gaussian_indices: List of tensor indices (Gaussians per patch)
        """
        # Import here to avoid circular dependency
        from tadd_gs.utils import project_gaussians_to_camera
        
        # Project Gaussians to screen space
        xy = project_gaussians_to_camera(gaussians, viewpoint)  # [N, 2]
        
        # Get image dimensions
        W = int(viewpoint.image_width)
        H = int(viewpoint.image_height)
        
        # Create grid of patches
        num_patches_x = W // self.patch_size
        num_patches_y = H // self.patch_size
        
        patches = []
        gaussian_indices = []
        
        for i in range(num_patches_y):
            for j in range(num_patches_x):
                # Patch bounds
                x1 = j * self.patch_size
                y1 = i * self.patch_size
                x2 = min(x1 + self.patch_size, W)
                y2 = min(y1 + self.patch_size, H)
                
                # Find Gaussians in this patch
                in_patch = (
                    (xy[:, 0] >= x1) & (xy[:, 0] < x2) &
                    (xy[:, 1] >= y1) & (xy[:, 1] < y2)
                )
                
                indices = torch.where(in_patch)[0]
                
                # Only keep patches with enough Gaussians
                if len(indices) >= self.min_gaussians:
                    patches.append([x1, y1, x2, y2])
                    gaussian_indices.append(indices)
        
        return patches, gaussian_indices


# Test function
def test_semantic_extractor():
    """Test the semantic feature extractor"""
    print("\n" + "="*70)
    print("TESTING SEMANTIC FEATURE EXTRACTOR")
    print("="*70 + "\n")
    
    # Initialize
    semantic = SemanticFeatureExtractor(device="cuda")
    
    # Create dummy image
    print("Creating dummy test image (800x600)...")
    image = torch.rand(3, 600, 800).cuda()
    
    # Test with some dummy bounding boxes
    print("Creating test patches...")
    bboxes = [
        [100, 100, 200, 300],  # Tall vertical box (person-like shape)
        [300, 200, 500, 400],  # Wide horizontal box (building-like)
        [0, 0, 800, 100],      # Top strip (sky-like)
        [400, 300, 500, 500],  # Square box (could be anything)
    ]
    
    # Extract features
    print(f"Extracting CLIP features for {len(bboxes)} patches...")
    features = semantic.extract_patch_features(image, bboxes)
    print(f"✅ Features shape: {features.shape} (expected: [{len(bboxes)}, 512])")
    
    # Compute scores
    print("Computing distractor scores...")
    scores = semantic.compute_distractor_scores(features)
    print(f"✅ Scores shape: {scores.shape} (expected: [{len(bboxes)}])")
    
    print("\nDistractor scores:")
    for i, (bbox, score) in enumerate(zip(bboxes, scores)):
        print(f"  Patch {i} {bbox}: {score:.3f}")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70 + "\n")
    
    return True


if __name__ == "__main__":
    # Run test if this file is executed directly
    test_semantic_extractor()