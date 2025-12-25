"""
TADD-GS Two-Layer Decomposition
Separates static scene from distractor layer
"""

import torch
import os
from scene.gaussian_model import GaussianModel

class DecomposedGaussianModel:
    """
    Two-layer Gaussian model: Static scene + Distractor layer
    
    Based on your framework:
    - Decomposes scene into static and distractor layers
    - Can render each layer separately or combined
    - Preserves distractors for analysis/editing
    """
    def __init__(self):
        self.static_gaussians = None
        self.distractor_gaussians = None
        self.decomposed = False
    
    def decompose(self, gaussians, distractor_scores, threshold=0.7):
        """
        Decompose Gaussians into static and distractor layers
        
        Args:
            gaussians: GaussianModel with all Gaussians
            distractor_scores: Distractor probability for each Gaussian [N]
            threshold: Score threshold for classification
        
        Returns:
            static_mask: Boolean mask for static Gaussians
            distractor_mask: Boolean mask for distractor Gaussians
        """
        # Classify Gaussians
        distractor_mask = distractor_scores > threshold
        static_mask = ~distractor_mask
        
        # Store decomposed parameters
        self.static_params = self._extract_params(gaussians, static_mask)
        self.distractor_params = self._extract_params(gaussians, distractor_mask)
        
        self.decomposed = True
        
        print(f"[TADD-GS] Decomposed scene:")
        print(f"  Static Gaussians: {static_mask.sum().item():,}")
        print(f"  Distractor Gaussians: {distractor_mask.sum().item():,}")
        
        return static_mask, distractor_mask
    
    def _extract_params(self, gaussians, mask):
        """
        Extract parameters for a subset of Gaussians
        
        Args:
            gaussians: GaussianModel
            mask: Boolean mask [N]
        
        Returns:
            Dict with extracted parameters
        """
        params = {
            'xyz': gaussians._xyz[mask].detach().clone(),
            'features_dc': gaussians._features_dc[mask].detach().clone(),
            'features_rest': gaussians._features_rest[mask].detach().clone(),
            'scaling': gaussians._scaling[mask].detach().clone(),
            'rotation': gaussians._rotation[mask].detach().clone(),
            'opacity': gaussians._opacity[mask].detach().clone(),
        }
        return params
    
    def create_static_model(self, sh_degree, optimizer_type="default"):
        """
        Create GaussianModel containing only static Gaussians
        
        Args:
            sh_degree: Spherical harmonics degree
            optimizer_type: Optimizer type
        
        Returns:
            GaussianModel with static Gaussians
        """
        if not self.decomposed:
            raise ValueError("Must decompose first!")
        
        model = GaussianModel(sh_degree, optimizer_type)
        self._load_params_to_model(model, self.static_params)
        return model
    
    def create_distractor_model(self, sh_degree, optimizer_type="default"):
        """
        Create GaussianModel containing only distractor Gaussians
        
        Args:
            sh_degree: Spherical harmonics degree
            optimizer_type: Optimizer type
        
        Returns:
            GaussianModel with distractor Gaussians
        """
        if not self.decomposed:
            raise ValueError("Must decompose first!")
        
        model = GaussianModel(sh_degree, optimizer_type)
        self._load_params_to_model(model, self.distractor_params)
        return model
    
    def _load_params_to_model(self, model, params):
        """
        Load extracted parameters into a GaussianModel
        
        Args:
            model: GaussianModel to populate
            params: Dict of parameters
        """
        import torch.nn as nn
        
        model._xyz = nn.Parameter(params['xyz'].requires_grad_(True))
        model._features_dc = nn.Parameter(params['features_dc'].requires_grad_(True))
        model._features_rest = nn.Parameter(params['features_rest'].requires_grad_(True))
        model._scaling = nn.Parameter(params['scaling'].requires_grad_(True))
        model._rotation = nn.Parameter(params['rotation'].requires_grad_(True))
        model._opacity = nn.Parameter(params['opacity'].requires_grad_(True))
        
        # Initialize other required attributes
        model.max_radii2D = torch.zeros((params['xyz'].shape[0]), device="cuda")
        model.active_sh_degree = model.max_sh_degree
    
    def save_decomposed(self, path, iteration):
        """
        Save decomposed models to disk
        
        Args:
            path: Base path for saving
            iteration: Iteration number
        """
        if not self.decomposed:
            raise ValueError("Must decompose first!")
        
        # Create directories
        static_dir = os.path.join(path, f"static_iteration_{iteration}")
        distractor_dir = os.path.join(path, f"distractor_iteration_{iteration}")
        os.makedirs(static_dir, exist_ok=True)
        os.makedirs(distractor_dir, exist_ok=True)
        
        # Save static layer
        self._save_params(self.static_params, os.path.join(static_dir, "params.pth"))
        self._save_ply(self.static_params, os.path.join(static_dir, "point_cloud.ply"))
        
        # Save distractor layer
        self._save_params(self.distractor_params, os.path.join(distractor_dir, "params.pth"))
        self._save_ply(self.distractor_params, os.path.join(distractor_dir, "point_cloud.ply"))
        
        print(f"[TADD-GS] Saved decomposed models to {path}")
    
    def _save_params(self, params, filepath):
        """Save parameters as PyTorch checkpoint"""
        torch.save(params, filepath)
    
    def _save_ply(self, params, filepath):
        """
        Save parameters as PLY file for visualization
        
        Args:
            params: Dict of parameters
            filepath: Output PLY file path
        """
        import numpy as np
        from plyfile import PlyData, PlyElement
        from utils.sh_utils import SH2RGB
        
        xyz = params['xyz'].cpu().numpy()
        normals = np.zeros_like(xyz)
        
        # Convert SH to RGB for visualization
        f_dc = params['features_dc'].detach().cpu().numpy()
        f_rest = params['features_rest'].detach().cpu().numpy()
        
        # Flatten features
        f_dc_flat = f_dc.transpose(0, 2, 1).reshape(f_dc.shape[0], -1)
        f_rest_flat = f_rest.transpose(0, 2, 1).reshape(f_rest.shape[0], -1)
        
        opacities = params['opacity'].cpu().numpy()
        scale = params['scaling'].cpu().numpy()
        rotation = params['rotation'].cpu().numpy()
        
        # Create structured array
        attributes = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(f_dc_flat.shape[1]):
            attributes.append(f'f_dc_{i}')
        for i in range(f_rest_flat.shape[1]):
            attributes.append(f'f_rest_{i}')
        attributes.append('opacity')
        for i in range(scale.shape[1]):
            attributes.append(f'scale_{i}')
        for i in range(rotation.shape[1]):
            attributes.append(f'rot_{i}')
        
        dtype_full = [(attr, 'f4') for attr in attributes]
        
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        concatenated = np.concatenate((xyz, normals, f_dc_flat, f_rest_flat, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, concatenated))
        
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(filepath)


def decompose_scene(gaussians, distractor_scores, threshold=0.7):
    """
    Convenience function for scene decomposition
    
    Args:
        gaussians: GaussianModel
        distractor_scores: Distractor scores [N]
        threshold: Classification threshold
    
    Returns:
        DecomposedGaussianModel with static and distractor layers
    """
    decomposed = DecomposedGaussianModel()
    decomposed.decompose(gaussians, distractor_scores, threshold)
    return decomposed


__all__ = [
    'DecomposedGaussianModel',
    'decompose_scene'
]