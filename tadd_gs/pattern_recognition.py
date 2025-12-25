"""
TADD-GS Pattern Detection Module
Detects constant movement patterns (cyclic, linear) vs transient motion
"""

import torch
import torch.nn.functional as F
from collections import deque

class ConstantMoverDetector:
    """
    Detects constant movement patterns using temporal window analysis
    
    Based on your framework:
    - Distinguishes constant movers (repeating patterns) from transient distractors
    - Uses Fourier analysis for cyclic motion detection
    - Uses trajectory analysis for linear motion detection
    """
    def __init__(self, window_size=10, device='cuda'):
        """
        Args:
            window_size: Number of frames to analyze (5-10 recommended)
            device: Device to store tensors
        """
        self.window_size = window_size
        self.device = device
        
        # Store motion history as deque for efficiency
        self.motion_history = deque(maxlen=window_size)
        self.position_history = deque(maxlen=window_size)
    
    def update(self, motion, positions):
        """
        Update motion history with new frame
        
        Args:
            motion: Motion magnitude for each Gaussian [N]
            positions: 3D positions of Gaussians [N, 3]
        """
        self.motion_history.append(motion.detach().clone())
        self.position_history.append(positions.detach().clone())
    
    def detect_patterns(self):
        """
        Detect constant movement patterns
        
        Returns:
            cyclic_score: Score for cyclic motion [N]
            linear_score: Score for linear motion [N]
            constant_mover_score: Combined score [N]
        """
        if len(self.motion_history) < self.window_size:
            # Not enough history yet
            n_gaussians = self.motion_history[0].shape[0] if len(self.motion_history) > 0 else 0
            zero = torch.zeros(n_gaussians, device=self.device)
            return zero, zero, zero
        
        # Stack history into tensors [window_size, N]
        motion_tensor = torch.stack(list(self.motion_history), dim=0)
        position_tensor = torch.stack(list(self.position_history), dim=0)
        
        # Detect cyclic patterns
        cyclic_score = self._detect_cyclic(motion_tensor)
        
        # Detect linear trajectories
        linear_score = self._detect_linear(position_tensor)
        
        # Combine scores
        constant_mover_score = torch.max(cyclic_score, linear_score)
        
        return cyclic_score, linear_score, constant_mover_score
    
    def _detect_cyclic(self, motion_tensor):
        """
        Detect cyclic/repeating motion patterns using FFT
        
        Args:
            motion_tensor: [window_size, N] motion history
        
        Returns:
            cyclic_score: [N] score for each Gaussian (0-1)
        """
        # Apply FFT along time dimension
        fft = torch.fft.fft(motion_tensor, dim=0)
        power = torch.abs(fft) ** 2
        
        # Ignore DC component (index 0)
        power = power[1:, :]
        
        # Find peaks in frequency spectrum
        # High peaks indicate periodic motion
        max_power = power.max(dim=0)[0]
        mean_power = power.mean(dim=0)
        
        # Ratio of max to mean indicates how periodic the motion is
        periodicity = max_power / (mean_power + 1e-6)
        
        # Normalize to 0-1
        cyclic_score = torch.sigmoid((periodicity - 2.0) / 2.0)  # Threshold at 2.0
        
        return cyclic_score
    
    def _detect_linear(self, position_tensor):
        """
        Detect linear trajectories (consistent direction)
        
        Args:
            position_tensor: [window_size, N, 3] position history
        
        Returns:
            linear_score: [N] score for each Gaussian (0-1)
        """
        # Compute displacement vectors between consecutive frames
        displacements = position_tensor[1:] - position_tensor[:-1]  # [window_size-1, N, 3]
        
        # Compute average displacement direction
        avg_direction = displacements.mean(dim=0)  # [N, 3]
        avg_magnitude = avg_direction.norm(dim=-1)  # [N]
        
        # Normalize
        avg_direction = F.normalize(avg_direction + 1e-8, dim=-1)
        
        # Compute consistency: how aligned are individual displacements with average?
        normalized_displacements = F.normalize(displacements + 1e-8, dim=-1)
        
        # Dot product with average direction
        alignments = (normalized_displacements * avg_direction.unsqueeze(0)).sum(dim=-1)  # [window_size-1, N]
        
        # Average alignment (high = consistent linear motion)
        consistency = alignments.mean(dim=0)  # [N]
        
        # Linear score: high consistency + significant magnitude
        magnitude_score = torch.sigmoid((avg_magnitude - 0.01) / 0.01)  # Threshold at 0.01
        linear_score = consistency * magnitude_score
        
        # Clamp to 0-1
        linear_score = torch.clamp(linear_score, 0.0, 1.0)
        
        return linear_score
    
    def get_transient_score(self, motion_variance, constant_mover_score):
        """
        Distinguish transient distractors from constant movers
        
        Args:
            motion_variance: Total motion variance [N]
            constant_mover_score: Score for constant movement [N]
        
        Returns:
            transient_score: Score for transient motion [N]
        """
        # Normalize motion variance to 0-1
        max_var = motion_variance.max()
        if max_var > 0:
            normalized_var = motion_variance / max_var
        else:
            normalized_var = motion_variance
        
        # Transient = has motion but not constant pattern
        transient_score = normalized_var * (1.0 - constant_mover_score)
        
        return transient_score


def detect_constant_movers(motion_history, position_history, window_size=10):
    """
    Convenience function for pattern detection
    
    Args:
        motion_history: List of motion tensors [N] for each frame
        position_history: List of position tensors [N, 3] for each frame
        window_size: Analysis window size
    
    Returns:
        cyclic_score: Cyclic motion scores [N]
        linear_score: Linear motion scores [N]
        constant_mover_score: Combined constant mover scores [N]
    """
    if len(motion_history) < window_size:
        n = motion_history[0].shape[0] if len(motion_history) > 0 else 0
        zero = torch.zeros(n, device=motion_history[0].device if len(motion_history) > 0 else 'cuda')
        return zero, zero, zero
    
    # Take last window_size frames
    motion_window = motion_history[-window_size:]
    position_window = position_history[-window_size:]
    
    # Stack into tensors
    motion_tensor = torch.stack(motion_window, dim=0)
    position_tensor = torch.stack(position_window, dim=0)
    
    # Create detector and analyze
    detector = ConstantMoverDetector(window_size=window_size, device=motion_tensor.device)
    detector.motion_history = deque(motion_window, maxlen=window_size)
    detector.position_history = deque(position_window, maxlen=window_size)
    
    return detector.detect_patterns()


__all__ = [
    'ConstantMoverDetector',
    'detect_constant_movers'
]