# CLIP-GS: Semantic-Guided 3D Gaussian Splatting for Transient Object Removal

**A lightweight semantic filtering framework that removes transient objects (people, hands, moving items) from casual multi-view captures in 3D Gaussian Splatting using CLIP-based category-aware scoring and pruning — without motion heuristics or heavy scene decomposition.**

## 📄 Paper

**arXiv**: [https://arxiv.org/abs/2602.15516](https://arxiv.org/abs/2602.15516)  
*(Replace with actual arXiv link when available)*

**Abstract** (short): Transient objects cause ghosting artifacts in 3DGS. We propose CLIP-guided semantic filtering that accumulates per-Gaussian distractor scores across training iterations and suppresses them via opacity regularization + periodic pruning. Semantic classification resolves parallax ambiguity better than motion/visibility methods. Experiments on RobustNeRF show consistent gains over vanilla 3DGS while preserving real-time rendering and minimal memory overhead.

---

## How It Works

CLIP-GS extends standard 3D Gaussian Splatting with a semantic filtering branch:

1. **Render** each training view from the current set of 3D Gaussians.
2. **CLIP Scoring**: Encode the rendered image with CLIP ViT-B/32 and compute maximum cosine similarity against a set of distractor text prompts (e.g., “a photo of a person”, “a photo of pedestrians”, “a photo of hands”).
3. **Per-Gaussian Accumulation**: For every visible Gaussian in high-distractor-score views, accumulate a semantic score. Normalize by view count so the score reflects category consistency rather than observation frequency.
4. **Suppression**:
   - Add a semantic regularization term to the photometric loss that penalizes opacity of high semantic-score Gaussians.
   - Periodically prune Gaussians whose normalized semantic score exceeds a calibrated threshold (τ ≈ 0.015–0.02).

This removes ghosting from transients while correctly preserving static geometry — even elements visible in very few views (e.g. walls seen in only 15% of images) are retained when semantically classified as “building”.

---

## Architecture / Pipeline

```
Multi-view Images (with transients)
          ↓
3D Gaussian Splatting (differentiable rasterization)
          ↓
Render Training View
          ↓
CLIP ViT-B/32 Vision Encoder
          ↓
Compute distractor similarity (max over prompts)
          ↓
Per-Gaussian Score Accumulation + Normalization
          ↓
Opacity Regularization (in loss)  +  Periodic Pruning
          ↓
Clean Static 3D Gaussians (no ghosting)
          ↓
Real-time Novel View Synthesis
```

See **Figure 1** in the paper for the detailed pipeline diagram.

**Distractor Prompts Example** (RobustNeRF):
```python
D = ["a photo of a person", "a photo of people", 
     "a photo of pedestrians", "a photo of hands", "a photo of a balloon"]
```

---

## Results on RobustNeRF Benchmark

Evaluated on four sequences (Statue, Android, Yoda, Crab(2)) under identical training settings.

### Quantitative Results

| Method            | Statue PSNR↑ | Android PSNR↑ | Yoda PSNR↑ | Crab(2) PSNR↑ | Memory Overhead | Rendering |
|-------------------|--------------|---------------|------------|---------------|-----------------|-----------|
| Vanilla 3DGS     | 20.04       | 25.20        | 26.20     | 24.50        | Baseline       | Real-time |
| Mip-NeRF 360     | 19.74       | 25.80        | 26.12     | 25.80        | High           | Slow      |
| **CLIP-GS (Ours)** | **21.98**   | **26.12**    | **26.80** | **24.18**    | **Minimal**    | **Real-time** |

**Highlights**:
- Up to **+1.94 dB PSNR** improvement over vanilla 3DGS (Statue sequence)
- Consistent gains in SSIM and LPIPS
- Only **3.8%** of Gaussians pruned at the optimal threshold
- Both opacity regularization and periodic pruning contribute complementary gains
- Even generic prompts (“person”) deliver strong improvements (+0.7 dB)

### Qualitative Results
Vanilla 3DGS and Mip-NeRF 360 produce noticeable ghosting from walking people and moving objects.  
**CLIP-GS** cleanly removes these transients while preserving fine static details and scene boundaries.

See **Figure 2** in the paper for visual comparisons on held-out test views.

---

*This work is currently being extended into real-world applictations*
**Keywords**: 3D Gaussian Splatting · Transient Object Removal · CLIP · Vision-Language Models · Semantic Filtering · Neural Rendering · Ghosting Artifacts
