import torch
import torch.nn.functional as F

def temporal_contrastive_loss(feat_t, feat_t1, tau=0.07):
    sim = torch.mm(feat_t, feat_t1.T) / tau
    labels = torch.arange(feat_t.shape[0], device=feat_t.device)
    return F.cross_entropy(sim, labels)

def distractor_regularization_loss(variance, thresh=0.5):
    score = torch.sigmoid(variance / thresh)
    return (score * (1 - score)).mean()