
import torch
import torch.nn.functional as F


def noise_ceiling(fmri_test_raw):
    h1, h2 = fmri_test_raw[:, :5].mean(1), fmri_test_raw[:, 5:].mean(1)
    a, b = h1 - h1.mean(0), h2 - h2.mean(0)
    r = (a * b).sum(0) / (a.norm(dim=0) * b.norm(dim=0) + 1e-9)
    return torch.clamp(r, min=0) ** 2


def wcorr(pred, true, w):
    """Weighted Pearson between each row of pred (n, V) and true (V,), weights w (V,)."""
    w = w / (w.sum() + 1e-9)
    dp = pred - (pred * w).sum(1, keepdim=True)
    dt = true - (true * w).sum()
    return (w * dp * dt).sum(1) / (
        (w * dp * dp).sum(1).sqrt() * (w * dt * dt).sum().sqrt() + 1e-9)


def r_vox(pred, true, w):
    """Per-voxel Pearson across stimuli, averaged with the noise-ceiling weights."""
    p = pred - pred.mean(0, keepdim=True)
    t = true - true.mean(0, keepdim=True)
    r = (p * t).sum(0) / (p.norm(dim=0) * t.norm(dim=0) + 1e-9)
    return float((r * w).sum() / (w.sum() + 1e-9))


def retr_top1(pred, true, w):
    """Does the predicted fMRI identify the right stimulus among all test stimuli? Chance = 1/N."""
    sc = w.sqrt()[None]
    P = F.normalize((pred - (pred * w / w.sum()).sum(1, keepdim=True)) * sc, dim=1)
    T = F.normalize((true - (true * w / w.sum()).sum(1, keepdim=True)) * sc, dim=1)
    sim = P @ T.t()
    n = len(P)
    rank = (sim.argsort(1, descending=True) == torch.arange(n)[:, None]).float().argmax(1)
    return float((rank == 0).float().mean())
