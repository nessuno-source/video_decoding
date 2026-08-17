
import torch
import torch.nn.functional as F

N_FRAMES = 6


def resample_frames(v, n=N_FRAMES):
    """(T, C, H, W) -> (n, C, H, W) by selecting the nearest frames. No interpolation.

    Used only to guard against clips that are not already at n frames; when T == n it is a
    no-op, which is the normal case for this dataset.
    """
    if v.shape[0] == n:
        return v
    idx = torch.linspace(0, v.shape[0] - 1, n).round().long()
    return v[idx]


def to_videomae(v):
    """(B, T, C, H, W) -> (B, 16, C, 224, 224), the input VideoMAE expects.

    Two steps: bilinear in space if the frames are not already 224x224, then trilinear in time
    from T to 16. Note that the temporal step INTERPOLATES: going 6 -> 16 does not add
    information, it resamples the six frames onto a finer grid.
    """
    B, T, C, H, W = v.shape
    if (H, W) != (224, 224):
        v = F.interpolate(v.reshape(B * T, C, H, W), size=(224, 224),
                          mode="bilinear", align_corners=False).reshape(B, T, C, 224, 224)
    if T == 16:
        return v
    x = v.permute(0, 2, 1, 3, 4).float()
    x = F.interpolate(x, size=(16, 224, 224), mode="trilinear", align_corners=False)
    return x.permute(0, 2, 1, 3, 4)
