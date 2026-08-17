"""The three frozen visual backbones, in one place.

Both the stimulus features (step 01) and the candidate features (step 04) go through these, so
they live here rather than in either script: encoder and candidates MUST be embedded by the same
backbone, with the same input geometry and the same normalisation. If they diverge, the ridge is
applied to features from a space it never saw and the resulting scores are meaningless without
anything visibly breaking.
"""
import os

import torch
import torch.nn.functional as F
import torchvision

# --- early: VGG19-BN, ReLU layers relu1_2, relu2_2, relu3_3, relu4_2, relu5_1 ----------------
VGG_LAYERS = [5, 12, 22, 32, 42]
VGG_CHANNELS = [64, 128, 256, 512, 512]
GRID = 7
VGG_INPUT = 112
EARLY_DIM = sum(c * GRID * GRID for c in VGG_CHANNELS)      # 72128

# --- dorsal: VideoMAE base ------------------------------------------------------------------
VIDEOMAE_NAME = "MCG-NJU/videomae-base"
N_TOKENS, TOKEN_DIM = 1568, 768                              # 8 temporal x 14x14 spatial

# --- ventral: OpenCLIP ViT-bigG-14 ----------------------------------------------------------
VENTRAL_DIM = 1280

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


class VGGPerFrame(torch.nn.Module):
    """Multi-layer VGG19-BN features on a 7x7 grid, ONE VECTOR PER FRAME."""

    def __init__(self, device, size=VGG_INPUT):
        super().__init__()
        weights = torchvision.models.VGG19_BN_Weights.IMAGENET1K_V1
        vgg = torchvision.models.vgg19_bn(weights=weights)
        self.feats = vgg.features[:max(VGG_LAYERS) + 1].eval()
        for p in self.parameters():
            p.requires_grad_(False)
        self.size = size
        self.mean, self.std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)

    @torch.no_grad()
    def forward(self, x):                                    # (B, 3, H, W) in [0, 1]
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        outs, h = [], x
        for i, layer in enumerate(self.feats):
            h = layer(h)
            if i in VGG_LAYERS:
                outs.append(F.adaptive_avg_pool2d(h, GRID).flatten(1))
        return torch.cat(outs, 1)                            # (B, EARLY_DIM)


class OpenCLIPTower(torch.nn.Module):
    """ViT-bigG-14 vision tower, final projection. Set OPENCLIP_WEIGHTS for a local checkpoint."""

    def __init__(self, device):
        super().__init__()
        import open_clip
        ckpt = os.environ.get("OPENCLIP_WEIGHTS", "laion2b_s39b_b160k")
        model, _, _ = open_clip.create_model_and_transforms("ViT-bigG-14", pretrained=ckpt)
        self.visual = model.visual.to(device).eval()
        for p in self.parameters():
            p.requires_grad_(False)
        self.mean, self.std = CLIP_MEAN.to(device), CLIP_STD.to(device)

    @torch.no_grad()
    def forward(self, x):                                    # (B, 3, H, W) in [0, 1]
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return self.visual((x - self.mean) / self.std)       # (B, VENTRAL_DIM)


def load_videomae(device):
    from transformers import VideoMAEModel
    model = VideoMAEModel.from_pretrained(VIDEOMAE_NAME).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
