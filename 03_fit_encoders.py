import argparse
import sys

import numpy as np
import torch

import paths
from brainenc.metrics import noise_ceiling, r_vox, retr_top1
from brainenc.pooling import FactorizedAttnPool, TemporalAttnPool
from brainenc.ridge import fit_ridge, fit_ridge_dual
from brainenc.rois import load_roi_masks

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def apply_pool(model, X, batch=64):
    return torch.cat([model(X[i:i + batch].float().to(DEV)).cpu()
                      for i in range(0, len(X), batch)])


def load_features(subject):
    """{stream: (train, test)} with the learned pooling already applied where it exists."""
    out = {}

    ep = TemporalAttnPool().to(DEV).eval()
    ep.load_state_dict(torch.load(paths.pooler(subject, "early"), map_location=DEV))
    tr = torch.from_numpy(np.load(paths.early_features("train"), mmap_mode="r")[:])
    te = torch.from_numpy(np.load(paths.early_features("test"), mmap_mode="r")[:])
    out["early"] = (apply_pool(ep, tr), apply_pool(ep, te))
    del tr, te, ep
    torch.cuda.empty_cache() if DEV.startswith("cuda") else None

    dp = FactorizedAttnPool(C=768, num_queries=1, num_heads=4).to(DEV).eval()
    dp.load_state_dict(torch.load(paths.pooler(subject, "dorsal"), map_location=DEV))
    tr = torch.from_numpy(np.load(paths.dorsal_features("train"), mmap_mode="r")[:])
    te = torch.from_numpy(np.load(paths.dorsal_features("test"), mmap_mode="r")[:])
    out["dorsal"] = (apply_pool(dp, tr), apply_pool(dp, te))
    del tr, te, dp
    torch.cuda.empty_cache() if DEV.startswith("cuda") else None

    # ventral: no learned pooling, the frame mean was already applied at extraction time
    out["ventral"] = (
        torch.from_numpy(np.load(paths.ventral_features("train"))).float(),
        torch.from_numpy(np.load(paths.ventral_features("test"))).float(),
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, required=True, choices=list(paths.SUBJECTS))
    a = ap.parse_args()

    roi = paths.roi_map(a.subject)
    if not roi.exists():
        sys.exit(f"ROI map not found: {roi}\nrun 00_build_roi_masks.py first")
    for stream in ("early", "dorsal"):
        if not paths.pooler(a.subject, stream).exists():
            sys.exit(f"missing pooling checkpoint for {stream}\nrun 02_train_poolers.py first")

    masks = load_roi_masks(roi)
    fmri_train = torch.load(paths.fmri(a.subject, "train"), map_location="cpu").float().mean(1)
    raw_test = torch.load(paths.fmri(a.subject, "test"), map_location="cpu").float()
    nc_all, fmri_test = noise_ceiling(raw_test), raw_test.mean(1)

    X = load_features(a.subject)

    print(f"\nsubject {a.subject} - encoders evaluated on {len(fmri_test)} held-out stimuli")
    print(f"{'stream':>9} {'dim':>7} {'voxels':>7} {'NC':>6} {'alpha':>9} | {'r_vox':>8} {'retr':>8}")
    print("-" * 66)
    for stream, dual in (("early", True), ("ventral", False), ("dorsal", False)):
        mask = masks[stream]
        Y = fmri_train[:, mask]
        mu, sd = Y.mean(0), Y.std(0) + 1e-6
        nc = nc_all[mask]
        predict = (fit_ridge_dual if dual else fit_ridge)(X[stream][0], (Y - mu) / sd)
        pred = predict(X[stream][1])
        true = (fmri_test[:, mask] - mu) / sd
        print(f"{stream:>9} {X[stream][0].shape[1]:>7} {int(mask.sum()):>7} {nc.mean():>6.3f} "
              f"{predict.alpha:>9.0e} | {r_vox(pred, true, nc):>+8.4f} "
              f"{100 * retr_top1(pred, true, nc):>7.2f}%")

    

if __name__ == "__main__":
    main()
