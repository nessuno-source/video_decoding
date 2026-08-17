import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F

import paths
from brainenc.metrics import noise_ceiling
from brainenc.pooling import FactorizedAttnPool, TemporalAttnPool
from brainenc.rois import load_roi_masks

DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, PATIENCE = 120, 6
LR, WEIGHT_DECAY, CLIP = 3e-4, 1e-2, 1.0
VAL_FRAC = 0.12
BATCH = {"early": 32, "dorsal": 64}


def build(stream, seed):
    torch.manual_seed(seed)
    if stream == "dorsal":
        return FactorizedAttnPool(C=768, num_queries=1, num_heads=4).to(DEV), 768
    return TemporalAttnPool().to(DEV), 72128


def take(X, idx):
    return X[torch.as_tensor(np.asarray(idx), device=X.device, dtype=torch.long)].float()


def train(X, Y, stream, seed):
    """Returns the trained pooling module. The head is thrown away on the way out."""
    bs = BATCH[stream]
    n = len(X)
    perm = np.random.default_rng(seed).permutation(n)
    n_val = int(n * VAL_FRAC)
    va, tr = perm[:n_val], perm[n_val:]
    Y_va, Y_tr = Y[va].to(DEV), Y[tr].to(DEV)

    pool, d_out = build(stream, seed)
    head = torch.nn.Linear(d_out, Y.shape[1]).to(DEV)        # discarded after training
    params = list(pool.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    best, best_state, bad = float("inf"), None, 0
    for ep in range(EPOCHS):
        pool.train()
        order = np.random.default_rng(seed * 1000 + ep).permutation(len(tr))
        for i in range(0, len(tr), bs):
            b = tr[order[i:i + bs]]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV.startswith("cuda")):
                loss = F.mse_loss(head(pool(take(X, b))), Y_tr[order[i:i + bs]])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, CLIP)
            opt.step()
        sched.step()

        pool.eval()
        with torch.no_grad():
            vl = float(np.mean([F.mse_loss(head(pool(take(X, va[i:i + 64]))),
                                           Y_va[i:i + 64]).item()
                                for i in range(0, len(va), 64)]))
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in pool.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
        if ep % 10 == 0:
            print(f"  epoch {ep:>3}  val {vl:.5f}  best {best:.5f}")

    pool.load_state_dict(best_state)
    print(f"  stopped at epoch {ep}, best validation loss {best:.5f}")
    return pool.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, required=True, choices=list(paths.SUBJECTS))
    ap.add_argument("--stream", required=True, choices=["early", "dorsal"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    paths.ensure_dirs()

    roi = paths.roi_map(a.subject)
    if not roi.exists():
        sys.exit(f"ROI map not found: {roi}\nrun 00_build_roi_masks.py first")
    mask = load_roi_masks(roi)[a.stream]

    fmri_train = torch.load(paths.fmri(a.subject, "train"), map_location="cpu").float()
    fmri_train = fmri_train.mean(1)[:, mask]                  # average the repeats
    raw_test = torch.load(paths.fmri(a.subject, "test"), map_location="cpu").float()
    nc = noise_ceiling(raw_test)[mask]

    mu, sd = fmri_train.mean(0), fmri_train.std(0) + 1e-6
    Y = (fmri_train - mu) / sd

    src = (paths.early_features("train") if a.stream == "early"
           else paths.dorsal_features("train"))
    if not src.exists():
        sys.exit(f"features not found: {src}\nrun 01_extract_features.py first")
    X = torch.from_numpy(np.load(src, mmap_mode="r")[:]).to(DEV)

    print(f"subject {a.subject} | {a.stream} | {int(mask.sum())} voxels, "
          f"mean noise ceiling {nc.mean():.3f} | features {tuple(X.shape)}")
    pool = train(X, Y, a.stream, a.seed)

    dst = paths.pooler(a.subject, a.stream)
    torch.save(pool.state_dict(), dst)
    print(f"written {dst}")


if __name__ == "__main__":
    main()
