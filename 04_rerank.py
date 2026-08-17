
import argparse
import glob
import os
import re
import sys

import numpy as np
import torch

import paths
from brainenc.backbones import OpenCLIPTower, VGGPerFrame, load_videomae
from brainenc.metrics import noise_ceiling
from brainenc.pooling import FactorizedAttnPool, TemporalAttnPool
from brainenc.ridge import fit_ridge, fit_ridge_dual
from brainenc.rois import load_roi_masks
from brainenc.scoring import STREAMS, score_candidates
from brainenc.video import resample_frames, to_videomae

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- candidate clips ----------
def load_clip(path, n_frames=paths.N_FRAMES):
    """-> (T, 3, H, W) float in [0, 1].

    For .gif the file is assumed to hold [reference | reconstruction] side by side and only the
    RIGHT half is taken; this is the layout the evaluation code of this line of work uses, and
    keeping it means the pixels scored here are exactly the pixels the metrics see.
    """
    if path.endswith(".npy"):
        a = np.load(path)
    else:
        import imageio.v3 as iio
        a = iio.imread(path, index=None)
        a = a[..., :3] if a.ndim == 4 and a.shape[-1] == 4 else a
        a = np.split(a, 2, axis=2)[1]                       # right half = reconstruction
    t = torch.from_numpy(np.ascontiguousarray(a))
    if t.ndim == 4 and t.shape[-1] == 3:                    # (T,H,W,3) -> (T,3,H,W)
        t = t.permute(0, 3, 1, 2)
    t = t.float()
    if t.max() > 1.5:
        t = t / 255.0
    return resample_frames(t, n_frames)


def candidate_files(subject, idx):
    """{seed: path} for one stimulus."""
    out = {}
    for f in glob.glob(str(paths.candidates(subject) / f"sample_{idx}" / "*")):
        m = re.search(r"seed[_-]?(\d+)\.(gif|npy)$", os.path.basename(f))
        if m and os.path.getsize(f) > 0:
            out[int(m.group(1))] = f
    return out


@torch.no_grad()
def extract(subject, start, end):
    """Embed every candidate with the same backbones + poolers used for the stimuli."""
    dst = paths.candidate_features(subject)
    dst.mkdir(parents=True, exist_ok=True)

    vgg = VGGPerFrame(DEV).to(DEV).eval()
    early_pool = TemporalAttnPool().to(DEV).eval()
    early_pool.load_state_dict(torch.load(paths.pooler(subject, "early"), map_location=DEV))
    clip = OpenCLIPTower(DEV)
    vmae = load_videomae(DEV)
    dorsal_pool = FactorizedAttnPool(C=768, num_queries=1, num_heads=4).to(DEV).eval()
    dorsal_pool.load_state_dict(torch.load(paths.pooler(subject, "dorsal"), map_location=DEV))

    done = 0
    for idx in range(start, end):
        out_file = dst / f"sample_{idx}.npz"
        if out_file.exists():
            continue
        files = candidate_files(subject, idx)
        if not files:
            continue

        seeds, early, ventral, dorsal = [], [], [], []
        for seed in sorted(files):
            try:
                clip_t = load_clip(files[seed]).to(DEV)      # (T, 3, H, W)
            except Exception as e:                            # a corrupt file must not stop the run
                print(f"  [skip] sample_{idx} seed {seed}: {type(e).__name__}")
                continue
            v = clip_t[None]                                  # (1, T, 3, H, W)
            per_frame = vgg(clip_t)[None]                     # (1, T, EARLY_DIM)
            early.append(early_pool(per_frame)[0].half().cpu())
            ventral.append(clip(clip_t).mean(0).half().cpu())  # fixed mean over frames
            tokens = vmae(pixel_values=to_videomae(v)).last_hidden_state.float()
            dorsal.append(dorsal_pool(tokens)[0].half().cpu())
            seeds.append(seed)

        if not seeds:
            continue
        np.savez(out_file, seeds=np.array(seeds),
                 early=torch.stack(early).numpy(),
                 ventral=torch.stack(ventral).numpy(),
                 dorsal=torch.stack(dorsal).numpy())
        done += 1
        if done % 20 == 0:
            print(f"  {idx}: {done} stimuli written")
    print(f"features for {done} stimuli -> {dst}")

def fit_encoders(subject):
    """The three encoders of step 03, plus the measured test responses and the voxel weights."""
    masks = load_roi_masks(paths.roi_map(subject))
    fmri_train = torch.load(paths.fmri(subject, "train"), map_location="cpu").float().mean(1)
    raw_test = torch.load(paths.fmri(subject, "test"), map_location="cpu").float()
    nc_all, fmri_test = noise_ceiling(raw_test), raw_test.mean(1)

    # stimulus features, learned pooling applied where there is one
    ep = TemporalAttnPool().to(DEV).eval()
    ep.load_state_dict(torch.load(paths.pooler(subject, "early"), map_location=DEV))
    tr = torch.from_numpy(np.load(paths.early_features("train"), mmap_mode="r")[:])
    X = {"early": torch.cat([ep(tr[i:i + 64].float().to(DEV)).cpu()
                             for i in range(0, len(tr), 64)])}
    del tr, ep

    dp = FactorizedAttnPool(C=768, num_queries=1, num_heads=4).to(DEV).eval()
    dp.load_state_dict(torch.load(paths.pooler(subject, "dorsal"), map_location=DEV))
    tr = torch.from_numpy(np.load(paths.dorsal_features("train"), mmap_mode="r")[:])
    X["dorsal"] = torch.cat([dp(tr[i:i + 64].float().to(DEV)).cpu()
                             for i in range(0, len(tr), 64)])
    del tr, dp
    X["ventral"] = torch.from_numpy(np.load(paths.ventral_features("train"))).float()

    encoders, measured, nc_weights = {}, {}, {}
    for stream, dual in (("early", True), ("ventral", False), ("dorsal", False)):
        mask = masks[stream]
        Y = fmri_train[:, mask]
        mu, sd = Y.mean(0), Y.std(0) + 1e-6
        encoders[stream] = (fit_ridge_dual if dual else fit_ridge)(X[stream], (Y - mu) / sd)
        measured[stream] = (fmri_test[:, mask] - mu) / sd     # same normalisation as the target
        nc_weights[stream] = nc_all[mask]
        print(f"  encoder {stream:>7} ready ({int(mask.sum())} voxels)")
    return encoders, measured, nc_weights

def rank(subject, method, weights, tag):
    src = paths.candidate_features(subject)
    files = sorted(glob.glob(str(src / "sample_*.npz")),
                   key=lambda p: int(re.search(r"sample_(\d+)", p).group(1)))
    if not files:
        sys.exit(f"no candidate features in {src}\nrun: 04_rerank.py extract --subject {subject}")
    print(f"{len(files)} stimuli | method {method} | weights "
          + "/".join(f"{weights[s]:g}" for s in STREAMS))

    encoders, measured, nc_weights = fit_encoders(subject)

    stimuli, chosen, margin = [], [], []
    for f in files:
        idx = int(re.search(r"sample_(\d+)", f).group(1))
        z = np.load(f)
        feats = {s: z[s] for s in STREAMS}
        score, _ = score_candidates(feats, encoders, {s: measured[s][idx] for s in STREAMS},
                                    nc_weights, weights, method)
        order = np.argsort(-score)
        stimuli.append(idx)
        chosen.append(int(z["seeds"][order[0]]))
        # gap between the winner and the runner-up, on the z-scored scale: a diagnostic of how
        # decided the choice was, not part of the selection
        margin.append(float(score[order[0]] - score[order[1]]) if len(order) > 1 else np.nan)

    dst = paths.selection(subject, tag)
    np.savez(dst, stimulus=np.array(stimuli), seed=np.array(chosen),
             margin=np.array(margin), method=method,
             weights=np.array([weights[s] for s in STREAMS]))
    print(f"\nselected {len(stimuli)} candidates -> {dst}")
    print(f"median winner-runner-up margin: {np.nanmedian(margin):.4f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    e = sub.add_parser("extract", help="embed the candidate clips")
    e.add_argument("--subject", type=int, required=True, choices=list(paths.SUBJECTS))
    e.add_argument("--start", type=int, default=0)
    e.add_argument("--end", type=int, default=paths.N_TEST)

    r = sub.add_parser("rank", help="score the candidates and pick one per stimulus")
    r.add_argument("--subject", type=int, required=True, choices=list(paths.SUBJECTS))
    r.add_argument("--method", default="M2", choices=["M1", "M2"])
    r.add_argument("--we", type=float, default=1.0, help="early weight")
    r.add_argument("--wv", type=float, default=1.0, help="ventral weight")
    r.add_argument("--wd", type=float, default=1.0, help="dorsal weight")
    r.add_argument("--tag", default="M2_uniform")

    a = ap.parse_args()
    paths.ensure_dirs()
    for stream in ("early", "dorsal"):
        if not paths.pooler(a.subject, stream).exists():
            sys.exit(f"missing pooling checkpoint for {stream}\nrun 02_train_poolers.py first")

    if a.mode == "extract":
        extract(a.subject, a.start, a.end)
    else:
        rank(a.subject, a.method,
             {"early": a.we, "ventral": a.wv, "dorsal": a.wd}, a.tag)


if __name__ == "__main__":
    main()
