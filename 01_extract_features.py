
import argparse
import sys

import numpy as np
import torch

import paths
from brainenc.video import resample_frames, to_videomae

from brainenc.backbones import (EARLY_DIM, N_TOKENS, TOKEN_DIM, VENTRAL_DIM,
                                OpenCLIPTower, VGGPerFrame, load_videomae)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True, choices=["early", "ventral", "dorsal"])
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()

    paths.ensure_dirs()
    device = f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu"
    dst = {"early": paths.early_features, "ventral": paths.ventral_features,
           "dorsal": paths.dorsal_features}[a.stream](a.split)
    done = dst.with_suffix(dst.suffix + ".done")
    if done.exists():
        print(f"already cached: {dst}")
        return

    src = paths.stimuli(a.split)
    if not src.exists():
        sys.exit(f"stimulus file not found: {src}\nset CC2017_DATA")
    clips = torch.load(src, map_location="cpu", mmap=True)
    N = len(clips)

    if a.stream == "ventral":
        tower = OpenCLIPTower(device)
        out = np.lib.format.open_memmap(dst, mode="w+", dtype=np.float32,
                                        shape=(N, VENTRAL_DIM))
        for i in range(0, N, a.batch):
            v = torch.stack([resample_frames(clips[j].float())
                             for j in range(i, min(i + a.batch, N))]).to(device)
            b, t = v.shape[:2]
            feats = tower(v.reshape(b * t, *v.shape[2:])).reshape(b, t, -1)
            out[i:i + b] = feats.mean(1).float().cpu().numpy()    # fixed mean over frames
            if (i // a.batch) % 100 == 0:
                print(f"  {i}/{N}")
    elif a.stream == "early":
        model = VGGPerFrame(device).to(device)
        out = np.lib.format.open_memmap(dst, mode="w+", dtype=np.float16,
                                        shape=(N, paths.N_FRAMES, EARLY_DIM))
        for i in range(0, N, a.batch):
            v = torch.stack([resample_frames(clips[j].float())
                             for j in range(i, min(i + a.batch, N))])
            b, t = v.shape[:2]
            f = model(v.reshape(b * t, *v.shape[2:]).to(device))
            out[i:i + b] = f.reshape(b, t, EARLY_DIM).half().cpu().numpy()
            if (i // a.batch) % 100 == 0:
                print(f"  {i}/{N}")
    else:
        model = load_videomae(device)
        out = np.lib.format.open_memmap(dst, mode="w+", dtype=np.float16,
                                        shape=(N, N_TOKENS, TOKEN_DIM))
        for i in range(0, N, 16):
            v = clips[i:i + 16].float().to(device)
            h = model(pixel_values=to_videomae(v)).last_hidden_state
            assert h.shape[1:] == (N_TOKENS, TOKEN_DIM), h.shape
            out[i:i + len(v)] = h.half().cpu().numpy()
            if (i // 16) % 50 == 0:
                print(f"  {i}/{N}")

    out.flush()
    del out
    done.touch()
    print(f"written {dst}")


if __name__ == "__main__":
    main()
