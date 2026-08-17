
import argparse
import csv
import sys
from collections import Counter

import numpy as np
import torch
import nibabel as nib

import paths
from brainenc.rois import SCHEME_A, area_name, load_roi_masks

N_VERTICES = 59412        # cortical grayordinates, fs_LR_32k
N_SEGMENTS = 18           # training segments
TRIM_HEAD, TRIM_TAIL = 1, 4
DEV = "cuda" if torch.cuda.is_available() else "cpu"

DLABEL = paths.ATLAS / "CAB-NP_parcels_LR.dlabel.nii"
LABELKEY = (paths.ATLAS /
            "CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_LabelKey.txt")


def detrend_poly(x, deg=4):
    """Remove a degree-`deg` polynomial from every column of x (T x V)."""
    T = x.shape[0]
    t = torch.linspace(-1, 1, T, device=x.device, dtype=torch.double)
    V = torch.stack([t ** k for k in range(deg + 1)], 1)
    P = V @ torch.linalg.pinv(V)
    return x - P @ x


def zscore(x):
    return (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + 1e-8)


def load_cortex_train(subject_dir):
    """Concatenate the 18 preprocessed training segments -> (18*240, 59412)."""
    out = []
    for seg in range(1, N_SEGMENTS + 1):
        f = subject_dir / "fmri" / f"seg{seg}" / "cifti" / f"seg{seg}_1_Atlas.dtseries.nii"
        if not f.exists():
            sys.exit(f"missing CIFTI segment: {f}")
        x = np.asarray(nib.load(str(f)).get_fdata())[:, :N_VERTICES]
        x = torch.from_numpy(x).to(DEV).double()
        x = x[TRIM_HEAD:-TRIM_TAIL]
        out.append(zscore(detrend_poly(x, 4)).float())
        print(f"  seg{seg:>2}: {tuple(out[-1].shape)}")
    return torch.cat(out)


def glasser_lookup():
    """Per-vertex Glasser area name.

    The label table stored INSIDE the dlabel uses CAB-NP names (e.g. Visual2-34_L-Ctx), not
    Glasser names, so the names are taken from the GLASSERLABELNAME column of the LabelKey file.
    """
    for f in (DLABEL, LABELKEY):
        if not f.exists():
            sys.exit(f"missing atlas file: {f}\nset GLASSER_ATLAS to the CAB-NP directory")
    keys = np.asarray(nib.load(str(DLABEL)).get_fdata()).ravel().astype(int)[:N_VERTICES]
    rows = list(csv.reader(open(LABELKEY), delimiter="\t"))
    hdr = rows[0]
    ki, gi = hdr.index("KEYVALUE"), hdr.index("GLASSERLABELNAME")
    key2name = {int(r[ki]): r[gi] for r in rows[1:]}
    names = np.array([key2name.get(int(k), "???") for k in keys])
    return keys, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, required=True, choices=list(paths.SUBJECTS))
    ap.add_argument("--check", action="store_true",
                    help="re-derive and compare with the stored map instead of overwriting it")
    a = ap.parse_args()

    subject_dir = paths.RAW / f"subject{a.subject}"
    if not subject_dir.is_dir():
        sys.exit(f"raw CIFTI directory not found: {subject_dir}\nset CC2017_RAW")

    print(f"[1] cortical training timeseries from {subject_dir}")
    C = load_cortex_train(subject_dir)
    Cz = torch.nn.functional.normalize(zscore(C), dim=0)

    print(f"[2] preprocessed fMRI, subject {a.subject}")
    ft = torch.load(paths.fmri(a.subject, "train"), map_location="cpu")
    nc = ft[:, 0, :].to(DEV)                       # repeat 0 corresponds to the _1 CIFTI files
    ncz = torch.nn.functional.normalize(zscore(nc), dim=0)
    n_vox = ncz.shape[1]
    print(f"    {n_vox} voxels, {ncz.shape[0]} timepoints (expected {N_SEGMENTS * 240})")

    print(f"[3] matching {n_vox} voxels against {N_VERTICES} vertices")
    vertex_idx = np.zeros(n_vox, np.int32)
    match_corr = np.zeros(n_vox, np.float32)
    for i in range(0, n_vox, 2000):
        sim = ncz[:, i:i + 2000].t() @ Cz
        mx, am = sim.max(1)
        vertex_idx[i:i + 2000] = am.cpu().numpy()
        match_corr[i:i + 2000] = mx.cpu().numpy()
    print(f"    correlation: mean={match_corr.mean():.5f}  min={match_corr.min():.5f}  "
          f">0.999: {100 * (match_corr > 0.999).mean():.1f}%")
    if match_corr.mean() < 0.99:
        print("    WARNING: the preprocessing does not match; do not trust these areas")

    parcels, names = glasser_lookup()
    glasser = names[vertex_idx]
    out = {"vertex_idx": vertex_idx, "match_corr": match_corr, "glasser": glasser,
           "parcel": parcels[vertex_idx],
           "hemi": np.array([str(g)[0] for g in glasser])}

    dst = paths.roi_map(a.subject)
    if a.check:
        if not dst.exists():
            sys.exit(f"nothing to compare against: {dst} does not exist")
        ref = np.load(dst, allow_pickle=True)
        agree = (ref["glasser"].astype(str) == glasser.astype(str)).mean()
        print(f"\n[check] label agreement with stored map: {100 * agree:.2f}%")
        print(f"[check] identical vertex indices: "
              f"{100 * (ref['vertex_idx'] == vertex_idx).mean():.2f}%")
        return 0 if agree > 0.999 else 1

    np.savez(dst, **out)
    print(f"\nwritten {dst}")
    print(f"most frequent areas: {Counter(area_name(g) for g in glasser).most_common(8)}")

    masks = load_roi_masks(dst)
    print("\nstream masks (Scheme A)")
    for stream in SCHEME_A:
        print(f"  {stream:<8}{int(masks[stream].sum()):>6} voxels")
    print(f"  {'other':<8}{int(masks['other'].sum()):>6} voxels (not used)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
