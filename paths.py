import os
from pathlib import Path

REPO = Path(__file__).resolve().parent

DATA = Path(os.environ.get("CC2017_DATA", REPO / "data"))
RAW = Path(os.environ.get("CC2017_RAW", REPO / "data" / "raw"))
ATLAS = Path(os.environ.get("GLASSER_ATLAS", REPO / "data" / "atlas_glasser"))
CACHE = Path(os.environ.get("BRAINENC_CACHE", REPO / "cache"))
CKPT = Path(os.environ.get("BRAINENC_CKPT", REPO / "checkpoints"))
CANDIDATES = Path(os.environ.get("CC2017_CANDIDATES", REPO / "data" / "candidates"))

# derived file names, kept in one place so the scripts stay readable 
SUBJECTS = (1, 2, 3)
N_TRAIN, N_TEST, N_FRAMES = 4320, 1200, 6


def fmri(subject, split):
    return DATA / f"subj{subject:02d}_{split}_fmri.pt"


def stimuli(split):
    return DATA / f"GT_{split}_3fps.pt"


def roi_map(subject):
    return DATA / f"subj{subject:02d}_roi_map.npz"


def early_features(split):
    """VGG19-BN per-frame activations."""
    return CACHE / f"vgg19bn_perframe_{split}.npy"


def dorsal_features(split):
    """VideoMAE last_hidden_state."""
    return CACHE / f"videomae_tokens_{split}.npy"


def ventral_features(split):
    """OpenCLIP ViT-bigG-14 final projection, averaged over frames."""
    return CACHE / f"openclip_bigg_{split}.npy"


def pooler(subject, stream):
    return CKPT / f"{stream}_pool_subj{subject:02d}.pt"


# ---- candidate reconstructions, used by the reranking step ------------------------------
def candidates(subject):
    """Directory of candidate clips: sample_{i}/ with one file per candidate.

    Set CC2017_CANDIDATES to point at the output of whatever generator is being reranked.
    Layout expected by step 04:

        sample_0/<anything>_seed_<S>.gif      or   sample_0/<anything>_seed_<S>.npy
        sample_1/...
    """
    return CANDIDATES / f"subj{subject:02d}"


def candidate_features(subject):
    """Per-stimulus .npz with keys seeds / early / ventral / dorsal, written by step 04."""
    return CACHE / f"candidate_features_subj{subject:02d}"


def selection(subject, tag):
    """Chosen candidate per stimulus, written by step 04."""
    return CACHE / f"selection_subj{subject:02d}_{tag}.npz"


def candidates(subject):
    """Directory of candidate features, one file per test stimulus (step 04).

    Layout expected: sample_{i}.npz for i in 0..N_TEST-1, each holding

        seeds    (K,)          identifier of each candidate, only carried through to the output
        early    (K, 72128)    already pooled, i.e. the same space the encoders were fitted in
        ventral  (K, 1280)
        dorsal   (K, 768)

    Candidates are reconstructions of the test clips produced by a generative model; this
    repository does not generate them. To build the cache, run the extractors of step 01 on the
    candidate clips and apply the step 02 poolers, exactly as 03_fit_encoders.load_features does
    for the stimuli.
    """
    return CANDIDATES / f"subj{subject:02d}"


def selection(subject, method):
    return CACHE / f"selection_subj{subject:02d}_{method}.npz"


def ensure_dirs():
    CACHE.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print(f"repo    {REPO}")
    for name, p in (("DATA", DATA), ("RAW", RAW), ("ATLAS", ATLAS),
                    ("CACHE", CACHE), ("CKPT", CKPT), ("CAND", CANDIDATES)):
        print(f"{name:<8}{p}   {'ok' if p.exists() else 'MISSING'}")
