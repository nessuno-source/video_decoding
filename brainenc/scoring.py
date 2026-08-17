"""M2 scoring: how a candidate reconstruction is scored against a measured brain response.

Given several candidate reconstructions of the same stimulus, each is embedded, passed through
the encoder of each stream, and compared with the fMRI actually measured for that stimulus. No
ground-truth video is involved: the score is brain-only.

THE ORDER OF OPERATIONS IS NOT COMMUTATIVE and is the substance of the method:

    encoder(features)
      -> [M2] subtract the mean prediction across the candidates of this stimulus
      -> weighted correlation against the measured fMRI (noise-ceiling weights)
      -> z-score WITHIN the stimulus, per stream
      -> weighted sum of the three streams
      -> argmax

WHY THE COMMON MODE IS REMOVED. A strongly conditioned generator produces candidates that are
all similar, so their predicted responses are nearly identical and the raw correlation is
dominated by the component they share. Every candidate then scores about the same and the
ranking carries no information. Subtracting the across-candidate mean first leaves only what
DISTINGUISHES the candidates, which is the only part that can rank them. M1 below is the naive
variant, kept because it is the natural baseline for M2.

WHY THE Z-SCORE IS WITHIN STIMULUS. The three streams produce correlations on different scales,
and the scale also varies from stimulus to stimulus. Standardising within the stimulus makes the
three commensurable before they are summed; doing it across stimuli instead would let easy
stimuli dominate.
"""
import numpy as np
import torch

from .metrics import wcorr

STREAMS = ("early", "ventral", "dorsal")
UNIFORM_WEIGHTS = {"early": 1.0, "ventral": 1.0, "dorsal": 1.0}


def stream_scores(predicted, measured, weights, method="M2"):
    """(K, V) predictions for K candidates vs (V,) measured response -> (K,) correlations.

    `weights` are the per-voxel noise-ceiling weights. With method="M2" the across-candidate
    mean is removed first; with "M1" it is not.
    """
    if method == "M2":
        predicted = predicted - predicted.mean(0, keepdim=True)
    elif method != "M1":
        raise ValueError(f"unknown method: {method}")
    return wcorr(predicted, measured, weights).numpy()


def zscore_within(v):
    """Standardise the scores of one stimulus. With a single stream this is monotone, so the
    argmax is unchanged; it matters only once the streams are summed."""
    v = np.asarray(v, dtype=float)
    return (v - v.mean()) / (v.std() + 1e-9)


def combine(per_stream, weights=None):
    """{stream: (K,) scores} -> (K,) combined score.

    Weights default to uniform. That is a METHOD CHOICE, not an optimisation: tuning them on the
    same stimuli the result is reported on would be circular, and the gain does not justify the
    objection. Streams with weight zero are skipped entirely.
    """
    weights = UNIFORM_WEIGHTS if weights is None else weights
    total = 0.0
    for stream, scores in per_stream.items():
        w = weights.get(stream, 0.0)
        if w:
            total = total + w * zscore_within(scores)
    return total


def select(per_stream, weights=None):
    """Index of the winning candidate."""
    return int(np.argmax(combine(per_stream, weights)))


@torch.no_grad()
def score_candidates(features, encoders, measured, nc_weights, weights=None, method="M2"):
    """One stimulus, end to end.

    features    {stream: (K, D) candidate features}
    encoders    {stream: predict(X) -> (K, V)}   the fitted ridges
    measured    {stream: (V,) measured response, in the encoder's normalisation}
    nc_weights  {stream: (V,) noise-ceiling weights}

    Returns (combined score (K,), {stream: raw correlations (K,)}).
    """
    weights = UNIFORM_WEIGHTS if weights is None else weights
    per_stream = {}
    for stream in STREAMS:
        if not weights.get(stream, 0.0):
            continue
        predicted = encoders[stream](torch.as_tensor(features[stream]).float())
        per_stream[stream] = stream_scores(predicted, measured[stream],
                                           nc_weights[stream], method)
    return combine(per_stream, weights), per_stream
