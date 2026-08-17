# Brain ranking for cc2017: video features → fMRI streams

Code to reproduce the encoding models: three encoders that map features of a video clip to the
fMRI response of a visual ROI, plus the voxel selection they rely on.

Everything here is self-contained. The only paths are the ones in `paths.py`, which point at the
dataset; no script reads anything outside this repository and the dataset directory.

---

## What is built


The three ROIs are groups of Glasser areas, bilateral:

- **early** V1, V2, V3, V4
- **ventral** FFC, PIT, V8, VMV1–3, VVC, PHA1–3, TE2p
- **dorsal** V3A, V3B, V6, V6A, V7, IPS1, LO1–3, FST, MT, MST, V3CD, V4t, PH, IP0

Dataset: 3 subjects, 4320 training stimuli (2 repeats each) and 1200 held-out test stimuli
(10 repeats each), clips of 6 frames at 3 fps.

---

## Running it

```bash
export CC2017_DATA=/path/to/cc2017_video_fmri_dataset
export CC2017_RAW=/path/to/cc2017_raw          # only for step 00
export GLASSER_ATLAS=/path/to/atlas_glasser    # only for step 00
python paths.py                                # check what is found

python 00_build_roi_masks.py --subject 1                    # once per subject
python 01_extract_features.py --stream early   --split train   # once, shared by all subjects
python 01_extract_features.py --stream early   --split test
python 01_extract_features.py --stream ventral --split train
python 01_extract_features.py --stream ventral --split test
python 01_extract_features.py --stream dorsal  --split train
python 01_extract_features.py --stream dorsal  --split test
python 02_train_poolers.py --subject 1 --stream early        # once per subject and stream
python 02_train_poolers.py --subject 1 --stream dorsal
python 03_fit_encoders.py --subject 1

export CC2017_CANDIDATES=/path/to/candidate_reconstructions  # only for step 04
python 04_rerank.py extract --subject 1                      # candidate clips -> features
python 04_rerank.py rank    --subject 1                      # M2, uniform weights
```

---



```
encoder(candidate features)
  -> [M2] subtract the mean prediction across the candidates of this stimulus
  -> weighted correlation with the measured fMRI (noise-ceiling weights)
  -> z-score WITHIN the stimulus, per stream
  -> weighted sum of the three streams
  -> argmax
```

| file | role |
|---|---|
| `paths.py` | every filesystem path, and nothing else |
| `brainenc/rois.py` | Scheme A grouping, voxel → stream masks |
| `brainenc/backbones.py` | the three frozen visual backbones, shared by steps 01 and 04 |
| `brainenc/pooling.py` | the two learned pooling modules |
| `brainenc/ridge.py` | ridge regression |
| `brainenc/metrics.py` | noise ceiling, weighted correlation, encoding scores |
| `brainenc/scoring.py` | M1/M2 candidate scoring, within-stimulus z-score, combination |
| `brainenc/video.py` | clip resampling shared by the extractors |



