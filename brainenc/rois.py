
import re

import numpy as np

SCHEME_A = {
    "early": ["V1", "V2", "V3", "V4"],
    "ventral": ["FFC", "PIT", "V8", "VMV1", "VMV2", "VMV3", "VVC",
                "PHA1", "PHA2", "PHA3", "TE2p"],
    "dorsal": ["V3A", "V3B", "V6", "V6A", "V7", "IPS1", "LO1", "LO2", "LO3",
               "FST", "MT", "MST", "V3CD", "V4t", "PH", "IP0"],
}
STREAMS = ("early", "ventral", "dorsal")


def area_name(glasser_label):
    """'L_V3A_ROI' -> 'V3A'. Hemisphere is dropped: the masks are bilateral."""
    return re.sub(r"_ROI$", "", re.sub(r"^[LR]_", "", str(glasser_label)))


def load_roi_masks(roi_map_path):
    """Return {'early','ventral','dorsal','other': bool mask over voxels}.

    'other' collects every voxel not assigned to any of the three streams; it is never used as
    an input, but keeping it makes the partition explicit and lets you check the coverage.
    """
    d = np.load(roi_map_path, allow_pickle=True)
    areas = np.array([area_name(x) for x in d["glasser"]])
    masks, covered = {}, np.zeros(len(areas), dtype=bool)
    for stream, area_list in SCHEME_A.items():
        m = np.isin(areas, area_list)
        masks[stream] = m
        covered |= m
    masks["other"] = ~covered
    return masks


def summarise(roi_map_path):
    masks = load_roi_masks(roi_map_path)
    total = len(masks["early"])
    return {k: int(v.sum()) for k, v in masks.items()} | {"total": total}
