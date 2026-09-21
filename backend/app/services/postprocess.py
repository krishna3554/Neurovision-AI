"""Post-processing (FR-5): small-component removal, hole filling, stats."""
import numpy as np
from scipy import ndimage as ndi


def refine(prob, thr=0.5, min_voxels=20):
    mask = prob > thr
    lab, n = ndi.label(mask)
    if n:
        sizes = ndi.sum(mask, lab, range(1, n + 1))
        keep = np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s >= min_voxels])
        mask = keep
    return ndi.binary_fill_holes(mask)


def lesion_stats(mask, spacing_mm):
    vox = float(np.prod(spacing_mm))  # mm^3 per voxel
    return {
        "volume_cm3": float(mask.sum() * vox / 1000.0),
        "max_prob_slice": None,
    }


# [CHOICE] severity rule — documented, supplementary only, not validated.
def severity_level(volume_cm3: float) -> dict:
    if volume_cm3 < 5:
        return {"level": 1, "label": "Small", "note": "Routine review"}
    if volume_cm3 < 15:
        return {"level": 2, "label": "Moderate", "note": "Priority review"}
    if volume_cm3 < 40:
        return {"level": 3, "label": "Large", "note": "Requires immediate review"}
    return {"level": 4, "label": "Very large", "note": "Requires immediate review"}


def dice_confidence(prob, mask) -> float:
    """[CHOICE] defensible proxy: mean predicted probability inside the lesion
    (Dice needs ground truth, so never present this as a Dice score)."""
    if mask.sum() == 0:
        return 0.0
    return float(prob[mask > 0].mean())
