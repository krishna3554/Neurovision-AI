"""Loader + validator (FR-1, FR-2).

[REPORT] must catch: corruption, missing modalities, shape mismatch,
empty masks, NaN.
"""
from dataclasses import dataclass, field

import nibabel as nib
import numpy as np

REQUIRED = ("dwi", "adc", "flair")


@dataclass
class ValidationResult:
    is_valid: bool = True
    errors: list = field(default_factory=list)

    def is_success(self):
        return self.is_valid

    def get_errors(self):
        return self.errors


def validate_study(paths: dict, mask_path: str | None = None) -> ValidationResult:
    r = ValidationResult()
    missing = [m for m in REQUIRED if m not in paths]
    if missing:
        r.errors.append(f"Missing modalities: {missing}")
    vols = {}
    for m, p in paths.items():
        try:
            img = nib.load(p)
            vols[m] = (img.shape, np.asarray(img.dataobj))
        except Exception as e:
            r.errors.append(f"{m}: corrupted or unreadable ({e})")
    shapes = {m: v[0] for m, v in vols.items()}
    if len(set(shapes.values())) > 1:
        r.errors.append(f"Shape mismatch: {shapes}")
    for m, (_, arr) in vols.items():
        if np.isnan(arr).any() or np.isinf(arr).any():
            r.errors.append(f"{m}: contains NaN/Inf")
        if not np.any(arr):
            r.errors.append(f"{m}: empty volume")
    if mask_path:  # training-time only
        try:
            if not np.any(np.asarray(nib.load(mask_path).dataobj)):
                r.errors.append("Empty lesion mask")
        except Exception as e:
            r.errors.append(f"mask unreadable ({e})")
    r.is_valid = not r.errors
    return r
