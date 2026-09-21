"""Preprocessing (FR-3). [REPORT] normalize, resample, align (+ skull strip).

Skull stripping [CHOICE]: ISLES 2022 volumes are already skull-stripped.
If inputs are not, run HD-BET or SynthStrip before this step.
"""
from monai.transforms import (
    Compose,
    ConcatItemsd,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
)

KEYS = ["dwi", "adc", "flair"]


def build_preproc(spacing=(1.0, 1.0, 1.0), with_label=False):
    keys = KEYS + (["label"] if with_label else [])
    modes = ["bilinear"] * 3 + (["nearest"] if with_label else [])
    return Compose(
        [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(keys=keys, pixdim=spacing, mode=modes),
            CropForegroundd(keys=keys, source_key="dwi"),
            NormalizeIntensityd(keys=KEYS, nonzero=True, channel_wise=True),
            ConcatItemsd(keys=KEYS, name="image", dim=0),  # -> [3, X, Y, Z]
            EnsureTyped(keys=["image"] + (["label"] if with_label else [])),
        ]
    )
