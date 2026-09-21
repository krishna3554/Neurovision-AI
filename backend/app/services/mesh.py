"""3D mesh building (FR-7) via marching cubes."""
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import marching_cubes


def mask_to_mesh(vol, level=0.5, spacing=(1, 1, 1), step=2):
    v = np.asarray(vol, dtype=np.float32)
    if v.max() <= level:
        raise ValueError("mask is empty (no surface at level 0.5)")
    v = np.pad(v, 1)  # zero border guarantees a closed surface
    v, f, n, _ = marching_cubes(v, level=level, spacing=spacing, step_size=step)
    return {"vertices": v.round(2).tolist(), "faces": f.tolist()}


def build_meshes(brain_mask, lesion_mask, spacing):
    smooth = ndi.gaussian_filter(np.asarray(brain_mask, dtype=np.float32), 1.0) > 0.5
    out = {"brain": mask_to_mesh(smooth, spacing=spacing, step=3)}
    if np.asarray(lesion_mask).sum() == 0:
        out["lesion"] = {"vertices": [], "faces": []}  # no lesion predicted
    else:
        out["lesion"] = mask_to_mesh(lesion_mask, spacing=spacing, step=1)
    return out
