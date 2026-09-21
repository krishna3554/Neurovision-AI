"""3D mesh building (FR-7) via marching cubes."""
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import marching_cubes


def mask_to_mesh(vol, level=0.5, spacing=(1, 1, 1), step=2):
    v, f, n, _ = marching_cubes(
        vol.astype(np.float32), level=level, spacing=spacing, step_size=step
    )
    return {"vertices": v.round(2).tolist(), "faces": f.tolist()}


def build_meshes(brain_mask, lesion_mask, spacing):
    smooth = ndi.gaussian_filter(brain_mask.astype(np.float32), 1.0) > 0.5
    return {
        "brain": mask_to_mesh(smooth, spacing=spacing, step=3),
        "lesion": mask_to_mesh(lesion_mask, spacing=spacing, step=1),
    }
