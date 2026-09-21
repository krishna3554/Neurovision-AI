"""Tri-view informative-slice scorer.

[REPORT] identifies informative axial, coronal, sagittal slices; runs before
patch sampling; Fig 4.4 shows six consecutive axial slices selected.
[CHOICE] the scoring formula.
"""
import numpy as np


def _norm(x):
    x = np.asarray(x, dtype=np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def slice_scores(vol, mask=None, axis=2):
    """vol: [3, X, Y, Z] (dwi, adc, flair, z-scored). mask: [X, Y, Z] or None.

    Score = brain coverage + DWI hyperintensity + ADC hypointensity
    (+ lesion area in training).
    """
    dwi, adc = vol[0], vol[1]
    other = tuple(i for i in range(3) if i != axis)
    brain = np.abs(vol).sum(0) > 0
    coverage = brain.sum(axis=other)
    dwi_hi = (dwi > 1.5).sum(axis=other)  # restricted diffusion appears bright
    adc_lo = (adc < -1.0).sum(axis=other)  # and dark on ADC
    s = 0.2 * _norm(coverage) + 0.4 * _norm(dwi_hi) + 0.4 * _norm(adc_lo)
    if mask is not None:
        s = s + 2.0 * _norm(mask.sum(axis=other))  # training-time supervision
    return s


def top_slices(vol, mask=None, k=6):
    """Return {'axial': [...], 'coronal': [...], 'sagittal': [...]} contiguous windows."""
    out = {}
    for name, axis in (("sagittal", 0), ("coronal", 1), ("axial", 2)):
        s = slice_scores(vol, mask, axis)
        kernel = np.ones(k) / k
        win = np.convolve(s, kernel, mode="valid")  # best contiguous window of k slices
        start = int(win.argmax())
        out[name] = list(range(start, start + k))
    return out


def quality_ok(vol, tau=1e-6) -> bool:
    """Fig 3.6 'Quality OK?' decision: fail on all-zero DWI signal etc."""
    return bool(np.max(slice_scores(vol)) >= tau)


def tri_view_sampling_weights(vol, mask, shape):
    """Per-voxel sampling weight map for patch centres (used in training)."""
    w = np.zeros(shape, dtype=np.float32)
    for axis in range(3):
        s = slice_scores(vol, mask, axis)
        shp = [1, 1, 1]
        shp[axis] = -1
        w += s.reshape(shp)
    return w / (w.sum() + 1e-8)
