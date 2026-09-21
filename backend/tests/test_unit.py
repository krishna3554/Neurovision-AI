"""Unit tests (report Table 5.2): 6 tests."""
import numpy as np


def test_loader_nifti_import(valid_files):
    import nibabel as nib

    for p in valid_files.values():
        assert np.any(np.asarray(nib.load(p).dataobj))


def test_normalizer_stable_range(valid_files):
    from backend.app.services.preprocess import build_preproc

    out = build_preproc()({k: v for k, v in valid_files.items()})
    img = np.asarray(out["image"])  # [3, X, Y, Z]
    brain = img[:, np.abs(img).sum(0) > 0]
    assert abs(brain.mean()) < 0.5 and 0.5 < brain.std() < 2.0


def test_triview_informative_first():
    from backend.app.services.triview import top_slices

    vol = np.zeros((3, 16, 16, 32), dtype=np.float32) + 0.1
    vol[0, 6:10, 6:10, 20:24] = 5.0  # synthetic lesion block
    vol[1, 6:10, 6:10, 20:24] = -5.0
    mask = np.zeros((16, 16, 32), dtype=np.uint8)
    mask[6:10, 6:10, 20:24] = 1
    top = top_slices(vol, mask, k=6)["axial"]
    assert any(20 <= z < 24 for z in top)


def test_segmentation_outputs_mask():
    import torch

    from backend.app.models.net import TriViewResMambaUNet

    m = TriViewResMambaUNet().eval()
    with torch.no_grad():
        out = m(torch.zeros(1, 3, 32, 32, 32))
    p = torch.sigmoid(out)
    assert p.shape == (1, 1, 32, 32, 32)
    assert 0.0 <= p.min() and p.max() <= 1.0


def test_atlas_mapper_label():
    from backend.app.services.atlas import Atlas, map_to_atlas

    atlas = Atlas(load=False)
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[1:3, 1:3, 1:3] = 1  # left side
    r = map_to_atlas(mask, atlas)
    assert r["hemisphere"] == "Left"


def test_renderer_creates_mesh():
    from backend.app.services.mesh import mask_to_mesh

    ball = np.zeros((16, 16, 16), dtype=np.uint8)
    ball[5:11, 5:11, 5:11] = 1
    m = mask_to_mesh(ball)
    assert m["vertices"] and m["faces"]
