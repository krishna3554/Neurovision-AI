"""Shared fixtures: tiny synthetic NIfTI studies (64^3, bright-cube lesion)."""
import nibabel as nib
import numpy as np
import pytest
from fastapi.testclient import TestClient

SHAPE = (32, 32, 32)


def _vol(seed=0, lesion=True):
    rng = np.random.default_rng(seed)
    v = np.abs(rng.normal(50, 10, SHAPE)).astype(np.float32)
    if lesion:
        v[12:20, 12:20, 12:20] += 120  # bright cube "lesion" on DWI
    return v


def _write(path, arr):
    nib.save(nib.Nifti1Image(arr, np.eye(4)), str(path))


@pytest.fixture()
def valid_files(tmp_path):
    paths = {}
    for i, m in enumerate(("dwi", "adc", "flair")):
        p = tmp_path / f"{m}.nii.gz"
        _write(p, _vol(seed=i, lesion=True))
        paths[m] = str(p)
    return paths


@pytest.fixture()
def corrupt_file(tmp_path):
    p = tmp_path / "dwi.nii.gz"
    p.write_bytes(b"not a nifti")
    return str(p)


@pytest.fixture()
def client():
    from backend.app.main import app

    return TestClient(app)


@pytest.fixture()
def done_study(client, valid_files):
    """Upload + run pipeline synchronously with a stub seg model."""
    import numpy as np

    from backend.app.services import pipeline as P
    from backend.app.services.atlas import Atlas

    files = {
        m: open(p, "rb") for m, p in valid_files.items()
    }
    r = client.post(
        "/api/upload",
        files={m: (f"{m}.nii.gz", fh, "application/gzip") for m, fh in files.items()},
    )
    for fh in files.values():
        fh.close()
    assert r.status_code == 200, r.text
    sid = r.json()["study_id"]

    class StubSeg:
        def predict(self, image):
            prob = np.zeros(image.shape[1:], dtype=np.float32)
            prob[12:20, 12:20, 12:20] = 0.95
            ent = -(prob * np.log(prob + 1e-8) + (1 - prob) * np.log(1 - prob + 1e-8))
            return prob, ent

    from backend.app.services.agent import NeuroAgent

    P.run_pipeline(sid, P.JOBS[sid]["paths"], StubSeg(), Atlas(load=False), NeuroAgent())
    return sid
