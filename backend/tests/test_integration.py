"""Integration tests (report Table 5.3): 5 tests."""
import numpy as np


def test_frontend_backend_upload_response(client, valid_files):
    files = {m: open(p, "rb") for m, p in valid_files.items()}
    try:
        r = client.post(
            "/api/upload",
            files={m: (f"{m}.nii.gz", fh, "application/gzip") for m, fh in files.items()},
        )
    finally:
        for fh in files.values():
            fh.close()
    assert r.status_code == 200
    s = client.get(f"/api/status/{r.json()['study_id']}")
    assert s.json()["state"] == "UPLOADED"


def test_backend_model_inference():
    import torch

    from backend.app.services.inference import SegmentationModel

    seg = SegmentationModel.__new__(SegmentationModel)  # skip weight loading
    import torch.nn as nn

    from backend.app.models.net import TriViewResMambaUNet

    seg.dev = "cpu"
    seg.model = TriViewResMambaUNet().eval()

    import monai.inferers as I

    x = torch.zeros(3, 32, 32, 32)
    with torch.no_grad():
        p = I.sliding_window_inference(x[None], (32, 32, 32), 1, seg.model, overlap=0.5)
    assert p.shape == (1, 1, 32, 32, 32)


def test_model_postprocessing_localized_output():
    from backend.app.services.postprocess import lesion_stats, refine

    prob = np.zeros((20, 20, 20), dtype=np.float32)
    prob[5:10, 5:10, 5:10] = 0.9
    mask = refine(prob)
    assert mask.sum() > 0
    assert lesion_stats(mask, (1.0, 1.0, 1.0))["volume_cm3"] > 0


def test_visualization_dashboard_render(client, done_study):
    assert client.get(f"/api/results/{done_study}").status_code == 200
    assert client.get(f"/api/mesh/{done_study}").status_code == 200


def test_agent_retrieval_context(done_study):
    from backend.app.services.agent import NeuroAgent

    agent = NeuroAgent()
    assert len(agent.chunks) > 0
    assert len(agent.retrieve("DWI diffusion stroke", k=2)) > 0
