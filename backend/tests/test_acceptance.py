"""Acceptance tests AT-01..AT-05 (report Table 5.1)."""


def test_AT01_valid_scan_accepted(client, valid_files):
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
    assert "study_id" in r.json()


def test_AT02_corrupted_rejected(client, valid_files, corrupt_file):
    with open(valid_files["adc"], "rb") as adc, open(valid_files["flair"], "rb") as flair, open(
        corrupt_file, "rb"
    ) as dwi:
        r = client.post(
            "/api/upload",
            files={
                "dwi": ("dwi.nii.gz", dwi, "application/gzip"),
                "adc": ("adc.nii.gz", adc, "application/gzip"),
                "flair": ("flair.nii.gz", flair, "application/gzip"),
            },
        )
    assert r.status_code == 422  # error message in body


def test_AT03_full_scan_produces_dashboard(client, done_study):
    r = client.get(f"/api/results/{done_study}")
    assert r.status_code == 200
    body = r.json()
    for k in ("volume_cm3", "confidence", "hemisphere", "regions", "severity"):
        assert k in body


def test_AT04_3d_view_mesh_endpoint(client, done_study):
    r = client.get(f"/api/mesh/{done_study}")
    assert r.status_code == 200
    body = r.json()
    for k in ("brain", "lesion"):
        assert body[k]["vertices"] and body[k]["faces"]


def test_AT05_agent_returns_explanation(client, done_study):
    r = client.post(f"/api/chat/{done_study}", json={"question": "Where is the lesion?"})
    assert r.status_code == 200
    assert len(r.json()["answer"]) > 0
