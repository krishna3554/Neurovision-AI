"""State machine + orchestrator. [REPORT] Fig 3.4 states, retry on Error."""
import traceback
from enum import Enum

import numpy as np

from backend.app.core.config import settings

STATE_ORDER = [
    "IDLE",
    "UPLOADED",
    "VALIDATED",
    "PREPROCESSING",
    "INFERENCE",
    "POST-PROCESSING",
    "VISUALIZATION",
    "RESPONSE GENERATION",
    "COMPLETED",
]


class State(str, Enum):
    IDLE = "IDLE"
    UPLOADED = "UPLOADED"
    VALIDATED = "VALIDATED"
    PREPROCESSING = "PREPROCESSING"
    INFERENCE = "INFERENCE"
    POST_PROCESSING = "POST-PROCESSING"
    VISUALIZATION = "VISUALIZATION"
    RESPONSE_GENERATION = "RESPONSE GENERATION"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR HANDLING"


JOBS: dict[str, dict] = {}


def set_state(job_id, state, **kw):
    JOBS.setdefault(job_id, {}).update(state=state, **kw)


def _to_numpy_image(out) -> np.ndarray:
    import torch

    img = out["image"]
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    return np.asarray(img)  # [3, X, Y, Z]


def run_pipeline(job_id, paths, seg_model, atlas, agent):
    """Full 6-stage pipeline: Input -> Preprocess -> Slice select -> Segment
    -> Post-process+Mapping -> Output. Synchronous; called as background task."""
    from backend.app.services.atlas import map_to_atlas
    from backend.app.services.mesh import build_meshes
    from backend.app.services.postprocess import (
        dice_confidence,
        lesion_stats,
        refine,
        severity_level,
    )
    from backend.app.services.preprocess import build_preproc
    from backend.app.services.triview import quality_ok, top_slices
    from backend.app.services.validator import validate_study

    try:
        # Decision diamond 1: Valid?
        res = validate_study(paths)
        if not res.is_success():
            set_state(job_id, State.ERROR, error="; ".join(res.get_errors()))
            return
        set_state(job_id, State.VALIDATED)

        set_state(job_id, State.PREPROCESSING)
        pre = build_preproc(spacing=settings.target_spacing)
        out = pre({k: v for k, v in paths.items()})
        vol = _to_numpy_image(out)  # [3, X, Y, Z]

        # Decision diamond 2: Quality OK? (Fig 3.6)
        if not quality_ok(vol, tau=settings.quality_threshold):
            set_state(job_id, State.ERROR, error="Quality check failed: no usable DWI signal")
            return
        slices = top_slices(vol, k=settings.top_k_slices)

        set_state(job_id, State.INFERENCE)
        import torch

        prob, ent = seg_model.predict(torch.tensor(vol))

        set_state(job_id, State.POST_PROCESSING)
        mask = refine(prob, thr=settings.threshold, min_voxels=settings.min_lesion_voxels)
        stats = lesion_stats(mask, settings.target_spacing)
        mapping = map_to_atlas(mask.astype(np.uint8), atlas)
        conf = dice_confidence(prob, mask)
        sev = severity_level(stats["volume_cm3"])

        set_state(job_id, State.VISUALIZATION)
        brain_mask = (np.abs(vol).sum(0) > 0).astype(np.uint8)
        meshes = build_meshes(brain_mask, mask.astype(np.uint8), settings.target_spacing)
        z = int(np.argmax(mask.sum(axis=(0, 1)))) if mask.sum() else int(mask.shape[2] // 2)
        n_pix = int(mask[:, :, z].sum())

        set_state(job_id, State.RESPONSE_GENERATION)
        results = {
            "volume_cm3": stats["volume_cm3"],
            "confidence": conf,
            "hemisphere": mapping["hemisphere"],
            "territory": mapping.get("territory", ""),
            "regions": mapping["regions"],
            "severity": sev,
            "slices": slices,
            "slice_z": z,
            "slice_pixels": n_pix,
            "max_prob": float(prob.max()),
        }
        opening = agent.generate(
            results, "Summarize this stroke finding for the dashboard."
        )
        results["opening"] = opening

        set_state(job_id, State.COMPLETED, results=results, meshes=meshes)
        JOBS[job_id]["results"] = results
        JOBS[job_id]["meshes"] = meshes
    except Exception as e:  # UI offers Retry / Return to idle
        set_state(job_id, State.ERROR, error=f"{e}\n{traceback.format_exc(limit=3)}")
