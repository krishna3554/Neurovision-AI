"""FastAPI app: upload -> analyze -> status/results/mesh/chat."""
import shutil
import uuid

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import settings
from backend.app.services.agent import NeuroAgent
from backend.app.services.atlas import Atlas
from backend.app.services.inference import SegmentationModel
from backend.app.services.pipeline import JOBS, State, run_pipeline
from backend.app.services.validator import validate_study

app = FastAPI(title="NeuroVision AI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.output_dir.mkdir(parents=True, exist_ok=True)

SEG = None
ATLAS = None
AGENT = None


def get_seg():
    global SEG
    if SEG is None:
        SEG = SegmentationModel()
    return SEG


def get_atlas():
    global ATLAS
    if ATLAS is None:
        try:
            ATLAS = Atlas()
        except Exception:
            ATLAS = Atlas(load=False)
    return ATLAS


def get_agent():
    global AGENT
    if AGENT is None:
        AGENT = NeuroAgent()
    return AGENT


@app.post("/api/upload")
async def upload(
    dwi: UploadFile = File(...),
    adc: UploadFile = File(...),
    flair: UploadFile = File(...),
):
    sid = uuid.uuid4().hex[:8]
    paths = {}
    for name, f in (("dwi", dwi), ("adc", adc), ("flair", flair)):
        p = settings.upload_dir / f"{sid}_{name}.nii.gz"
        with open(p, "wb") as out:
            shutil.copyfileobj(f.file, out)
        paths[name] = str(p)
    res = validate_study(paths)
    JOBS[sid] = {
        "state": State.UPLOADED if res.is_success() else State.ERROR,
        "paths": paths,
        "errors": res.get_errors(),
    }
    if not res.is_success():
        raise HTTPException(422, {"errors": res.get_errors()})  # AT-02
    return {"study_id": sid, "state": JOBS[sid]["state"]}


@app.post("/api/analyze/{sid}")
def analyze(sid: str, bg: BackgroundTasks):
    if sid not in JOBS:
        raise HTTPException(404, "unknown study")
    bg.add_task(run_pipeline, sid, JOBS[sid]["paths"], get_seg(), get_atlas(), get_agent())
    return {"started": True}


@app.post("/api/retry/{sid}")
def retry(sid: str, bg: BackgroundTasks):
    """Retry from ERROR state."""
    if sid not in JOBS:
        raise HTTPException(404, "unknown study")
    JOBS[sid].pop("error", None)
    bg.add_task(run_pipeline, sid, JOBS[sid]["paths"], get_seg(), get_atlas(), get_agent())
    return {"started": True}


@app.get("/api/status/{sid}")
def status(sid: str):
    if sid not in JOBS:
        raise HTTPException(404, "unknown study")
    return {"state": JOBS[sid]["state"], "error": JOBS[sid].get("error")}


@app.get("/api/results/{sid}")
def results(sid: str):
    if sid not in JOBS or "results" not in JOBS[sid]:
        raise HTTPException(404, "results not ready")
    return JOBS[sid]["results"]


@app.get("/api/mesh/{sid}")
def mesh(sid: str):
    if sid not in JOBS or "meshes" not in JOBS[sid]:
        raise HTTPException(404, "mesh not ready")
    return JOBS[sid]["meshes"]


@app.post("/api/chat/{sid}")
def chat(sid: str, body: dict):
    if sid not in JOBS or "results" not in JOBS[sid]:
        raise HTTPException(404, "results not ready")
    return {"answer": get_agent().generate(JOBS[sid]["results"], body.get("question", ""))}
