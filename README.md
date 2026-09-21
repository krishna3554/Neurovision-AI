# NeuroVision AI

AI-powered stroke detection & neuro-analysis platform: 3D MRI segmentation
(TriViewResMambaUNet), atlas localization, 3D visualization, and a
retrieval-grounded NeuroAssist agent.

> **Research / education only — not a diagnostic device.** AI insights are
> supplementary. Final diagnosis rests with the attending physician.

Companion doc: see the Detailed Implementation Guide (project brief) for the
report-to-deliverable map. Fidelity tags used across the code:
**[REPORT]** = explicitly specified in the report, **[CHOICE]** = sensible
default where the report is silent.

## Layout

```
backend/
  app/
    core/config.py          # settings (spacing, patch, top-k, threshold)
    models/blocks.py, net.py# TriViewResMambaUNet (ResBlock3D + StateSpaceBlock)
    services/
      validator.py          # FR-1/FR-2 upload validation
      preprocess.py         # FR-3 MONAI pipeline
      triview.py            # informative-slice scoring + sampling weights
      inference.py          # FR-5 sliding-window predict + uncertainty
      postprocess.py        # FR-5 mask refine + lesion stats
      atlas.py              # FR-6 Harvard-Oxford mapping + MNI registration
      mesh.py               # FR-7 marching-cubes meshes
      agent.py              # FR-8 RAG NeuroAssist agent
      pipeline.py           # FR-9 state machine + orchestrator
    main.py                 # FastAPI app
  knowledge/*.md            # retrieval KB (DWI/ADC/FLAIR, territories, limits)
  tests/                    # AT-01..05, 6 unit, 5 integration
training/
  train.py                  # training loop w/ deep supervision + resume
  evaluate.py               # metrics + Fig 4.1–4.6 generators
data/splits.json            # subject-level train/val/test split
frontend/                   # Next.js: landing, upload, processing/[id], dashboard/[id]
docs/                       # Figs 3.1–3.6 diagram sources/notes
storage/uploads, storage/outputs
```

## Quickstart

```bash
# backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add LLM_API_KEY
uvicorn app.main:app --reload --port 8000

# frontend
cd frontend
npm install
npm run dev            # http://localhost:3000
```

## Data (ISLES 2022)

```
ISLES-2022/
  sub-strokecaseXXXX/ses-0001/*_{dwi,adc,flair}.nii.gz
  derivatives/sub-strokecaseXXXX/ses-0001/*_msk.nii.gz
```

Create `data/splits.json`: `{"train": [...], "val": [...], "test": [...]}` (subject-level, 80/10/10, fixed seed).

## Training

```bash
python training/train.py --root /path/to/ISLES-2022 --epochs 150
python training/evaluate.py --root /path/to/ISLES-2022 --checkpoint training/checkpoints/best.pt --out docs/figures
```

On Kaggle, write checkpoints to `/kaggle/working/` and re-attach as a dataset to resume.

## API

| Method | Route | Description |
|---|---|---|
| POST | `/api/upload` | Upload dwi+adc+flair (.nii.gz) → `study_id` (422 on invalid, AT-02) |
| POST | `/api/analyze/{sid}` | Launch background pipeline |
| GET | `/api/status/{sid}` | State machine state + error |
| GET | `/api/results/{sid}` | Metrics, regions, slices, stats |
| GET | `/api/mesh/{sid}` | Brain + lesion meshes |
| POST | `/api/chat/{sid}` | `{"question": ...}` → RAG answer |

## Tests

```bash
cd backend
pytest -v   # AT-01..AT-05, 6 unit, 5 integration (synthetic 64³ NIfTis, no dataset needed)
```

## Metric targets (test set)

Dice ≈ 0.81, IoU ≈ 0.70, Precision ≈ 0.89, Recall ≈ 0.78,
Specificity ≈ 0.9998, HD95 ≈ 4.0. Treat as targets, not guarantees.

## Functional requirements

FR-1 upload · FR-2 validation · FR-3 preprocessing · FR-4 tri-view selection ·
FR-5 segmentation+postprocess · FR-6 atlas mapping · FR-7 3D visualization ·
FR-8 neuro-agent · FR-9 checkpoint recovery.

## License / disclaimer

Research prototype. No HIPAA claim. Not for clinical use without validation
and regulatory clearance.
