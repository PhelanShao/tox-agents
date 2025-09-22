# Tox-Agents: AI-Powered Molecular Toxicity Toolkit

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Next.js](https://img.shields.io/badge/Next.js-14.2+-black.svg)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-latest-green.svg)](https://fastapi.tiangolo.com/)

We warmly welcome you to explore our newly updated AI inference platform (updated on September 22, 2025). Our comprehensive architecture integrates multiple advanced models and tools to provide powerful toxicity predictions.

## Key Features:

### Integrated AI Models:

* **ToxD4C**: An advanced model specifically designed for toxicity prediction, trained from scratch on extensive and diverse datasets, making it exceptionally robust and accurate.
* **Uni-Mol Transfer Learning**: Utilizes transfer learning from pre-trained large molecular models to enhance predictive capabilities.

### Rich Molecular Descriptors:

Our platform allows users to leverage computational chemistry descriptors, especially electronic and reactivity-based descriptors, significantly aiding in the interpretation of molecular toxicity behaviors.

### Online and Open-Source:

This project is open-source, providing both frontend build code and backend inference/calling schemes.

Our online inference services are deployed at:

* https://toxagents.lwy-ai4water-lab.com
* https://www.bohrium.com/apps/toxagents

### Recommended Offline Version:

Due to considerations including server costs and LLM API limitations, we highly recommend using our offline compiled version for toxicity inference. This package includes:

* Integrated ToxD4C inference executable
* Uni-Mol inference framework supporting the loading of various pre-trained models

Authors particularly recommend **ToxD4C**, as it leverages larger and richer datasets, despite not being based on larger-scale pre-trained models.

To use the offline version:

1. Download the executable (.exe) package.
2. Double-click to start and wait approximately one minute for the program to load fully.

## Advantages of ToxD4C:

ToxD4C excels in analyzing micro-level chemical reaction mechanisms, enabling precise tracking of toxicity changes from reactants to products. This feature is especially powerful when combined with **Molreac** (soon to be open-source, currently available for trial).

## Why Move Beyond SMILES?

Real chemical structures are inherently three-dimensional. Using realistic molecular conformations is crucial as conformational space significantly influences toxicity. Combined with low-energy conformational searches and molecular reaction networks, ToxD4C enables far more insightful analyses compared to traditional tools, whose predictions can often be difficult to interpret and less accurate.

## Practical Use Case: BPA Degradation Analysis

Ever wondered about the potential toxicity of Bisphenol A (BPA) degradation products?

Here's our recommended workflow:

1. Obtain the BPA structure by downloading its 3D SDF file from PubChem (ensure it is a 3D, not 2D, structure).
2. Convert molecular coordinates from the SDF file into an XYZ file. Alternatively, generate structures from SMILES strings using empirical force fields.
3. Submit the XYZ file to **Molreacone** to simulate the reaction network. This produces a `.reacnet` file containing detailed reaction pathways.
4. Select experimentally validated products or reaction pathways of interest, then save the lowest-energy pathway structures as XYZ files.
5. Load these XYZ files into **ToxD4C** to begin inference, obtaining predictions for 31 toxicity endpoints along the reaction pathway.

You can further analyze structural variations and associated toxicities using our integrated computational chemistry tool, **Toxagents**, and optionally leverage platforms like **DeepSeek** for deeper analysis.

We invite you to experience our platform and unlock deeper insights into molecular toxicity!

Tox-Agents combines deep-learning toxicity models, interactive chemical visualization, and an LLM-assisted analyst into a single workflow. The modern stack in `src/` ships the production agent: a FastAPI backend, a Next.js frontend, and orchestration scripts that wire them together.

The public deployments stay online at https://toxagents.lwy-ai4water-lab.com/ and https://www.bohrium.com/apps/toxagents (LTS). This repository tracks the latest source so you can reproduce or extend the stack locally.

## What's in this repository
- `src/` – packaged runtime for the intelligent agent; includes the FastAPI backend (`frontend/backend`), Next.js SPA (`frontend`), orchestration (`start_full_system.py`), shared predictors, visualizers, and chatbot utilities.
- `data/` – sanitized examples used by the demos; training datasets live elsewhere (see below).
- `ToxD4C_framework/`, `trainfordl/`, `trainforml/` – research and training code for the ToxD4C deep model, UniMol transfer learning, and classic ML baselines.
- `requirements.txt` / `requirements_full.txt` – minimal runtime stack vs. research stack (with optional UniMol + LightRAG extras).
- `README_original_gradio.md` – legacy documentation for the initial Gradio prototype.

> The previous README tracked an older file layout; everything below reflects the current `src` bundle.

## Integrated agent quick start (recommended)

**Prerequisites**
- Python 3.8+ (any environment works; the launcher now only warns if `unimol_tools` is missing for UniMol transfer inference).
- Node.js 18+ and npm (for Next.js 14).
- Optional: PyMOL if you want local 3D rendering.

**1. Install dependencies**
```bash
cd src
pip install -r ../requirements_full.txt  # or requirements.txt for a lean runtime
npm install --prefix frontend
```

**2. Provide runtime assets**
- Place UniMol checkpoints under `src/models/` (e.g. `models/ToxPred_modelmini`, `models/MD_model`, `models/refscale.npz`).
- Add any NPZ, CSV, or descriptor files needed by your workflows.
- Missing assets no longer crash the UI—the frontend highlights the required file and the expected directory.

**3. Launch the orchestrated stack**
```bash
python start_full_system.py
```
The launcher clears ports (`3000`, `8000`, `50001-50003`), validates the environment, installs frontend deps on demand, and then boots:
- FastAPI backend on `http://localhost:8000`
- Next.js frontend on `http://localhost:3000`

Backend logs stream to the console; when you see `✅ 后端服务启动成功`, the API is ready. The script monitors both processes and forwards stdout/stderr so you can stop everything with `Ctrl+C` once.

**4. Verify**
- API health: `curl http://localhost:8000/health`
- Frontend: open `http://localhost:3000`
- End-to-end smoke test: `python frontend/test_real_prediction.py` (runs from `src/frontend`).

## Running services manually

### Backend only
```bash
cd src/frontend/backend
uvicorn main_fixed:app --host 0.0.0.0 --port 8000 --reload
```
Key files:
- `main_fixed.py` lazily loads the predictors from `src/` and exposes conversion, prediction, visualization, export, and chat endpoints.
- `simple_rag_service.py` delivers a lightweight document store in `src/simple_rag_storage/`.
- `chatbot.py` contains the Gradio interface and request assembly logic.

The backend resolves model paths relative to `src/` by default. Override via environment variables before launch:
```bash
export BINARY_MODEL_PATH="models/ToxPred_modelmini"
export PROPERTY_MODEL_PATH="models/MD_model"
export REFSCALE_PATH="models/refscale.npz"
```
If a referenced model is missing, responses include a clear instruction indicating which directory to populate.

### Frontend only
```bash
cd src/frontend
npm install   # first run
npm run dev   # serve on http://localhost:3000
```
Set the API URL when pointing to a remote backend:
```bash
export NEXT_PUBLIC_API_URL="https://your-backend.example.com"
npm run dev
```
Additional frontend env vars:
```bash
NEXT_PUBLIC_BINARY_MODEL_PATH=models/ToxPred_modelmini
NEXT_PUBLIC_PROPERTY_MODEL_PATH=models/MD_model
NEXT_PUBLIC_REFERENCE_PATH=models/refscale.npz
```
These paths only drive UI hints—the backend still controls the actual inference files.

## Model assets and data sources
- **UniMol transfers**: copy checkpoints into `src/models/`; you can reuse `ToxPred_modelmini/` and `MD_model/` from the production deployment.
- **ToxD4C weights/data**: download from the shared drive (TOXRIC, TDC, Wu et al.) and place under `ToxD4C_framework/data` to retrain.
- **Sample labels**: `data/DATA_labels.csv` hosts cleaned labels derived from `21sttox10k`.

The project supports XYZ, NPZ, SDF, MOL, and SMILES inputs. The converter utilities in `src/interface.py` handle cross-format preparation.

## Chatbot prompt governance (development requirement)
The `src/chatbot.py` interface currently forwards user turns directly to the configured LLM. To align the agent with the updated ToxD4C analysis policy, store prompt metadata in `src/frontend/backend/llm_report_config.json` (or an equivalent shared location) and load it before sending the first request.

Recommended configuration:
```json
{
  "llm_model": "TBD",
  "llm_model_version": "TBD",
  "prompts": {
    "A1_system_prompt": {
      "role": "Chem Risk Analyst aligned to the ToxD4C workflow; produce auditable, uncertainty-aware toxicity interpretations from molecular images, structured descriptor JSON, and optional assay/context files.",
      "grounding": "Use SHAP thresholds (Table 1 digest) as the only quantitative rule base; do not invent data.",
      "evidence_style": "All claims must be backed by an Evidence Matrix (descriptor → value → threshold → direction → reliability).",
      "uncertainty_and_applicability_domain": "Note gaps (units, missing fields). If ECFP4 similarity or embedding Mahalanobis are provided, flag AD in/out; otherwise state 'AD unknown'.",
      "tools": [
        "User KB: ingest CSV/JSON/PDF/image; cite file names.",
        "Web: search authoritative sources; cite links.",
        "Optional plugins: literature.search, cheminfo.lookup, sim.qm, sim.docking, sim.md (generate job cards/protocols; never claim execution without tool confirmation)."
      ],
      "required_outputs": [
        "Quick verdict",
        "Evidence Matrix",
        "Mechanism hypotheses",
        "AD/uncertainty note",
        "Next actions (docking/MD/QM plans with parameters)",
        "Reproducibility facts (seed, version if provided)"
      ],
      "reasoning_policy": "No chain of thought; provide decision records (rules applied and outcomes)."
    },
    "A2_shap_thresholds_digest": [
      {"descriptor": "XLogP", "threshold": 3.05893, "direction": "higher → higher risk", "reliability": 0.979},
      {"descriptor": "HOMO–LUMO gap (a.u.)", "threshold": 0.33105, "direction": "lower → higher risk", "reliability": 0.999},
      {"descriptor": "ALIE Ave (a.u.)", "threshold": 0.50461, "direction": "lower → higher risk", "reliability": 0.999},
      {"descriptor": "Quadrupole moment (a.u.)", "threshold": 21.1766, "direction": "higher → higher risk", "reliability": 0.9998},
      {"descriptor": "Weight (Da)", "threshold": 246.334, "direction": "higher → higher risk", "reliability": 0.992},
      {"descriptor": "LUMO (a.u.)", "threshold": -0.00517, "direction": "more negative → higher risk", "reliability": 0.986},
      {"descriptor": "ALIEmin (eV)", "threshold": 11.2949, "direction": "lower minima → higher risk", "reliability": 0.489},
      {"descriptor": "Negative ESP surface (Bohr²)", "threshold": 359.924, "direction": "higher → higher risk", "reliability": 0.999},
      {"descriptor": "Heavy atom count", "threshold": 14.3053, "direction": "higher → higher risk", "reliability": 0.990},
      {"descriptor": "Complexity", "threshold": 184.588, "direction": "higher → higher risk", "reliability": 1.0},
      {"descriptor": "Rotatable bonds", "threshold": 2.52924, "direction": "too high → entropy penalty; near-threshold optimal", "reliability": 0.9999},
      {"descriptor": "ESPmin (kcal/mol)", "threshold": -36.8484, "direction": "more negative → higher risk", "reliability": 0.958},
      {"descriptor": "HOMO (a.u.)", "threshold": -0.29269, "direction": "less negative (higher) → higher risk", "reliability": 0.999},
      {"descriptor": "LEA Var (eV)", "threshold": 0.06576, "direction": "higher → higher risk", "reliability": 0.9996},
      {"descriptor": "Molecular radius (Å)", "threshold": 6.30992, "direction": "higher → higher risk", "reliability": 0.996},
      {"descriptor": "LEA Ave (a.u.)", "threshold": -0.97949, "direction": "more negative → higher risk", "reliability": 0.814}
    ]
  }
}
```

Implementation checklist:
- Load `A1_system_prompt` as the system message before the first user turn.
- Surface the SHAP threshold table to the model so the Evidence Matrix can cite it explicitly.
- Persist decision records (rules applied, threshold comparisons) alongside chat transcripts for auditing.
- Ensure the chatbot surfaces "AD unknown" whenever the backend does not supply applicability-domain metrics.

## Training and evaluation scripts
- ToxD4C training: `ToxD4C_framework/train.py`
- UniMol fine-tuning: `trainfordl/3528_datasets/3528_train.py`
- Classical ML baselines: `trainforml/ml_train.py`

The datasets referenced above require external downloads; see the linked Google Drive in `ToxD4C_framework/README` for instructions.

## Licensing
Released under the MIT License (see `LICENSE`).
