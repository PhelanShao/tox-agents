# Integrated Full System Bundle

This directory bundles the code that `start_full_system.py` needs to launch the modern frontend and FastAPI backend. The layout keeps the original relationships between components while stripping large data/model assets so it stays lightweight.

## Structure

- `start_full_system.py` – orchestration script that clears ports, checks dependencies, then launches backend and Next.js frontend.
- `frontend/` – Next.js application (without `node_modules`). Install dependencies with `npm install` before running.
  - `frontend/backend/` – FastAPI service entrypoints (`main_fixed.py` is used by the launcher).
- `*.py` at the bundle root – shared backend logic imported by the API (converters, predictors, visualization helpers, RAG service, etc.).
- `models/` – placeholder directory; copy your UniMol checkpoints and reference files here (e.g. `models/ToxPred_modelmini`, `models/MD_model`, `models/refscale.npz`).
- `simple_rag_storage/` – empty workspace for the lightweight RAG service to persist documents at runtime.

## Getting Started

1. Provide runtime assets:
   - Place model directories/files under `models/` (adjust `NEXT_PUBLIC_*` env vars or API params if you use custom paths).
   - Supply any required NPZ datasets the backend should access.
2. Install dependencies:
   - Python environment: install FastAPI stack plus scientific libraries used by the backend modules.
   - Node.js frontend: run `npm install` inside `frontend/`.
3. Launch the system from this directory with `python start_full_system.py` (after activating the proper Python environment).

Environment variables for the frontend (optional):

```bash
export NEXT_PUBLIC_API_URL="http://localhost:8000"
export NEXT_PUBLIC_BINARY_MODEL_PATH="models/ToxPred_modelmini"
export NEXT_PUBLIC_PROPERTY_MODEL_PATH="models/MD_model"
export NEXT_PUBLIC_REFERENCE_PATH="models/refscale.npz"
```

The backend endpoints automatically resolve relative paths against the bundle root, so the defaults work once the model folders live inside `models/`.
