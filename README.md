# Tox-Agents: AI-Powered Molecular Toxicity Toolkit

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Next.js](https://img.shields.io/badge/Next.js-14.2+-black.svg)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-latest-green.svg)](https://fastapi.tiangolo.com/)

Tox-Agents is an AI-driven inference platform for exploring molecular toxicity at scale. The ecosystem brings together state-of-the-art deep learning models, transfer-learning frameworks, and reaction-network tooling so that chemists and toxicologists can move from raw structures to defensible insights with minimal friction. The current release (September 22, 2025) unifies the full-stack web experience with offline executables, enabling both rapid experimentation and production-grade studies.

---

## Platform Highlights

- **ToxD4C** – a from-scratch toxicity model trained on a large, diverse dataset for robust endpoint prediction. ([GitHub](https://github.com/PhelanShao/ToxD4C))
- **Uni-Mol Transfer Learning** – leverages powerful pre-trained molecular encoders to boost inference accuracy. ([Uni-Mol Tools](https://github.com/deepmodeling/unimol_tools))
- **Reaction network integration** – tight coupling with Molreac and ReacNet Analyzer for pathway-aware toxicity analysis.
- **Interactive visualization** – frontend widgets expose molecular properties, descriptors, and inference diagnostics in real time.

## Data Availability

- **Tox-D4C dataset**: https://doi.org/10.6084/m9.figshare.30156718.v1
- **Uni-Mol processed data**: `data/data/original/processed_final8k213_original.csv`
- **Reacnet dataset**: https://doi.org/10.6084/m9.figshare.30171562

Computational chemistry descriptors—especially those tied to electronic structure and reactivity—are surfaced directly in the UI to help you interpret model outputs beyond a single toxicity score.

![Platform demo](https://github.com/PhelanShao/tox-agents/blob/main/figure/2.gif)

---

## Access Options

### Online Services

The hosted instance is fully open source and ready to explore:

- https://toxagents.lwy-ai4water-lab.com
- https://www.bohrium.com/apps/toxagents

![Online workflow](https://github.com/PhelanShao/tox-agents/blob/main/figure/3.gif)

### Recommended: Offline Package

For heavy workloads, offline use delivers better stability, avoids LLM API quotas, and reduces server costs. The bundled package includes:

- The all-in-one **ToxD4C** inference executable.
- The **Uni-Mol** inference framework with support for loading multiple pre-trained checkpoints.

![Offline package](https://github.com/PhelanShao/tox-agents/blob/main/figure/4.png)

**Author tip:** We strongly recommend the standalone ToxD4C executable. Although it forgoes massive pre-training, its large in-domain dataset yields highly reliable predictions.

**Getting started offline**

1. Download the `.exe` bundle from the [ToxD4C releases page](https://github.com/PhelanShao/ToxD4C/releases/tag/V2.1).
2. Double-click to launch and allow a minute for initialization before the interface loads.

---

## Micro-Level Toxicity Insight with ToxD4C

ToxD4C excels at tracking subtle shifts in toxicity throughout a reaction pathway. Pair it with **Molreac** for generating reaction networks and **ReacNet Analyzer** for visual inspection to follow how reactants evolve into products.

- **Molreac:ONE**: https://molreac.lwy-ai4water-lab.com/ | https://www.bohrium.com/apps/molreacone

![Molreac demo](https://github.com/PhelanShao/tox-agents/blob/main/figure/molreacone1.png)

Combining 3D conformer searches with reaction-network analysis delivers rich insight that SMILES-only workflows miss. Low-energy structures are directly linked to observed toxicity trends.

---

## Example Workflow: Bisphenol A (BPA) Degradation

Curious whether BPA degradation products remain toxic? Follow this pipeline:

1. **Acquire a 3D structure**: Download the 3D SDF for BPA from PubChem.
2. **Prepare the input**: Copy coordinates into a new `.xyz` file, or generate the 3D structure from SMILES via an empirical force field.
3. **Simulate the network**: Load the `.xyz` file into **Molreac:ONE** to generate reaction pathways (`.reacnet`).
4. **Analyze the network**: Use [ReacNet Analyzer](https://github.com/PhelanShao/reaction_network) to transform `.reacnet` files into interactive HTML visualizations.
5. **Extract pathways**: Identify products that match experiments or select compelling branches and export the minimum-energy structures as `.xyz` files.
6. **Predict toxicity**: Batch the `.xyz` files through **ToxD4C** to evaluate how all 31 toxicity endpoints evolve along the reaction path.

![Reaction analysis](https://github.com/PhelanShao/tox-agents/blob/main/figure/6.png)

With the resulting structural and descriptor data, you can perform manual interpretation, feed the outputs into Tox-Agents for further computation, or escalate to downstream platforms such as DeepSeek for meta-analysis.

---

## Guidance from the Authors

### Common Pitfalls

- Avoid using 2D PubChem structures for inference; always obtain a 3D geometry.
- Reject unrealistic or highly distorted conformations—the predictions will not be meaningful.
- Transition states are insightful but not validated as ground-truth inputs for toxicity; treat them as exploratory evidence only.

### Best Practices

- Optimize geometries with an empirical force field before prediction.
- Sample low-energy conformers, ideally across an entire reaction pathway, to reveal micro-level toxicity shifts.

![Conformer sampling](https://github.com/PhelanShao/tox-agents/blob/main/figure/molreac0.gif)
![Reaction dynamics](https://github.com/PhelanShao/tox-agents/blob/main/figure/molreac1.gif)

---

## Repository Layout

- `src/` – production runtime for the intelligent agent, including the FastAPI backend (`frontend/backend`), Next.js SPA (`frontend`), orchestration scripts, shared predictors, visualizers, and chatbot utilities.
- `data/` – sanitized example datasets used in demos (training corpora are hosted separately).
- `ToxD4C_framework/`, `trainfordl/`, `trainforml/` – research and training code for the ToxD4C deep model, UniMol transfer learning, and traditional ML baselines.
- `requirements.txt` / `requirements_full.txt` – minimal runtime stack vs. the full research environment (including optional UniMol + LightRAG extras).
- `README_original_gradio.md` – legacy documentation for the original Gradio prototype.

> The previous README tracked an older file layout; the sections below reflect the modern `src` bundle.

---

## Integrated Agent Quick Start (Recommended)

### Prerequisites

- Python 3.8+
- Node.js 18+ with npm
- Optional: PyMOL for local 3D visualization

### 1. Install Dependencies

```bash
cd src
pip install -r ../requirements_full.txt  # or requirements.txt for a lean runtime
npm install --prefix frontend
```

### 2. Provide Runtime Assets

- Place UniMol checkpoints under `src/models/` (for example, `models/ToxPred_modelmini`, `models/MD_model`, `models/refscale.npz`).
- Add any NPZ, CSV, or descriptor files required for your workflow.
- Missing assets no longer crash the UI—the frontend highlights the required file and target directory.

### 3. Launch the Orchestrated Stack

```bash
python start_full_system.py
```

The launcher clears ports `3000`, `8000`, and `50001-50003`, validates the environment, installs frontend dependencies on demand, and boots the stack:

- FastAPI backend → `http://localhost:8000`
- Next.js frontend → `http://localhost:3000`

Backend logs stream to the console. Once you see `✅ 后端服务启动成功`, the API is ready. Terminate both services together with `Ctrl+C`.

### 4. Verify the Deployment

- API health check: `curl http://localhost:8000/health`
- Frontend smoke test: open `http://localhost:3000`
- End-to-end test: `python frontend/test_real_prediction.py` (run from `src/frontend`).

---

## Running Services Manually

### Backend Only

```bash
cd src/frontend/backend
uvicorn main_fixed:app --host 0.0.0.0 --port 8000 --reload
```

Key modules:

- `main_fixed.py` lazily loads predictors from `src/` and exposes conversion, prediction, visualization, export, and chat endpoints.
- `simple_rag_service.py` serves a lightweight document store located at `src/simple_rag_storage/`.
- `chatbot.py` implements the Gradio interface and request assembly logic.

Override default model paths by exporting environment variables before launch:

```bash
export BINARY_MODEL_PATH="models/ToxPred_modelmini"
export PROPERTY_MODEL_PATH="models/MD_model"
export REFSCALE_PATH="models/refscale.npz"
```

If a referenced model is missing, the backend returns a clear message detailing which directory to populate.

### Frontend Only

```bash
cd src/frontend
npm install   # first run
npm run dev   # serves http://localhost:3000
```

To connect a remote backend, configure the API endpoint before starting the dev server:

```bash
export NEXT_PUBLIC_API_URL="https://your-backend.example.com"
npm run dev
```

Optional UI hints can be set via environment variables (the backend still governs actual model loading):

```bash
NEXT_PUBLIC_BINARY_MODEL_PATH=models/ToxPred_modelmini
NEXT_PUBLIC_PROPERTY_MODEL_PATH=models/MD_model
NEXT_PUBLIC_REFERENCE_PATH=models/refscale.npz
```

---

## Model Assets and Data Sources

- **UniMol checkpoints** – copy into `src/models/` (for example, reuse `ToxPred_modelmini/` and `MD_model/` from production deployments).
- **ToxD4C weights and datasets** – download from the shared drive (TOXRIC, TDC, Wu et al.) and place under `ToxD4C_framework/data` to retrain.
- **Sample labels** – `data/DATA_labels.csv` contains cleaned labels derived from `21sttox10k`.

Converters in `src/interface.py` support XYZ, NPZ, SDF, MOL, and SMILES inputs for streamlined data preparation.

## Chatbot Prompt Governance (Development)

`src/chatbot.py` currently forwards user turns directly to the configured LLM. To align responses with the ToxD4C analysis policy, store prompt metadata in `src/frontend/backend/llm_report_config.json` (or another shared location) and load it before sending the first request. A representative configuration:

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
- Make the SHAP threshold table visible to the model so the Evidence Matrix can cite it explicitly.
- Persist decision records (rules applied and threshold comparisons) alongside chat transcripts for auditing.
- Surface "AD unknown" whenever the backend does not provide applicability-domain metrics.

## Training and Evaluation Scripts

- ToxD4C training – `ToxD4C_framework/train.py`
- UniMol fine-tuning – `trainfordl/3528_datasets/3528_train.py`
- Classical ML baselines – `unimol_pipeline/run_fingerprint_training.py`

Datasets referenced above require external downloads; consult `ToxD4C_framework/README` for detailed instructions.

## Licensing

Released under the MIT License (see `LICENSE`).

---

Need help or have feedback? Please open an issue or reach out via the project discussions. Happy experimenting!
