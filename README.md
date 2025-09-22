# Tox-Agents: AI-Powered Molecular Toxicity Toolkit

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Next.js](https://img.shields.io/badge/Next.js-14.2+-black.svg)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-latest-green.svg)](https://fastapi.tiangolo.com/)

Welcome to explore our newly updated AI inference platform (updated on September 22, 2025). Comprehensive architecture integrates multiple advanced models and tools to provide powerful toxicity predictions.
## Key Features:

* **ToxD4C**: A formidable toxicity prediction model, built from the ground up on a massive and diverse dataset for exceptional accuracy. ([GitHub Repo](https://github.com/PhelanShao/ToxD4C))
* **Uni-Mol Transfer Learning**: A framework that harnesses the power of large, pre-trained molecular models to supercharge its predictive capabilities. ([Uni-Mol Tools](https://github.com/deepmodeling/unimol_tools))

### Go Deeper Than Just a Number

Yeah, you can get the essential molecular properties you need! We put computational chemistry descriptors—especially those related to electronics and reactivity—at your fingertips. Using these is crucial for truly *understanding* how a molecule behaves and exerts its toxicity.
![演示动画](https://github.com/PhelanShao/tox-agents/blob/main/figure/2.gif)

### Online, Open-Source, and Ready to Go!

This is an online and fully open-source project! With our latest update, we've opened up our frontend build code and the backend inference and calling schemes.

You can access our online inference services here:

* [https://toxagents.lwy-ai4water-lab.com](https://toxagents.lwy-ai4water-lab.com)
* [https://www.bohrium.com/apps/toxagents](https://www.bohrium.com/apps/toxagents)
* 👇
![演示动画](https://github.com/PhelanShao/tox-agents/blob/main/figure/3.gif )
### For the Best Experience, We **Highly** Recommend the Offline Version

Considering server costs, LLM API limitations, and overall stability, the compiled offline version is the best way to go for serious toxicity inference.

The offline package includes:
* An all-in-one **ToxD4C** inference executable.
* The **Uni-Mol** inference framework, which allows you to load various pre-trained models.
![演示动画](https://github.com/PhelanShao/tox-agents/blob/main/figure/4.png )
**An Insider Tip from the Authors:** We especially recommend **ToxD4C**. Why? While it's not based on a larger pre-trained model, it was trained from scratch on a significantly larger and more diverse dataset, making it incredibly robust.

Getting started is easy:
1.  Download the executable (`.exe`) package from our [**Releases Page**](https://github.com/PhelanShao/ToxD4C/releases/tag/V2.1).
2.  Double-click to run it. Please give it a minute or so to load everything up before the interface appears.

### The Real Power of ToxD4C: Micro-Level Insight

When you start using ToxD4C, you'll see its true advantage. It's a fantastic tool for studying the subtle changes in chemical reactions, allowing you to track how toxicity evolves from reactant to product. It's especially powerful when paired with **Molreac** for exploring reaction networks! (Molreac will be open-sourced soon, but you can try it now).

**Molreac:ONE:**
* [https://molreac.lwy-ai4water-lab.com/](https://molreac.lwy-ai4water-lab.com/)
* [https://www.bohrium.com/apps/molreacone](https://www.bohrium.com/apps/molreacone)
*(Yes, we have other versions, but this one is the simplest to get started with!)*
![演示动画](https://github.com/PhelanShao/tox-agents/blob/main/figure/molreacone1.png)
### Why Move Beyond SMILES? Because Real Chemistry is 3D.

So, it's time to stop thinking in SMILES strings! Aren't real molecules three-dimensional, with complex spatial structures? A molecule's conformational space is directly linked to its toxicity.

By combining ToxD4C with low-energy conformer searches and molecular reaction networks, you can produce brilliant, insightful analysis in your next research project. No more wrestling with a pile of hard-to-interpret and less-accurate predictions from other tools.

### Let's Walk Through a Real-World Example: BPA Degradation

Ever wondered if the breakdown products of the endocrine disruptor Bisphenol A (BPA) are potentially toxic?

Here’s our recommended workflow:

1.  **Get the Structure**: Grab the 3D structure of BPA. You can download the SDF file directly from PubChem (and please, make sure it's the **3D** version, not 2D!).
![演示动画](https://github.com/PhelanShao/tox-agents/blob/main/figure/pubchem1.png)
3.  **Prep the Input**: Copy the molecular coordinates from the SDF file into a new `.xyz` file. Alternatively, you can generate a 3D structure from a SMILES string using an empirical force field.
4.  **Simulate the Reaction**: Toss that `.xyz` file into **Molreacone** and tell it to simulate the reaction network. It will work its magic and generate a `.reacnet` file containing all the reaction pathways.
5.  **ReacNet Analyzer**:Use ReacNet Analyzer to parse *.reacnet files to generate network file html [**ReacNet Analyzer**](https://github.com/PhelanShao/reaction_network).
![演示动画](https://github.com/PhelanShao/reaction_network/blob/main/reacnet/demo2.gif)
![演示动画](https://github.com/PhelanShao/reaction_network/blob/main/reacnet/demo3.gif)
7.  **Extract the Path**: Find the products that match experimental data or select a reaction pathway that interests you. Save the structures along the minimum energy path as `.xyz` files.
8.  **Predict Toxicity**: Load these `.xyz` files into **ToxD4C** to begin inference. Just like that, you have a detailed map of how **31 different toxicity endpoints** change along the entire chemical reaction path!
![演示动画](https://github.com/PhelanShao/tox-agents/blob/main/figure/6.png)
Want to know *why* the toxicity changes? You now have the power to find out. You can analyze the structural shifts yourself, or feed the structures into **Toxagents** to get detailed computational chemistry data. For an even deeper dive, you can organize the properties and toxicity trends and let a platform like **DeepSeek** help you analyze the results!

---

## A Friendly Guide from the Authors: Getting the Best Results

#### Common Pitfalls to Avoid:
* **Using the wrong structure**: Never use a 2D structure from PubChem for inference. It just won't work correctly.
* **Using unrealistic geometries**: Feeding the model a highly distorted or physically impossible structure will give you meaningless results (though we'd be curious to see how you managed to create one!).
* **Using transition states**: While interesting for studying reaction mechanisms, transition states are not validated ground-truth structures for toxicity prediction. Use them for insight, not for final conclusions.

#### Best Practices:
* **Always optimize first**: Use an empirical force field to optimize your molecular geometry before running predictions.
* **Explore conformational space**: Use molecular dynamics to sample low-energy conformers of your target molecule. Better yet, sample along a reaction path to see how toxicity responds to micro-level structural changes. That's where the real insights are!

![演示动画](https://github.com/PhelanShao/tox-agents/blob/main/figure/molreac0.gif)
![演示动画](https://github.com/PhelanShao/tox-agents/blob/main/figure/molreac1.gif)

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
