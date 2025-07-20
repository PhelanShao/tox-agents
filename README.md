# AI-Powered Molecular Toxicity Prediction Platform (Tox-agents)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Next.js](https://img.shields.io/badge/Next.js-14.2+-black.svg)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-latest-green.svg)](https://fastapi.tiangolo.com/)

A comprehensive AI-powered platform for molecular toxicity prediction, combining deep learning models (UniMol(https://github.com/deepmodeling/Uni-Mol), ToxD4C(https://github.com/PhelanShao/ToxD4C)), interactive 3D visualization, and intelligent conversational analysis. This platform provides accurate molecular toxicity assessments through a modern web interface.

## Key Features

### AI Models & Frameworks
- **UniMol Integration**: Molecular property prediction using pre-trained UniMol models
- **ToxD4C Framework**: Custom deep learning framework for toxicity prediction with SMILES and XYZ support
- **Binary Classification**: Toxicity prediction using ToxPred models
- **Property Prediction**: Multi-dimensional molecular property analysis using MD models

### Core Functionalities
- **File Format Support**: XYZ, NPZ, SDF, MOL, SMILES input formats
- **3D Molecular Visualization**: PyMOL-based rendering with multiple display styles
- **Interactive Interface**: Modern Next.js frontend with real-time updates
- **Batch Processing**: Multiple molecule analysis with sequence navigation
- **Export Capabilities**: CSV, JSON, and image export options

### Intelligent Chat System
- **RAG-Enhanced Conversations**: Knowledge-augmented AI chat for molecular analysis
- **Multi-Model Support**: Integration with various LLM providers (OpenRouter, etc.)
- **Context Integration**: Combines prediction results with conversational AI
- **Vision Capabilities**: Support for image-based molecular structure analysis

## System Architecture

### Frontend Stack (Next.js 14)
```
frontend/
├── src/app/                    # Next.js App Router
├── src/components/
│   ├── chat/                  # Chat interface components
│   ├── layout/                # Header, navigation components
│   ├── prediction/            # Result display components
│   ├── toxd4c/               # ToxD4C-specific interface
│   ├── ui/                   # Base UI components (file upload, etc.)
│   └── visualization/        # 3D molecular visualization
├── src/lib/                   # API client and utilities
└── package.json              # Dependencies: React 18, TypeScript, Tailwind CSS
```

**Key Dependencies:**
- Next.js 14.2+ with TypeScript
- Tailwind CSS + Framer Motion for animations
- TanStack Query for API state management
- Molstar for 3D molecular visualization
- React Dropzone for file uploads

### Backend Stack (FastAPI)
```
frontend/backend/
├── main_fixed.py             # Main FastAPI server
└── Core Python Modules:
    ├── interface.py          # Main processing logic
    ├── predictor.py         # Binary prediction engine
    ├── MoleculePredictor.py # Property prediction
    ├── visualizer.py        # PyMOL 3D visualization
    ├── chatbot.py           # Chat interface logic
    ├── simple_rag_service.py # RAG system
    └── toxd4c_wrapper.py    # ToxD4C integration
```

**Key Features:**
- Async FastAPI server with CORS support
- Lazy module loading for faster startup
- Multi-format file processing (XYZ ↔ NPZ conversion)
- Integration with original Gradio-based functionality

## AI Models & Frameworks

### 1. UniMol Integration
The platform integrates UniMol models for molecular property prediction:

**Model Locations:**
- `ToxPred_modelmini/` - Binary toxicity classification models
- `MD_model/` - Multi-property prediction models
- `refscale.npz` - Reference scaling data

**Supported Tasks:**
- Binary toxicity classification (toxic/non-toxic)
- Multi-property prediction (molecular descriptors, ADMET properties)
- NPZ format processing for molecular data
- Batch prediction capabilities

### 2. ToxD4C Framework Architecture

ToxD4C is a state-of-the-art deep learning framework engineered for high-accuracy, interpretable, and robust prediction of molecular toxicity. It moves beyond traditional 2D graph models by creating a holistic molecular representation through the intelligent fusion of topological structure, 3D geometry, and expert-curated chemical features.

#### ToxD4C Architecture Diagram

```
Input Molecule (SMILES)
│
├──> [RDKit Preprocessing] ──> 1. 2D Graph (Atoms, Bonds)
│                           │
│                           ├──> 3D Conformation (Coordinates)
│                           │
│                           └──> Chemical Fingerprints & Descriptors
│
└──────────────────────────────────────────────────────────────────────────┐
                                                                            │
┌─────────────────────────── ENCODING & FUSION ─────────────────────────────┤
│                                                                           │
│   [Branch 1: Hybrid Encoder]──────────────────┐                           │
│   │ GNN (Local) + Transformer (Global)        │                           │
│   └───────────[Dynamic Fusion]──────────────► │                           │
│                                               │                           │
│   [Branch 2: Geometric Encoder (Optional)]──► │                           │
│   │ (Processes 3D Coordinates)                │                           │
│                                               │                           │
│   [Branch 3: Hierarchical Encoder (Optional)]─► │                           │
│   │ (Multi-scale GCN features)                │                           │
│                                               ├─► [Main Feature Fusion] ──► Fused Molecular Representation
│   [Branch 4: Fingerprint Module (Optional)]─► │      (Concatenation +      (High-dimensional Vector)
│   │ (ECFP, MACCS, etc. w/ Attention Fusion)   │       Linear Layer)
│   └───────────────────────────────────────────┘                           │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
                                │
                                │
┌─────────────────────────── LEARNING & PREDICTION ─────────────────────────┐
│                               │                                           │
│   [Supervised Contrastive Loss (Self-Supervision)]                        │
│   │ (Refines the representation space)                                    │
│                               │                                           │
│                               ▼                                           │
│   [Multi-Task Prediction Head]                                            │
│   │                                                                       │
│   ├──> Task 1 (e.g., Carcinogenicity) Prediction  + [Uncertainty]         │
│   ├──> Task 2 (e.g., Ames Mutagenicity) Prediction  + [Uncertainty]         │
│   ├──> ...                                                                │
│   └──> Task N (e.g., LD50) Prediction             + [Uncertainty]         │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

#### Key Innovations

1. **Dynamic GNN-Transformer Hybrid Architecture**: Combines the local feature extraction power of Graph Attention Networks (GAT) with the global context modeling of Transformers. A novel Dynamic Fusion Module uses cross-attention to allow these two branches to inform each other before being fused with learned, data-driven weights.

2. **Supervised Contrastive Learning for Representation Quality**: Employs a SupConLoss to structure the embedding space. It pushes molecules with different toxicity profiles apart and pulls those with similar profiles together, forcing the model to learn chemically and biologically meaningful representations.

3. **Multi-Scale Chemical Feature Integration**: Incorporates information at multiple levels of chemical abstraction:
   - **Hierarchical GNN Encoder**: Captures graph features at varying neighborhood sizes
   - **Enhanced Fingerprint Module**: Integrates a wide array of classical fingerprints (ECFP, MACCS, etc.) and physicochemical descriptors, using an attention mechanism to weigh their importance dynamically

4. **Uncertainty-Aware Multi-Task Learning**: A flexible prediction head handles dozens of classification and regression tasks simultaneously. Crucially, it can model its own uncertainty for each task, allowing it to down-weight noisy or difficult tasks during training for a more robust learning process.

#### Core Modules

**GNN-Transformer Hybrid Encoder**: This is the backbone of the model.
- **GNN Branch**: A Graph Attention Network (GAT) processes the molecular graph, capturing local chemical environments and bond information
- **Transformer Branch**: Treats the atoms as a sequence, using self-attention to model long-range, through-space interactions that are difficult for GNNs to capture
- **Dynamic Fusion Module**: Uses cross-attention to let the GNN features attend to the Transformer features and vice-versa

**Molecular Fingerprint Enhancement**: This module injects expert chemical knowledge into the model.
- **Comprehensive Fingerprints**: Calculates a suite of fingerprints (ECFP, MACCS, RDKit, Avalon, Atom-Pair) and ~15 key physicochemical descriptors
- **Attention-based Fusion**: Each fingerprint type is first passed through its own small neural network, then an attention mechanism calculates importance scores

**Hierarchical Encoder**: Provides a multi-scale view of the molecule's topology.
- Consists of several GCN blocks with varying depths (number of layers)
- A shallow GCN block captures very local information, deeper blocks aggregate information from larger neighborhoods
- The representations from all scales are concatenated and fused

**Supervised Contrastive Loss**: Key part of the training process, used for representation learning.
- Creates a semantically meaningful embedding space where the distance between molecules reflects their toxicological similarity
- Defines "positive pairs" as molecules with similar toxicity profiles and "negative pairs" as molecules with dissimilar profiles

**Multi-Scale Prediction Head**: The final output stage.
- **Multi-Task Learning**: Has separate, independent neural network "heads" for each toxicity endpoint
- **Uncertainty Weighting**: For each task, the model can optionally predict its own uncertainty (log variance)

### 3. RAG (Retrieval-Augmented Generation) System
Implemented in `simple_rag_service.py`:

**Core Components:**
- Document storage in `simple_rag_storage/`
- JSON-based knowledge base management
- Integration with multiple LLM providers via OpenRouter
- Context-aware response generation

**Supported Models (from chatbot.py):**
- Google Gemini 2.0 Flash
- OpenAI GPT-4o and O1 series
- DeepSeek R1 models
- Anthropic Claude 3.7 Sonnet
- Mistral Pixtral Large
- Custom model support

## Quick Start

### Prerequisites
- Python 3.8+ (Conda environment recommended)
- Node.js 18+
- Required for 3D visualization: PyMOL or compatible OpenGL drivers

### 1. Environment Setup
```bash
# Navigate to project directory
cd backupunimolpy

# Activate conda environment (if using conda)
conda activate unimol

# Verify Python dependencies are installed
python -c "import torch, rdkit, numpy, pandas"
```

### 2. Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies (use legacy peer deps if needed)
npm install --legacy-peer-deps

# Start development server
npm run dev
```
Frontend available at: http://localhost:50001 (configured in package.json)

### 3. Backend Setup
```bash
# Navigate to backend directory
cd frontend/backend

# Start FastAPI server with real UniMol integration
python main_fixed.py
```
Backend API available at: http://localhost:8000

### 4. Automated Launch
```bash
# Use the comprehensive startup script
python start_full_system.py
```
This script handles port cleanup, dependency checks, and service coordination.

## Supported File Formats & Features

### Input Formats
- **XYZ files**: 3D molecular coordinates
- **NPZ files**: Processed molecular data for UniMol models
- **SMILES strings**: Chemical structure notation (ToxD4C)
- **SDF/MOL files**: Standard molecular data formats

### Processing Capabilities
- **File Conversion**: XYZ ↔ NPZ conversion via `interface.py`
- **Binary Classification**: Toxic/non-toxic prediction
- **Property Prediction**: Multi-dimensional molecular analysis
- **3D Visualization**: PyMOL-based rendering with multiple styles
- **Batch Processing**: Multiple molecule analysis with navigation

### Export Options
- **CSV**: Prediction results and molecular properties
- **JSON**: Structured data export
- **PNG/Images**: 3D molecular visualizations
- **Frame Data**: Sequential molecular data export

## API Endpoints (FastAPI)

### Core Prediction APIs
```python
# Binary toxicity prediction
POST /api/predict/binary
- file: UploadFile (NPZ format)
- model_path: str (default: ToxPred_modelmini)

# Multi-property prediction
POST /api/predict/property
- file: UploadFile (NPZ format)
- model_path: str (default: MD_model)
- reference_path: str (default: refscale.npz)

# 3D molecular visualization
POST /api/visualize/molecule
- file: UploadFile (XYZ format)
- style: str (ball_and_stick, wireframe, spacefill)
```

### File Processing APIs
```python
# Format conversion
POST /api/convert/xyz-to-npz
POST /api/convert/npz-to-xyz

# ToxD4C predictions
POST /api/toxd4c/predict/smiles
POST /api/toxd4c/predict/file
```

### Utility APIs
```python
GET /health                    # System status
GET /test-import              # Module import test
POST /api/chat/query          # RAG chat interface
```

## Frontend Features (Next.js)

### Modern UI Components
- **Glassmorphism Design**: Semi-transparent backgrounds with blur effects
- **Responsive Layout**: Tailwind CSS with mobile-first approach
- **Dark/Light Theme**: System-aware theme switching
- **Framer Motion**: Smooth animations and transitions

### Key Components (src/components/)
```
├── ui/file-upload.tsx         # Drag & drop file upload
├── visualization/             # 3D molecular visualization
│   └── molecular-visualization.tsx
├── prediction/               # Results display
│   └── prediction-results.tsx
├── chat/                    # AI chat interface
│   └── chat-interface.tsx
├── toxd4c/                  # ToxD4C-specific UI
│   └── toxd4c-interface.tsx
└── layout/                  # Navigation and layout
    └── header.tsx
```

### Interactive Features
- **React Dropzone**: File upload with validation
- **Molstar Integration**: 3D molecular structure viewer
- **Real-time Updates**: TanStack Query for API state management
- **Export Functions**: CSV/JSON download capabilities

## Technical Implementation

### Startup & Configuration
The platform uses several startup scripts:
- `start_full_system.py` - Comprehensive system launcher with port management
- `start_services.py` - Service coordination script
- `toxpre.py` - Original Gradio interface (fallback)

### Model Integration
- **Lazy Loading**: Models loaded on-demand to reduce startup time
- **Path Configuration**: Configurable model paths in API endpoints
- **Error Handling**: Graceful fallback when models unavailable
- **Memory Management**: Automatic cleanup of temporary files

### Development Tools
- **Docker Support**: Multi-stage Dockerfile for containerization
- **Testing Scripts**: Integration tests in `frontend/test_*.py`
- **Debugging**: Comprehensive logging and error reporting
- **Documentation**: Multiple README files for different components

## Project Structure

```
backupunimolpy/                    # Main project directory
├── Startup Scripts
│   ├── start_full_system.py      # Comprehensive system launcher
│   ├── start_services.py         # Service coordination
│   └── toxpre.py                 # Original Gradio interface
│
├── frontend/                      # Next.js Frontend Application
│   ├── src/
│   │   ├── app/                  # Next.js App Router (page.tsx, layout.tsx)
│   │   ├── components/           # React Components
│   │   │   ├── chat/            # AI chat interface
│   │   │   ├── layout/          # Header, navigation
│   │   │   ├── prediction/      # Results display
│   │   │   ├── toxd4c/         # ToxD4C-specific UI
│   │   │   ├── ui/             # File upload, base components
│   │   │   └── visualization/   # 3D molecular viewer
│   │   └── lib/                 # API client, utilities
│   ├── backend/
│   │   └── main_fixed.py        # FastAPI server with real UniMol
│   ├── package.json             # Node.js dependencies
│   └── public/                  # Static assets
│
├── AI Models & Frameworks
│   ├── ToxD4C/                  # Custom toxicity prediction framework
│   │   ├── models/              # Deep learning implementations
│   │   ├── configs/             # Model configurations
│   │   └── requirements.txt     # Python dependencies
│   ├── ToxPred_modelmini/       # UniMol binary classification
│   ├── ToxPred_modellarge/      # Larger model variant
│   ├── MD_model/                # Multi-property prediction
│   └── unimol3528/              # Additional model variant
│
├── RAG & Chat System
│   ├── simple_rag_service.py    # RAG implementation
│   ├── simple_rag_storage/      # Knowledge base
│   └── chatbot.py               # Chat interface logic
│
├── Core Processing Modules
│   ├── interface.py             # Main processing logic
│   ├── predictor.py            # Binary prediction engine
│   ├── MoleculePredictor.py    # Property prediction
│   ├── visualizer.py           # PyMOL 3D visualization
│   ├── toxd4c_wrapper.py       # ToxD4C integration
│   └── file_converter.py       # Format conversion utilities
│
├── Additional Tools
│   ├── molreacone/             # Molecular dynamics interface
│   ├── 123/                    # Example molecular data
│   └── prediction_results/     # Output storage
│
└── Documentation
    ├── README.md               # Main documentation
    ├── STARTUP_GUIDE.md        # Startup instructions
    ├── FIXED_ISSUES.md         # Issue tracking
    └── frontend/README*.md     # Frontend-specific docs
```



## Use Cases & Applications

### Drug Discovery & Development
- **Early-Stage Screening**: Rapid toxicity assessment of drug candidates
- **Lead Optimization**: Identify safer molecular variants
- **ADMET Profiling**: Comprehensive pharmacokinetic property prediction
- **Risk Assessment**: Quantitative toxicity risk evaluation

### Environmental Safety
- **Chemical Risk Assessment**: Evaluate environmental impact of new chemicals
- **Regulatory Compliance**: Support for REACH, OECD guidelines
- **Green Chemistry**: Design safer, more sustainable molecules
- **Pollution Monitoring**: Assess toxicity of environmental contaminants

### Academic Research
- **Computational Toxicology**: Research tool for toxicity mechanisms
- **Chemical Space Exploration**: Systematic molecular property analysis
- **Method Development**: Platform for new prediction algorithm testing
- **Educational Tool**: Teaching molecular toxicology concepts

## Advanced Configuration

### Model Customization
```python
# Custom UniMol configuration
unimol_config = {
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.0003,
    'patience': 10,
    'metrics': 'auc,f1_score,precision,recall'
}

# ToxD4C model parameters
toxd4c_config = {
    'hidden_dim': 512,
    'num_layers': 6,
    'num_heads': 8,
    'dropout': 0.1,
    'use_contrastive_loss': True
}
```

### RAG System Configuration
```python
# RAG service setup
rag_config = {
    'api_key': 'your-openrouter-api-key',
    'base_url': 'https://openrouter.ai/api/v1',
    'model': 'google/gemini-2.0-flash-001',
    'max_tokens': 2048,
    'temperature': 0.7
}
```

## � System Requirements & Setup

### Hardware Requirements
- **RAM**: 8GB+ (16GB recommended for large datasets)
- **Storage**: 10GB+ for models and dependencies
- **CPU**: Multi-core processor recommended
- **GPU**: Optional, CUDA-compatible for faster inference

### Software Dependencies
- **Python**: 3.8+ with conda environment support
- **Node.js**: 18+ for frontend development
- **System Libraries**: OpenGL drivers for 3D visualization

## Troubleshooting

### Common Issues

#### 1. Model Loading Errors
```bash
# Check model files exist
ls -la ToxPred_modelmini/
ls -la MD_model/

# Verify conda environment
conda list | grep torch
conda list | grep rdkit
```

#### 2. Frontend Build Issues
```bash
# Clear npm cache
npm cache clean --force

# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

#### 3. Backend API Errors
```bash
# Check port availability
netstat -tulpn | grep :8000

# Restart backend service
pkill -f "python.*main_fixed.py"
python frontend/backend/main_fixed.py
```

#### 4. PyMOL Visualization Issues
```bash
# Install PyMOL dependencies
conda install -c conda-forge pymol-open-source

# Check OpenGL support
python -c "import pymol; pymol.cmd.get_version()"
```

### Performance Optimization

#### GPU Acceleration
```python
# Enable CUDA for faster inference
import torch
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("Using CPU")
```

#### Memory Management
```python
# Clear GPU memory
torch.cuda.empty_cache()

# Optimize batch size based on available memory
batch_size = min(32, available_memory // estimated_memory_per_sample)
```

## Usage Examples

### Testing the System
```bash
# Health check
curl http://localhost:8000/health

# Test module imports
curl http://localhost:8000/test-import

# Run integration tests
cd frontend
python test_real_prediction.py
```

### API Usage Examples
```bash
# Binary toxicity prediction
curl -X POST "http://localhost:8000/api/predict/binary" \
  -F "file=@molecule.npz" \
  -F "model_path=/mnt/backup2/ai4s/backupunimolpy/ToxPred_modelmini"

# Property prediction
curl -X POST "http://localhost:8000/api/predict/property" \
  -F "file=@molecule.npz" \
  -F "model_path=/mnt/backup2/ai4s/backupunimolpy/MD_model" \
  -F "reference_path=/mnt/backup2/ai4s/backupunimolpy/refscale.npz"

# ToxD4C SMILES prediction
curl -X POST "http://localhost:8000/api/toxd4c/predict/smiles" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CCO"}'
```

### Example SMILES for Testing
From the codebase documentation:
- Aspirin: `CC(=O)OC1=CC=CC=C1C(=O)O`
- Ibuprofen: `CC(C)CC1=CC=C(C=C1)C(C)C(=O)O`
- Acetaminophen: `CC(=O)NC1=CC=C(O)C=C1`
- Caffeine: `CN1C=NC2=C1C(=O)N(C(=O)N2C)C`

## � Project Status & Features

### Implemented Features
- **Frontend**: Modern Next.js interface with TypeScript
- **Backend**: FastAPI server with real UniMol integration
- **File Processing**: XYZ ↔ NPZ conversion and validation
- **3D Visualization**: PyMOL-based molecular rendering
- **AI Chat**: RAG-enhanced conversational interface
- **ToxD4C Integration**: SMILES and XYZ file processing
- **Batch Processing**: Multiple molecule analysis
- **Export Functions**: CSV, JSON, and image export

### Recent Updates (from FIXED_ISSUES.md)
- **Port Management**: Automatic port cleanup and conflict resolution
- **Model Integration**: Lazy loading for faster startup
- **Error Handling**: Comprehensive error recovery and logging
- **Docker Support**: Containerization for deployment
- **Testing Suite**: Integration tests and validation scripts

### Additional Components
- **MolreacONE**: Molecular dynamics simulation interface (in `molreacone/`)
- **QCxMS Integration**: Quantum chemistry calculation support
- **Multiple Model Variants**: ToxPred_modelmini, ToxPred_modellarge, unimol3528
- **Visualization Tools**: Multiple rendering styles and export options

## Acknowledgments

### Key Technologies
- **UniMol**: Universal molecular representation learning framework
- **PyMOL**: Open-source molecular visualization system
- **RDKit**: Open-source cheminformatics toolkit
- **Next.js**: React framework for production-ready applications
- **FastAPI**: Modern, fast web framework for building APIs
- **PyTorch**: Deep learning framework for model implementation

### Development Tools
- **Conda**: Package and environment management
- **TypeScript**: Type-safe JavaScript development
- **Tailwind CSS**: Utility-first CSS framework
- **Framer Motion**: Animation library for React

## Related Resources

### Core Technologies
- [UniMol](https://github.com/dptech-corp/Uni-Mol) - Universal Molecular Representation
- [PyMOL](https://pymol.org/) - Molecular Visualization System
- [RDKit](https://www.rdkit.org/) - Cheminformatics Toolkit
- [Next.js](https://nextjs.org/) - React Framework
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python Web Framework

### Documentation Files
- `README.md` - Main project documentation
- `frontend/README.md` - Frontend-specific documentation
- `frontend/START_GUIDE.md` - Detailed startup instructions
- `frontend/TROUBLESHOOTING.md` - Common issues and solutions
- `FIXED_ISSUES.md` - Recent fixes and improvements
- `ToxD4C/README.md` - ToxD4C framework documentation

## Getting Help

### Quick Start
1. Run `python start_full_system.py` for automated setup
2. Check `http://localhost:8000/health` for backend status
3. Access frontend at `http://localhost:50001`
4. Review logs for any startup issues

### Troubleshooting
- **Port conflicts**: Script automatically handles port cleanup
- **Model loading**: Check paths in `ToxPred_modelmini/` and `MD_model/`
- **Dependencies**: Ensure conda environment is activated
- **Frontend issues**: Try `npm install --legacy-peer-deps`

---
