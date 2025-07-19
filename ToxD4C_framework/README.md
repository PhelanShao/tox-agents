# ToxD4C: A Unified Deep Learning Framework for Advanced Molecular Toxicity Prediction

ToxD4C is a state-of-the-art deep learning framework engineered for high-accuracy, interpretable, and robust prediction of molecular toxicity. It moves beyond traditional 2D graph models by creating a holistic molecular representation through the intelligent fusion of topological structure, 3D geometry, and expert-curated chemical features, all enhanced by advanced representation learning techniques.

---

## 🚀 Key Innovations

ToxD4C distinguishes itself through several key architectural and methodological innovations:

1.  **Dynamic GNN-Transformer Hybrid Architecture**: Combines the local feature extraction power of Graph Attention Networks (GAT) with the global context modeling of Transformers. A novel **Dynamic Fusion Module** uses cross-attention to allow these two branches to inform each other before being fused with learned, data-driven weights.
2.  **Supervised Contrastive Learning for Representation Quality**: Employs a `SupConLoss` to structure the embedding space. It pushes molecules with different toxicity profiles apart and pulls those with similar profiles together, forcing the model to learn chemically and biologically meaningful representations.
3.  **Multi-Scale Chemical Feature Integration**: Incorporates information at multiple levels of chemical abstraction:
    *   **Hierarchical GNN Encoder**: Captures graph features at varying neighborhood sizes.
    *   **Enhanced Fingerprint Module**: Integrates a wide array of classical fingerprints (ECFP, MACCS, etc.) and physicochemical descriptors, using an attention mechanism to weigh their importance dynamically.
4.  **Uncertainty-Aware Multi-Task Learning**: A flexible prediction head handles dozens of classification and regression tasks simultaneously. Crucially, it can model its own uncertainty for each task, allowing it to down-weight noisy or difficult tasks during training for a more robust learning process.

---

## 🏗️ Architectural Deep Dive

The power of ToxD4C lies in its modular, multi-branch architecture. Data flows through a series of specialized encoders, is fused intelligently, and then used for prediction.

### Architecture Diagram

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

### Core Modules Explained

#### 1. GNN-Transformer Hybrid Encoder (`gnn_transformer_hybrid.py`)
This is the backbone of the model.
*   **GNN Branch**: A Graph Attention Network (GAT) processes the molecular graph, capturing local chemical environments and bond information.
*   **Transformer Branch**: Treats the atoms as a sequence, using self-attention to model long-range, through-space interactions that are difficult for GNNs to capture.
*   **Dynamic Fusion Module**: The innovation lies here. Instead of a simple sum or concatenation, it uses **cross-attention** to let the GNN features attend to the Transformer features and vice-versa. It then learns a dynamic, per-molecule weight to decide how much to trust the local vs. global view, before producing a fused node representation.

#### 2. Molecular Fingerprint Enhancement (`molecular_fingerprint_enhanced.py`)
This module injects expert chemical knowledge into the model.
*   **Comprehensive Fingerprints**: It calculates a suite of fingerprints (ECFP, MACCS, RDKit, Avalon, Atom-Pair) and ~15 key physicochemical descriptors.
*   **Attention-based Fusion**: Each fingerprint type is first passed through its own small neural network. Then, an attention mechanism calculates importance scores for each fingerprint type for the given molecule. The final representation is a weighted sum, allowing the model to focus on the most relevant chemical information.

#### 3. Hierarchical Encoder (`hierarchical_encoder.py`)
This provides a multi-scale view of the molecule's topology.
*   It consists of several GCN blocks with varying depths (number of layers).
*   A shallow GCN block captures very local information. Deeper blocks aggregate information from larger and larger neighborhoods.
*   The representations from all scales are concatenated and fused, giving the model a rich, multi-resolution understanding of the graph structure.

#### 4. Supervised Contrastive Loss (`contrastive_loss.py`)
This is a key part of the training process, used for representation learning.
*   **The Goal**: To create a semantically meaningful embedding space where the distance between molecules reflects their toxicological similarity.
*   **The Method**: It defines "positive pairs" as two different molecules that have similar toxicity profiles (based on their labels). "Negative pairs" are molecules with dissimilar profiles. The loss function then trains the encoders to pull the representations of positive pairs closer together while pushing negative pairs apart. This results in a much more robust and generalizable model.

#### 5. Multi-Scale Prediction Head (`multi_scale_prediction_head.py`)
This is the final output stage.
*   **Multi-Task Learning**: It has separate, independent neural network "heads" for each toxicity endpoint. This allows for specialized predictors while still benefiting from a shared, rich molecular representation.
*   **Uncertainty Weighting**: For each task, the model can optionally predict a second value: its own uncertainty (log variance). During loss calculation, predictions with high uncertainty are given a lower weight. This prevents the model from being penalized heavily for noisy or inherently unpredictable tasks, leading to more stable training.

---

## 🚀 Getting Started

### 1. Installation

Clone the repository and install the required dependencies using the provided script. This will set up the correct Conda environment and install all necessary packages.

```bash
git clone https://github.com/your-username/ToxD4C.git
cd ToxD4C
bash install_dependencies.sh
```

### 2. Training

The main training script `train_real_data.py` is designed to work with pre-processed LMDB datasets for maximum I/O efficiency.

```bash
# Activate the conda environment
conda activate toxd4c

# Start training
python train_real_data.py \
    --data_dir data/dataset \
    --num_epochs 50 \
    --batch_size 64 \
    --learning_rate 1e-4
```

**Key training parameters:**
*   `--data_dir`: Path to the directory containing `train.lmdb`, `valid.lmdb`, and `test.lmdb`.
*   `--num_epochs`: Number of training epochs.
*   `--batch_size`: Batch size for training.
*   `--learning_rate`: Initial learning rate for the AdamW optimizer.
*   `--config_path`: Path to a custom model configuration file (optional).

The training process logs metrics to the console and saves the best-performing model checkpoint in the `checkpoints_real/` directory.

### 3. Inference

To predict toxicity for new molecules, use the `inference_toxd4c.py` script. It takes a simple text file with one SMILES string per line as input.

```bash
# 1. Create a file with SMILES strings
echo "CCO" > molecules_to_predict.smi
echo "CC(=O)Oc1ccccc1C(=O)O" >> molecules_to_predict.smi # Aspirin

# 2. Run inference using the trained model
python inference_toxd4c.py \
    --model_path checkpoints_real/toxd4c_real_best.pth \
    --smiles_file molecules_to_predict.smi \
    --output_file inference_results.csv
```
The script will generate a `inference_results.csv` file containing the detailed predicted toxicity profiles for the input molecules.

---

## 🤝 Contribution

Contributions are welcome! If you have ideas for improvements, new features, or have found a bug, please feel free to open an issue or submit a pull request.

---

**ToxD4C - Making molecular toxicity prediction more accurate, reliable, and intelligent.** 🧬✨