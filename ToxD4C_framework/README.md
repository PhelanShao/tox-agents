# ToxD4C: Multi-Modal Multi-Task Deep Learning Framework for Molecular Toxicity Prediction

ToxD4C is a comprehensive deep learning framework for predicting molecular toxicity across 31 endpoints (26 classification and 5 regression tasks).

## Architecture Overview

ToxD4C employs a multi-modal molecular representation strategy:

### Molecular Representations
- **Fingerprint Space**: 2048-bit ECFP (radius=2), 167-bit MACCS keys, 2048-bit RDKit fingerprints, and 15 physicochemical descriptors
- **Graph Space**: 119-dimensional atomic features derived from SMILES strings
- **Geometric Space**: Multi-conformer ensemble generated with RDKit ETKDG and MMFF94 energy minimization

### Model Architecture
- **Hybrid Core**: 3-layer Graph Attention Network + 3-layer Transformer in parallel
- **Geometric Encoder**: Distance-based message passing with Gaussian RBF expansion
- **Dynamic Fusion Module**: Bidirectional cross-attention with learnable gating for fusing GNN and Transformer features
- **Multi-Task Prediction Heads**: Task-specific prediction layers for each toxicity endpoint

### Training Features
- Mask-aware loss function for handling missing labels
- Multi-conformer augmentation (configurable conformer sampling per epoch)
- Scaffold-based data splitting for realistic generalization evaluation
- Optional contrastive learning for improved representations

## Directory Structure

```
ToxD4C_framework/
├── configs/
│   └── toxd4c_config.py          # Model configuration and task definitions
├── data/
│   ├── lmdb_dataset.py           # LMDB dataset loader
│   ├── multi_conformer_dataset.py # Multi-conformer dataset
│   └── confidence_evaluator.py   # Similarity-based confidence scoring
├── models/
│   ├── toxd4c.py                 # Main ToxD4C model
│   ├── architectures/
│   │   ├── gnn_transformer_hybrid.py  # GNN-Transformer hybrid architecture
│   │   └── gcn_stack.py          # GCN backbone option
│   ├── encoders/
│   │   ├── geometric_encoder.py  # Distance-based geometric encoder
│   │   ├── geometric_topological_encoder.py  # Dual encoder
│   │   └── hierarchical_encoder.py  # Hierarchical molecular encoder
│   ├── fingerprints/
│   │   └── molecular_fingerprint_enhanced.py  # Fingerprint encoder
│   ├── heads/
│   │   └── multi_scale_prediction_head.py  # Multi-task prediction heads
│   └── losses/
│       └── contrastive_loss.py   # Supervised contrastive loss
├── training/
│   └── splits.py                 # Data splitting utilities
├── train.py                      # Standard training script
├── train_multi_conformer.py      # Multi-conformer training script
├── preprocess_data.py            # Data preprocessing
├── preprocess_multi_conformers.py # Multi-conformer preprocessing
├── inference_toxd4c.py           # Inference script
└── requirements.txt              # Python dependencies
```

## Installation

```bash
cd ToxD4C_framework
pip install -r requirements.txt
```

## Data Preparation

1. Download the ToxD4C dataset from [Figshare](https://doi.org/10.6084/m9.figshare.30156718.v1)
2. Preprocess the data:
```bash
python preprocess_data.py --input_dir data/raw --output_dir data/dataset
```

3. For multi-conformer training:
```bash
python preprocess_multi_conformers.py --input_dir data/dataset --output_dir data/multi_conformer --n_conformers 11
```

## Training

### Standard Training
```bash
python train.py --data_dir data/dataset --output_dir outputs/standard
```

### Multi-Conformer Training
```bash
python train_multi_conformer.py \
    --data_dir data/multi_conformer \
    --conformer_mode random_n \
    --sample_n_conformers 9 \
    --epochs 100 \
    --output_dir outputs/multi_conformer
```

### Ablation Studies
```bash
# GNN only
python train.py --experiment_name gnn_only

# GNN + Transformer
python train.py --experiment_name gnn_transformer

# Full model with 3D geometry
python train.py --experiment_name gnn_transformer_3d
```

## Inference

```bash
python inference_toxd4c.py \
    --checkpoint checkpoints/best_model.pth \
    --input molecules.smi \
    --output predictions.csv
```

## Confidence Scoring

ToxD4C includes a similarity-based confidence scoring system:
- Computes maximum Tanimoto similarity between query compounds and training set using ECFP fingerprints
- Provides reliability grades (A-D) based on similarity thresholds

## Toxicity Endpoints

### Classification Tasks (26)
- Carcinogenicity, Ames Mutagenicity, Respiratory toxicity, Eye irritation/corrosion
- Cardiotoxicity (1, 5, 10, 30 µM thresholds)
- CYP enzymes (1A2, 2C9, 2C19, 2D6, 3A4)
- Nuclear receptors (AR, AR-LBD, AhR, Aromatase, ER, ER-LBD, PPAR-gamma)
- Stress response (ARE, ATAD5, HSE, MMP, p53)

### Regression Tasks (5)
- Acute oral toxicity (LD50)
- Aquatic toxicity (LC50DM, LC50, IGC50)
- Bioconcentration factor (BCF)

## Citation

If you use ToxD4C in your research, please cite:
```bibtex
@article{toxd4c2025,
  title={ToxD4C: Multi-Modal Multi-Task Deep Learning for Comprehensive Molecular Toxicity Prediction},
  author={...},
  journal={...},
  year={2025}
}
```

## License

This project is licensed under the GPL-3.0 License - see the LICENSE file for details.
