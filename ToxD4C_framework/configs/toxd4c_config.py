from typing import Dict, List, Any


CLASSIFICATION_TASKS = [
    'Carcinogenicity', 'Ames Mutagenicity', 'Respiratory toxicity', 
    'Eye irritation', 'Eye corrosion', 'Cardiotoxicity1', 'Cardiotoxicity10', 
    'Cardiotoxicity30', 'Cardiotoxicity5', 'CYP1A2', 'CYP2C19', 'CYP2C9', 
    'CYP2D6', 'CYP3A4', 'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 
    'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 
    'SR-HSE', 'SR-MMP', 'SR-p53'
]

REGRESSION_TASKS = [
    'Acute oral toxicity (LD50)', 'LC50DM', 'BCF', 'LC50', 'IGC50'
]

def get_toxd4c_task_configs() -> Dict[str, Dict[str, Any]]:
    task_configs = {}
    
    for task in CLASSIFICATION_TASKS:
        task_configs[task] = {
            'output_dim': 1,
            'task_type': 'classification',
            'activation': 'sigmoid',
            'loss_function': 'binary_cross_entropy'
        }
    
    for task in REGRESSION_TASKS:
        task_configs[task] = {
            'output_dim': 1,
            'task_type': 'regression',
            'activation': 'linear',
            'loss_function': 'mse'
        }
    
    return task_configs


TASK_GROUPS = {
    'cardiotoxicity': ['Cardiotoxicity1', 'Cardiotoxicity10', 'Cardiotoxicity30', 'Cardiotoxicity5'],
    'cyp_enzymes': ['CYP1A2', 'CYP2C19', 'CYP2C9', 'CYP2D6', 'CYP3A4'],
    'nuclear_receptors': ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma'],
    'stress_response': ['SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'],
    'toxicity_endpoints': ['Carcinogenicity', 'Ames Mutagenicity', 'Respiratory toxicity', 'Eye irritation', 'Eye corrosion'],
    'environmental_fate': ['LC50DM', 'BCF', 'LC50', 'IGC50'],
    'acute_toxicity': ['Acute oral toxicity (LD50)']
}

TASK_WEIGHTS = {
    'Carcinogenicity': 2.0,
    'Ames Mutagenicity': 2.0,
    'Acute oral toxicity (LD50)': 2.0,
    
    'Cardiotoxicity1': 1.5,
    'Cardiotoxicity5': 1.5,
    'Cardiotoxicity10': 1.3,
    'Cardiotoxicity30': 1.3,
    
    'CYP3A4': 1.5,
    'CYP2D6': 1.3,
    'CYP1A2': 1.2,
    'CYP2C9': 1.2,
    'CYP2C19': 1.2,
    
    'LC50': 1.3,
    'IGC50': 1.3,
    'BCF': 1.2,
    'LC50DM': 1.2,
}

def get_task_weights() -> Dict[str, float]:
    weights = TASK_WEIGHTS.copy()
    all_tasks = CLASSIFICATION_TASKS + REGRESSION_TASKS
    
    for task in all_tasks:
        if task not in weights:
            weights[task] = 1.0
    
    return weights


def get_enhanced_toxd4c_config() -> Dict[str, Any]:
    return {
        'atom_features_dim': 119,
        'bond_features_dim': 12,
        'edge_features_dim': 12,
        
        'hidden_dim': 512,
        'num_encoder_layers': 6,
        'num_attention_heads': 8,
        'dropout': 0.1,
        
        'use_geometric_encoder': True,
        'geometric_hidden_dim': 256,
        'max_distance': 10.0,
        'num_rbf': 50,
        
        'use_hierarchical_encoder': True,
        'hierarchy_levels': [2, 4, 8],
        
        'use_hybrid_architecture': True,
        # GNN backbone options: 'graph_attention' (default) or 'pyg_gcn_stack'
        'gnn_backbone': 'graph_attention',
        'use_gnn': True,  # Ablation toggle for the GNN branch
        'use_transformer': True,  # Ablation toggle for the transformer branch
        'gnn_layers': 3,
        # If 'pyg_gcn_stack' is selected, this controls the stack depth (recommended 2-4)
        'gcn_stack_layers': 3,
        'transformer_layers': 3,
        'fusion_method': 'cross_attention',
        'use_dynamic_fusion': True,  # Ablation toggle for the dynamic fusion module
        
        'use_fingerprints': True,
        'fingerprint_dim': 512,
        'fingerprint_configs': {
            'ecfp': {'n_bits': 2048, 'radius': 2},
            'maccs': {'n_bits': 167},
            'rdkit_fp': {'n_bits': 2048},
            'descriptors': {'n_features': 15}
        },
        
        'task_configs': get_toxd4c_task_configs(),
        'task_weights': get_task_weights(),
        'task_groups': TASK_GROUPS,
        
        'use_contrastive_learning': True,
        'contrastive_temperature': 0.1,
        'contrastive_weight': 0.3,
        
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'batch_size': 64,
        'max_epochs': 100,
        'patience': 10,
        'warmup_steps': 1000,
        
        'use_data_augmentation': True,
        'augmentation_ratio': 0.1,
        
        'eval_metrics': ['accuracy', 'f1_score', 'auc', 'precision', 'recall'],
        'regression_metrics': ['rmse', 'mae', 'r2'],
        
        'uncertainty_weighting': True,
        'gradient_clipping': 1.0,
        'label_smoothing': 0.1,
    }


DATASET_SPLITS = {
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1,
    'random_seed': 42,
    'stratify': True
}

DATA_PATHS = {
    'raw_data_dir': 'data/raw/',
    'processed_data_dir': 'data/processed/',
    'model_save_dir': 'checkpoints/',
    'results_dir': 'results/',
    'logs_dir': 'logs/'
}

if __name__ == "__main__":
    config = get_enhanced_toxd4c_config()
    print("ToxD4C Configuration:")
    print(f"Number of classification tasks: {len(CLASSIFICATION_TASKS)}")
    print(f"Number of regression tasks: {len(REGRESSION_TASKS)}")
    print(f"Total number of tasks: {len(CLASSIFICATION_TASKS) + len(REGRESSION_TASKS)}")
    print(f"Model hidden dimension: {config['hidden_dim']}")
    print(f"Number of attention heads: {config['num_attention_heads']}")
    print(f"Number of encoder layers: {config['num_encoder_layers']}")
    
    print("\nTask Groups:")
    for group_name, tasks in TASK_GROUPS.items():
        print(f"{group_name}: {len(tasks)} tasks")
    
    print("\nHigh-Weight Tasks:")
    weights = get_task_weights()
    high_weight_tasks = {k: v for k, v in weights.items() if v > 1.0}
    for task, weight in sorted(high_weight_tasks.items(), key=lambda x: x[1], reverse=True):
        print(f"{task}: {weight}")
