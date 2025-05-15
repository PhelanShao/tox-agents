import os
import numpy as np
import pandas as pd
from collections import OrderedDict
from unimol_tools import MolTrain, MolPredict
import logging
from logging import Formatter
import torch
from torch.utils.data import Dataset
import json
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
Formatter.default_msec_format = '%s.%03d'

# Constants
EXCLUDE_COLUMNS = ['symbol', 'coord', 'SampleName']
TEST_SIZE = 0.1
RANDOM_STATE = 42

# Set model save directory
current_date = datetime.now().strftime("%Y%m%d")
MODEL_SAVE_PATH = f"./21stmodel_{current_date}"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

class AugmentedDataset(Dataset):
    def __init__(self, data, label=None, aug_level=1):
        self.data = data
        self.label = label if label is not None else np.zeros((len(data), 1))
        self.aug_level = aug_level

    def __getitem__(self, idx):
        if self.aug_level > 0:
            # 添加距离矩阵的随机噪声
            distance_matrix = self.data[idx]['src_distance']
            noise = np.random.randn(*distance_matrix.shape) * np.random.uniform(0.0005, 0.001) * self.aug_level
            noise = np.triu(noise)
            noise = noise + noise.T - np.diag(noise.diagonal())
            self.data[idx]['src_distance'] += np.clip(noise, 0, None)
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)


def load_and_split_data(train_file='21stregression.npz', verbose=True):
    logger.info("Loading molecular data...")
    data = np.load(train_file, allow_pickle=True)
    target_columns = get_target_columns(data)
    coordinates = [np.array(coord) for coord in data['coord']]
    atoms = [list(symbols) for symbols in data['symbol']]
    targets_df = pd.DataFrame({key: data[key] for key in target_columns})
    indices = np.arange(len(coordinates))
    train_idx, test_idx = train_test_split(indices, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    train_targets_normalized = scaler.fit_transform(targets_df.iloc[train_idx])
    test_targets_normalized = scaler.transform(targets_df.iloc[test_idx])
    
    train_data = {'coordinates': [coordinates[i] for i in train_idx], 'atoms': [atoms[i] for i in train_idx], 'target': train_targets_normalized.tolist()}
    test_data = {'coordinates': [coordinates[i] for i in test_idx], 'atoms': [atoms[i] for i in test_idx], 'target': test_targets_normalized.tolist()}
    
    np.save(os.path.join(MODEL_SAVE_PATH, 'test_targets_original.npy'), targets_df.iloc[test_idx].values)
    return train_data, test_data, scaler, target_columns


def get_target_columns(data):
    return [col for col in data.files if col not in EXCLUDE_COLUMNS and np.issubdtype(data[col].dtype, np.number)]


def apply_tta(test_data, pred_clf, n_augments=5):
    augmented_predictions = []
    for _ in range(n_augments):
        test_data_aug = {'coordinates': [coord + np.random.randn(*coord.shape).clip(-3, 3) * 0.0025 for coord in test_data['coordinates']], 'atoms': test_data['atoms'], 'target': test_data['target']}
        augmented_predictions.append(pred_clf.predict(test_data_aug))
    return np.mean(augmented_predictions, axis=0)


def get_model_configs():
    base_config = {
        'model_name': 'unimolv1',
        'data_type': 'molecule',
        'task': 'multilabel_regression',
        'epochs': 40,
        'batch_size': 4,  # Fixed batch size
        'metrics': 'r2,mae',
        'loss_key': 'MSELoss',
        'target_normalize': 'auto',
        'lr_type': 'cosine',
        'optim_type': 'AdamW',
        'weight_decay': 1e-2,
        'warmup_ratio': 0.1,
        'seed': 42,
        'cuda': True,
        'amp': True,
        'learning_rate': 0.0002  # Fixed learning rate
    }
    
    # Return single configuration
    return [base_config]


def train_ensemble(train_data, test_data, scaler, target_columns):
    configs = get_model_configs()
    all_predictions = []
    
    for i, config in enumerate(configs):
        save_path = os.path.join(MODEL_SAVE_PATH, f'model_{i}')
        os.makedirs(save_path, exist_ok=True)
        model = MolTrain(**config, save_path=save_path)
        model.fit(train_data)
        pred_clf = MolPredict(load_model=save_path)
        all_predictions.append(apply_tta(test_data, pred_clf))
    
    ensemble_pred = np.mean(all_predictions, axis=0)
    ensemble_pred_orig = scaler.inverse_transform(ensemble_pred)
    test_targets_orig = np.load(os.path.join(MODEL_SAVE_PATH, 'test_targets_original.npy'))
    metrics = calculate_metrics(test_targets_orig, ensemble_pred_orig, target_columns)
    
    with open(os.path.join(MODEL_SAVE_PATH, 'ensemble_results.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    pd.DataFrame(ensemble_pred_orig, columns=target_columns).to_csv(os.path.join(MODEL_SAVE_PATH, 'ensemble_predictions.csv'), index=False)
    return metrics


def calculate_metrics(y_true, y_pred, target_columns):
    return {col: {'r2': float(r2_score(y_true[:, i], y_pred[:, i])), 'mae': float(mean_absolute_error(y_true[:, i], y_pred[:, i]))} for i, col in enumerate(target_columns)}


def main():
    if not torch.cuda.is_available():
        logger.warning("No GPU available - training will be slow!")
    else:
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    
    train_data, test_data, scaler, target_columns = load_and_split_data()
    results = train_ensemble(train_data, test_data, scaler, target_columns)
    logger.info("\nEnsemble evaluation completed. Results saved in {MODEL_SAVE_PATH}/ensemble_results.json")


if __name__ == "__main__":
    main()
