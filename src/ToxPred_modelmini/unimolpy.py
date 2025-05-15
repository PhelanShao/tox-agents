import os
import time
import numpy as np
import pandas as pd
from collections import OrderedDict
from unimol_tools import MolTrain, MolPredict
import logging
from logging import Formatter
import torch
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
Formatter.default_msec_format = '%s.%03d'

def load_coordinate_data(train_file='3998merged_structures_merged.npz'):
    """Load molecular coordinate data"""
    logger.info("Loading coordinate data...")
    train = np.load(train_file, allow_pickle=True)
    
    train_data = {
        'coordinates': [np.array(item) for item in train['coord']],
        'atoms': [list(item) for item in train['symbol']],
        'target': train['y'].tolist()  # Binary classification labels
    }
    
    return train_data

class UnimolModel:
    def __init__(self, unimol_configs):
        self.unimol_configs = unimol_configs
        self.models = []
        
    def train(self, coord_data):
        """Train multiple UnIMol models with TTA"""
        predictions = []
        
        # 确保GPU可用
        if not torch.cuda.is_available():
            logger.warning("No GPU available! Training may be slow.")
        else:
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            torch.cuda.empty_cache()  # 清理GPU内存
        
        for config in self.unimol_configs:
            try:
                # 创建保存路径
                exp_suffix = '_'.join(map(str, config.values()))
                save_path = f'./exp_{exp_suffix}'
                os.makedirs(save_path, exist_ok=True)
                
                # 准备训练数据
                train_data = {
                    'coordinates': coord_data['coordinates'],
                    'atoms': coord_data['atoms'],
                    'target': coord_data['target']
                }

                model = MolTrain(
                    task='classification',
                    data_type='molecule',  # 改为'molecule'而不是'oled'
                    epochs=config['epochs'],
                    learning_rate=config['learning_rate'],
                    batch_size=config['batch_size'],
                    patience=config['patience'],
                    metrics='auc,f1_score,precision,recall',
                    split='random',
                    remove_hs=False,  # 保持False以使用带H的模型
                    save_path=save_path,
                    cuda=True,
                    amp=True,
                    target_normalize=None,
                    drop_out=0.0,
                    optim_type='AdamW',
                    weight_decay=1e-2,
                    max_norm=5.0,
                    warmup_ratio=0.1,
                    max_epochs=config['epochs'],
                    device='cuda:0',
                    logger_level=1
                )

                # 监控GPU使用
                logger.info(f"Pre-training GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
                
                # 训练模型
                logger.info(f"Training UnIMol model with config: {config}")
                model.fit(train_data)
                
                # 训练后监控GPU
                logger.info(f"Post-training GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
                torch.cuda.empty_cache()  # 清理GPU内存
                
                # TTA预测
                pred_clf = MolPredict(load_model=save_path)
                tta_predictions = self._tta_predict(pred_clf, train_data)
                
                predictions.append(tta_predictions)
                self.models.append(model)
                
            except Exception as e:
                logger.error(f"Error training UnIMol model with config {config}: {str(e)}")
                torch.cuda.empty_cache()  # 出错时也清理GPU内存
                continue
        
        if not predictions:
            raise ValueError("No successful UnIMol model training!")
            
        # 平均多个模型的预测结果
        final_predictions = np.mean(predictions, axis=0)
        return final_predictions
    
    def _tta_predict(self, model, data, n_augment=5):
        """Test Time Augmentation prediction"""
        predictions = []
        for i in range(n_augment):
            try:
                aug_data = data.copy()
                aug_data['coordinates'] = [
                    coord + np.random.randn(*coord.shape).clip(-3, 3) * 0.0025 
                    for coord in data['coordinates']
                ]
                pred = model.predict(aug_data)
                predictions.append(pred)
                logger.info(f"Completed TTA iteration {i+1}/{n_augment}")
            except Exception as e:
                logger.error(f"Error in TTA iteration {i+1}: {str(e)}")
                continue
                
        if not predictions:
            raise ValueError("No successful TTA predictions!")
            
        return np.mean(predictions, axis=0)

def main():
   # UnIMol configurations
   unimol_configs = []
   
   # Different learning rate schedules
   lr_types = ['linear', 'cosine', 'polynomial']
   for lr_type in lr_types:
       config = OrderedDict(
           batch_size=8,
           epochs=40,
           learning_rate=0.0003,
           lr_type=lr_type,
           patience=10,
           max_norm=5.0
       )
       unimol_configs.append(config)
   
   # Different batch sizes
   for batch_size in [4, 8]:
       config = OrderedDict(
           batch_size=batch_size,
           epochs=40,
           learning_rate=0.0003,
           lr_type='cosine',
           patience=10,
           max_norm=5.0
       )
       unimol_configs.append(config)
   
   # GPU check and setup
   if torch.cuda.is_available():
       logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
       logger.info(f"Initial GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
       torch.cuda.empty_cache()
   else:
       logger.warning("No GPU available - training will be slow!")
   
   # 创建模型
   model = UnimolModel(unimol_configs)
   
   # 加载数据
   coord_data = load_coordinate_data()
   
   # 训练模型
   logger.info("Training UnIMol models...")
   try:
       predictions = model.train(coord_data)
       logger.info("UnIMol training completed successfully")
       
       # 确保预测结果是一维数组
       predictions = np.array(predictions).ravel()
       logger.info(f"Final predictions shape: {predictions.shape}")
       
       # 保存每个fold的模型路径
       model_dirs = [d for d in os.listdir('.') if d.startswith('exp_')]
       model_dirs = [os.path.join('.', d) for d in model_dirs]
       
       with open('model_dirs.json', 'w') as f:
           json.dump({
               'model_directories': model_dirs,
               'model_count': len(model_dirs)
           }, f, indent=4)
       
       # 创建预测结果DataFrame
       submission = pd.DataFrame({
           'id': range(len(predictions)),
           'prediction': (predictions > 0.5).astype(int)
       })
       
       logger.info(f"Created submission with shape: {submission.shape}")
       logger.info(f"Submission preview:\n{submission.head()}")
       
       submission.to_csv('unimol_submission.csv', index=False)
       
       # 保存详细训练结果
       results = {
           'training': {
               'configs': str(unimol_configs),
               'model_count': len(model_dirs),
               'prediction_shape': predictions.shape[0]
           },
           'predictions': {
               'raw_predictions': predictions.tolist(),
               'binary_predictions': (predictions > 0.5).astype(int).tolist()
           },
           'model_paths': model_dirs
       }
       
       with open('training_results.json', 'w') as f:
           json.dump(results, f, indent=4)
           
       logger.info("Training results saved successfully")
           
   except Exception as e:
       logger.error(f"Error in UnIMol training or prediction: {str(e)}")
       # 添加更多错误信息
       if 'predictions' in locals():
           logger.error(f"Predictions shape: {np.array(predictions).shape}")
           logger.error(f"Predictions type: {type(predictions)}")
       return None
   
   return model, model_dirs

if __name__ == "__main__":
   main()