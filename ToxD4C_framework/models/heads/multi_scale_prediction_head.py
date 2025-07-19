import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple, Optional

class MultiScalePredictionHead(nn.Module):
    def __init__(self,
                 input_dim: int,
                 task_configs: Dict[str, Dict[str, Any]],
                 dropout: float = 0.1,
                 uncertainty_weighting: bool = False,
                 classification_tasks_list: List[str] = None,
                 regression_tasks_list: List[str] = None):
        super().__init__()
        
        self.input_dim = input_dim
        self.task_configs = task_configs
        self.uncertainty_weighting = uncertainty_weighting
        self.classification_tasks_list = classification_tasks_list or []
        self.regression_tasks_list = regression_tasks_list or []
        
        self.task_heads = nn.ModuleDict()
        
        if self.uncertainty_weighting:
            self.uncertainty_heads = nn.ModuleDict()

        for task_name, config in task_configs.items():
            output_dim = config.get('output_dim', 1)
            
            self.task_heads[task_name] = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(input_dim // 2, output_dim)
            )
            
            if self.uncertainty_weighting:
                self.uncertainty_heads[task_name] = nn.Sequential(
                    nn.Linear(input_dim, input_dim // 2),
                    nn.ReLU(),
                    nn.Linear(input_dim // 2, output_dim)
                )

    def forward(self, x: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        predictions = {}
        uncertainties = {}

        for task_name in self.classification_tasks_list:
            if task_name in self.task_heads:
                predictions[task_name] = self.task_heads[task_name](x)
                if self.uncertainty_weighting:
                    uncertainties[task_name] = self.uncertainty_heads[task_name](x)

        for task_name in self.regression_tasks_list:
            if task_name in self.task_heads:
                predictions[task_name] = self.task_heads[task_name](x)
                if self.uncertainty_weighting:
                    uncertainties[task_name] = self.uncertainty_heads[task_name](x)

        # For compatibility with old format, we can still return concatenated tensors
        # but the dictionary format is more flexible for interpretation.
        cls_preds_list = [predictions[t] for t in self.classification_tasks_list if t in predictions]
        reg_preds_list = [predictions[t] for t in self.regression_tasks_list if t in predictions]

        cls_preds = torch.cat(cls_preds_list, dim=1) if cls_preds_list else torch.empty(x.size(0), 0, device=x.device)
        reg_preds = torch.cat(reg_preds_list, dim=1) if reg_preds_list else torch.empty(x.size(0), 0, device=x.device)

        output_preds = {
            'classification': cls_preds,
            'regression': reg_preds
        }

        return output_preds, uncertainties


if __name__ == '__main__':
    batch_size = 4
    input_dim = 512
    
    task_configs = {
        'Carcinogenicity': {'output_dim': 1, 'task_type': 'classification'},
        'Ames Mutagenicity': {'output_dim': 1, 'task_type': 'classification'},
        'Acute oral toxicity (LD50)': {'output_dim': 1, 'task_type': 'regression'}
    }
    
    features = torch.randn(batch_size, input_dim)
    
    print("--- Testing without uncertainty ---")
    prediction_head = MultiScalePredictionHead(
        input_dim=input_dim,
        task_configs=task_configs,
        uncertainty_weighting=False
    )
    
    preds, uncerts = prediction_head(features)
    
    print("Prediction tasks:")
    for task, pred in preds.items():
        print(f"  {task}: {pred.shape}")
        assert pred.shape == (batch_size,)
        
    print(f"Uncertainties: {uncerts}")
    assert uncerts is None
    print("✓ Test passed!")
    
    print("\n--- Testing with uncertainty ---")
    prediction_head_uncert = MultiScalePredictionHead(
        input_dim=input_dim,
        task_configs=task_configs,
        uncertainty_weighting=True
    )
    
    preds, uncerts = prediction_head_uncert(features)
    
    print("Prediction tasks:")
    for task, pred in preds.items():
        print(f"  {task}: {pred.shape}")
        assert pred.shape == (batch_size,)
        
    print("Uncertainties:")
    for task, uncert in uncerts.items():
        print(f"  {task}: {uncert.shape}")
        assert uncert.shape == (batch_size,)
        assert torch.all(uncert > 0)
        
    print("✓ Test passed!")