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
                 regression_tasks_list: List[str] = None,
                 single_endpoint_cls: Optional[int] = None,
                 single_endpoint_reg: Optional[int] = None):
        super().__init__()
        
        self.input_dim = input_dim
        self.task_configs = task_configs
        self.uncertainty_weighting = uncertainty_weighting
        self.single_endpoint_cls = single_endpoint_cls
        self.single_endpoint_reg = single_endpoint_reg
        
        # Filter tasks for single endpoint mode
        if single_endpoint_cls is not None:
            self.classification_tasks_list = [classification_tasks_list[single_endpoint_cls]] if classification_tasks_list and single_endpoint_cls < len(classification_tasks_list) else []
            self.regression_tasks_list = []
        elif single_endpoint_reg is not None:
            self.classification_tasks_list = []
            self.regression_tasks_list = [regression_tasks_list[single_endpoint_reg]] if regression_tasks_list and single_endpoint_reg < len(regression_tasks_list) else []
        else:
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cls_preds_list = []
        reg_preds_list = []

        for task_name in self.classification_tasks_list:
            if task_name in self.task_heads:
                cls_preds_list.append(self.task_heads[task_name](x))

        for task_name in self.regression_tasks_list:
            if task_name in self.task_heads:
                reg_preds_list.append(self.task_heads[task_name](x))

        cls_preds = torch.cat(cls_preds_list, dim=1) if cls_preds_list else torch.empty(x.size(0), 0, device=x.device)
        reg_preds = torch.cat(reg_preds_list, dim=1) if reg_preds_list else torch.empty(x.size(0), 0, device=x.device)

        return cls_preds, reg_preds


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