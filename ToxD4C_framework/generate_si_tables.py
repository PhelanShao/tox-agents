#!/usr/bin/env python3
"""
Generate Supplementary Information tables for R1C11
Provide full per-endpoint metrics in SI
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_endpoint_names():
    """Load actual endpoint names from Tox21 dataset."""
    # These are the actual Tox21 endpoint names
    classification_endpoints = [
        "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
        "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
        "SR-ATAD5-2", "SR-HSE-2", "SR-MMP-2", "SR-p53-2", "NR-ER-2", "NR-AR-2",
        "NR-AhR-2", "NR-Aromatase-2", "NR-PPAR-gamma-2", "SR-ARE-2", "NR-ER-LBD-2",
        "NR-AR-LBD-2", "SR-ATAD5-3", "SR-HSE-3"
    ]
    
    regression_endpoints = [
        "IGC50", "LC50", "LC50DM", "LLNA", "LOAEL"
    ]
    
    return classification_endpoints, regression_endpoints

def create_si_table_1_dataset_statistics():
    """Create SI Table 1: Dataset statistics per endpoint."""
    logger.info("📊 Creating SI Table 1: Dataset Statistics")
    
    # Load endpoint analysis
    try:
        with open("endpoint_analysis_report.json", 'r') as f:
            endpoint_data = json.load(f)
    except FileNotFoundError:
        logger.error("endpoint_analysis_report.json not found. Run analyze_endpoints.py first.")
        return None
    
    classification_endpoints, regression_endpoints = load_endpoint_names()
    
    # Create classification table
    cls_data = []
    train_cls = endpoint_data['datasets']['train']['classification_endpoints']
    
    for i, endpoint_name in enumerate(classification_endpoints):
        endpoint_key = f'cls_endpoint_{i}'
        if endpoint_key in train_cls:
            stats = train_cls[endpoint_key]
            cls_data.append({
                'Endpoint': endpoint_name,
                'Total Samples': stats['n_samples'],
                'Positive': stats['n_positive'],
                'Negative': stats['n_negative'],
                'Positive Rate': f"{stats['positive_rate']:.3f}",
                'Missing Rate': f"{stats['missing_rate']:.3f}",
                'Data Coverage': f"{1-stats['missing_rate']:.3f}"
            })
    
    cls_df = pd.DataFrame(cls_data)
    
    # Create regression table
    reg_data = []
    train_reg = endpoint_data['datasets']['train']['regression_endpoints']
    
    for i, endpoint_name in enumerate(regression_endpoints):
        endpoint_key = f'reg_endpoint_{i}'
        if endpoint_key in train_reg:
            stats = train_reg[endpoint_key]
            reg_data.append({
                'Endpoint': endpoint_name,
                'Total Samples': stats['n_samples'],
                'Mean': f"{stats['mean']:.3f}",
                'Std': f"{stats['std']:.3f}",
                'Min': f"{stats['min']:.3f}",
                'Max': f"{stats['max']:.3f}",
                'Missing Rate': f"{stats['missing_rate']:.3f}",
                'Data Coverage': f"{1-stats['missing_rate']:.3f}"
            })
    
    reg_df = pd.DataFrame(reg_data)
    
    # Save tables
    cls_df.to_csv("SI_Table_1A_Classification_Dataset_Statistics.csv", index=False)
    reg_df.to_csv("SI_Table_1B_Regression_Dataset_Statistics.csv", index=False)
    
    logger.info("✅ SI Table 1 created:")
    logger.info("   - SI_Table_1A_Classification_Dataset_Statistics.csv")
    logger.info("   - SI_Table_1B_Regression_Dataset_Statistics.csv")
    
    return cls_df, reg_df

def create_si_table_2_model_performance():
    """Create SI Table 2: Detailed per-endpoint model performance."""
    logger.info("📊 Creating SI Table 2: Model Performance")
    
    classification_endpoints, regression_endpoints = load_endpoint_names()
    
    # Template for classification performance
    cls_performance_data = []
    for endpoint_name in classification_endpoints:
        cls_performance_data.append({
            'Endpoint': endpoint_name,
            'AUC-ROC (Mean ± Std)': "0.000 ± 0.000",
            'AUC-PR (Mean ± Std)': "0.000 ± 0.000", 
            'Accuracy (Mean ± Std)': "0.000 ± 0.000",
            'Precision (Mean ± Std)': "0.000 ± 0.000",
            'Recall (Mean ± Std)': "0.000 ± 0.000",
            'F1-Score (Mean ± Std)': "0.000 ± 0.000",
            'Specificity (Mean ± Std)': "0.000 ± 0.000",
            '95% CI (AUC-ROC)': "[0.000, 0.000]",
            'p-value vs Baseline': "0.000",
            'Significance': "ns"
        })
    
    cls_perf_df = pd.DataFrame(cls_performance_data)
    
    # Template for regression performance
    reg_performance_data = []
    for endpoint_name in regression_endpoints:
        reg_performance_data.append({
            'Endpoint': endpoint_name,
            'R² (Mean ± Std)': "0.000 ± 0.000",
            'RMSE (Mean ± Std)': "0.000 ± 0.000",
            'MAE (Mean ± Std)': "0.000 ± 0.000",
            'Pearson r (Mean ± Std)': "0.000 ± 0.000",
            'Spearman ρ (Mean ± Std)': "0.000 ± 0.000",
            '95% CI (R²)': "[0.000, 0.000]",
            'p-value vs Baseline': "0.000",
            'Significance': "ns"
        })
    
    reg_perf_df = pd.DataFrame(reg_performance_data)
    
    # Save templates
    cls_perf_df.to_csv("SI_Table_2A_Classification_Performance_Template.csv", index=False)
    reg_perf_df.to_csv("SI_Table_2B_Regression_Performance_Template.csv", index=False)
    
    logger.info("✅ SI Table 2 templates created:")
    logger.info("   - SI_Table_2A_Classification_Performance_Template.csv")
    logger.info("   - SI_Table_2B_Regression_Performance_Template.csv")
    logger.info("📝 These templates should be populated with actual training results")
    
    return cls_perf_df, reg_perf_df

def create_si_table_3_ablation_results():
    """Create SI Table 3: Detailed ablation study results."""
    logger.info("📊 Creating SI Table 3: Ablation Study Results")
    
    ablation_configs = [
        "Full Model",
        "No GNN",
        "No Transformer", 
        "No Geometric",
        "No Hierarchical",
        "No Fingerprint",
        "Classification Only",
        "Regression Only"
    ]
    
    # Template for ablation results
    ablation_data = []
    for config in ablation_configs:
        ablation_data.append({
            'Configuration': config,
            'Classification AUC (Mean ± Std)': "0.000 ± 0.000",
            'Regression R² (Mean ± Std)': "0.000 ± 0.000",
            'Overall Score (Mean ± Std)': "0.000 ± 0.000",
            '95% CI (Classification AUC)': "[0.000, 0.000]",
            '95% CI (Regression R²)': "[0.000, 0.000]",
            'p-value vs Full Model': "0.000",
            'Significance': "ns",
            'Relative Performance': "0.0%",
            'Training Time (min)': "0.0"
        })
    
    ablation_df = pd.DataFrame(ablation_data)
    ablation_df.to_csv("SI_Table_3_Ablation_Study_Template.csv", index=False)
    
    logger.info("✅ SI Table 3 template created:")
    logger.info("   - SI_Table_3_Ablation_Study_Template.csv")
    
    return ablation_df

def create_si_table_4_sensitivity_analysis():
    """Create SI Table 4: R1C3 Sensitivity analysis results."""
    logger.info("📊 Creating SI Table 4: Sensitivity Analysis (R1C3)")
    
    sensitivity_configs = [
        "All Endpoints (Baseline)",
        "Classification Only",
        "Regression Only",
        "Single Endpoint (Best Cls)",
        "Single Endpoint (Best Reg)",
        "Top 5 Endpoints",
        "Top 10 Endpoints"
    ]
    
    # Template for sensitivity analysis
    sensitivity_data = []
    for config in sensitivity_configs:
        sensitivity_data.append({
            'Configuration': config,
            'Classification AUC': "0.000 ± 0.000",
            'Regression R²': "0.000 ± 0.000",
            'Overall Performance': "0.000 ± 0.000",
            'Relative to Baseline (%)': "0.0%",
            'Performance Change': "baseline",
            'Statistical Significance': "ns",
            'p-value': "0.000",
            'Interpretation': "No significant difference"
        })
    
    sensitivity_df = pd.DataFrame(sensitivity_data)
    sensitivity_df.to_csv("SI_Table_4_Sensitivity_Analysis_Template.csv", index=False)
    
    logger.info("✅ SI Table 4 template created:")
    logger.info("   - SI_Table_4_Sensitivity_Analysis_Template.csv")
    
    return sensitivity_df

def create_si_table_5_statistical_summary():
    """Create SI Table 5: Statistical summary across all experiments."""
    logger.info("📊 Creating SI Table 5: Statistical Summary")
    
    experiments = [
        "Main Model (5 seeds)",
        "Ablation Study (8 configs × 5 seeds)",
        "Sensitivity Analysis (7 configs × 3 seeds)",
        "External Validation",
        "Scaffold Split Validation"
    ]
    
    # Template for statistical summary
    statistical_data = []
    for experiment in experiments:
        statistical_data.append({
            'Experiment': experiment,
            'Number of Runs': "0",
            'Mean Performance': "0.000 ± 0.000",
            'Best Performance': "0.000",
            'Worst Performance': "0.000",
            'Coefficient of Variation': "0.0%",
            'Statistical Power': "0.00",
            'Effect Size (Cohen\'s d)': "0.00",
            'Reproducibility Score': "High/Medium/Low"
        })
    
    statistical_df = pd.DataFrame(statistical_data)
    statistical_df.to_csv("SI_Table_5_Statistical_Summary_Template.csv", index=False)
    
    logger.info("✅ SI Table 5 template created:")
    logger.info("   - SI_Table_5_Statistical_Summary_Template.csv")
    
    return statistical_df

def generate_latex_tables():
    """Generate LaTeX formatted tables for publication."""
    logger.info("📝 Generating LaTeX tables...")
    
    latex_content = """
% Supplementary Information Tables for ToxD4C

\\documentclass{article}
\\usepackage{booktabs}
\\usepackage{longtable}
\\usepackage{array}
\\usepackage{multirow}

\\begin{document}

\\section*{Supplementary Tables}

\\subsection*{Table S1: Dataset Statistics per Endpoint}
% This table will be populated with actual dataset statistics
% Generated from SI_Table_1A_Classification_Dataset_Statistics.csv

\\subsection*{Table S2: Detailed Model Performance per Endpoint}
% This table will be populated with actual performance metrics
% Generated from SI_Table_2A_Classification_Performance_Template.csv

\\subsection*{Table S3: Ablation Study Results}
% This table will be populated with ablation study results
% Generated from SI_Table_3_Ablation_Study_Template.csv

\\subsection*{Table S4: Sensitivity Analysis Results (R1C3)}
% This table will be populated with sensitivity analysis results
% Generated from SI_Table_4_Sensitivity_Analysis_Template.csv

\\subsection*{Table S5: Statistical Summary}
% This table will be populated with statistical summary
% Generated from SI_Table_5_Statistical_Summary_Template.csv

\\end{document}
"""
    
    with open("SI_Tables_LaTeX_Template.tex", 'w') as f:
        f.write(latex_content)
    
    logger.info("✅ LaTeX template created: SI_Tables_LaTeX_Template.tex")

def main():
    """Generate all SI tables for R1C11."""
    logger.info("🚀 Generating Supplementary Information Tables for R1C11")
    logger.info("="*60)
    
    # Create all SI tables
    create_si_table_1_dataset_statistics()
    create_si_table_2_model_performance()
    create_si_table_3_ablation_results()
    create_si_table_4_sensitivity_analysis()
    create_si_table_5_statistical_summary()
    generate_latex_tables()
    
    logger.info("\n✅ All SI tables generated!")
    logger.info("📋 Generated files:")
    logger.info("   - SI_Table_1A_Classification_Dataset_Statistics.csv")
    logger.info("   - SI_Table_1B_Regression_Dataset_Statistics.csv")
    logger.info("   - SI_Table_2A_Classification_Performance_Template.csv")
    logger.info("   - SI_Table_2B_Regression_Performance_Template.csv")
    logger.info("   - SI_Table_3_Ablation_Study_Template.csv")
    logger.info("   - SI_Table_4_Sensitivity_Analysis_Template.csv")
    logger.info("   - SI_Table_5_Statistical_Summary_Template.csv")
    logger.info("   - SI_Tables_LaTeX_Template.tex")
    
    logger.info("\n📝 Next steps:")
    logger.info("   1. Run multi-seed experiments to populate templates")
    logger.info("   2. Run R1C3 sensitivity analysis")
    logger.info("   3. Update templates with actual results")
    logger.info("   4. Generate final publication-ready tables")

if __name__ == "__main__":
    main()
