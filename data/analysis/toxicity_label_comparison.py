#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Toxicity Label Comparison Analysis
Comparing original toxicity labels (y, y_class) with EPA ToxCast/tcpl compliant labels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from scipy.stats import spearmanr, pearsonr
import os

# Set font to Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

def load_and_prepare_data():
    """Load and prepare data for comparison analysis"""
    
    # Load the final dataset
    data_path = 'data/final/processed_final8k213_tcpl_labeled_final.csv'
    data = pd.read_csv(data_path)
    
    print(f"Total records: {len(data):,}")
    
    # Filter for records with both original and tcpl labels
    valid_data = data[
        (data['y'].notna()) & 
        (data['y_class'].notna()) & 
        (data['tcpl_binary_compliant'] != -1) &
        (data['tcpl_ternary_compliant'] != -1)
    ].copy()
    
    print(f"Records with both label types: {len(valid_data):,}")
    
    return data, valid_data

def create_output_directory():
    """Create output directory for plots"""
    output_dir = 'output/GITHUB/analysis/plots'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def plot_continuous_score_comparison(valid_data, output_dir):
    """Compare continuous toxicity scores"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Continuous Toxicity Score Comparison', fontsize=18, fontweight='bold')
    
    # Scatter plot: y vs S_c_tcpl_compliant
    ax1 = axes[0, 0]
    scatter = ax1.scatter(valid_data['y'], valid_data['S_c_tcpl_compliant'], 
                         alpha=0.6, s=30, c='steelblue', edgecolors='white', linewidth=0.5)
    
    # Calculate correlation
    corr_pearson, p_pearson = pearsonr(valid_data['y'], valid_data['S_c_tcpl_compliant'])
    corr_spearman, p_spearman = spearmanr(valid_data['y'], valid_data['S_c_tcpl_compliant'])
    
    ax1.set_xlabel('Original Toxicity Score (y)')
    ax1.set_ylabel('EPA ToxCast/tcpl Score (S_c)')
    ax1.set_title(f'Continuous Score Correlation\nPearson r={corr_pearson:.3f} (p={p_pearson:.3e})\nSpearman ρ={corr_spearman:.3f} (p={p_spearman:.3e})')
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(valid_data['y'], valid_data['S_c_tcpl_compliant'], 1)
    p = np.poly1d(z)
    ax1.plot(valid_data['y'], p(valid_data['y']), "r--", alpha=0.8, linewidth=2)
    
    # Distribution comparison - Original y
    ax2 = axes[0, 1]
    ax2.hist(valid_data['y'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Original Toxicity Score (y)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Original Scores')
    ax2.grid(True, alpha=0.3)
    
    # Distribution comparison - tcpl S_c
    ax3 = axes[1, 0]
    ax3.hist(valid_data['S_c_tcpl_compliant'], bins=50, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('EPA ToxCast/tcpl Score (S_c)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of EPA ToxCast/tcpl Scores')
    ax3.grid(True, alpha=0.3)
    
    # Box plot comparison by original y_class
    ax4 = axes[1, 1]
    box_data = [valid_data[valid_data['y_class'] == cls]['S_c_tcpl_compliant'].values 
                for cls in sorted(valid_data['y_class'].unique())]
    box_labels = [f'Original Class {int(cls)}' for cls in sorted(valid_data['y_class'].unique())]
    
    bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True)
    colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.set_ylabel('EPA ToxCast/tcpl Score (S_c)')
    ax4.set_title('tcpl Scores by Original Classification')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/continuous_score_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/continuous_score_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    return corr_pearson, corr_spearman

def plot_binary_classification_comparison(valid_data, output_dir):
    """Compare binary classification results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Binary Classification Comparison', fontsize=18, fontweight='bold')
    
    # Confusion matrix
    ax1 = axes[0, 0]
    cm = confusion_matrix(valid_data['y_class'], valid_data['tcpl_binary_compliant'])
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['tcpl Low', 'tcpl High'],
                yticklabels=['Original Low', 'Original High'])
    ax1.set_title('Confusion Matrix\n(Original vs EPA ToxCast/tcpl)')
    ax1.set_xlabel('EPA ToxCast/tcpl Binary Classification')
    ax1.set_ylabel('Original Binary Classification')
    
    # Calculate agreement metrics
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    
    # Agreement by class
    ax2 = axes[0, 1]
    agreement_data = []
    labels = []
    
    for orig_class in sorted(valid_data['y_class'].unique()):
        for tcpl_class in sorted(valid_data['tcpl_binary_compliant'].unique()):
            count = len(valid_data[(valid_data['y_class'] == orig_class) & 
                                 (valid_data['tcpl_binary_compliant'] == tcpl_class)])
            agreement_data.append(count)
            labels.append(f'Orig {int(orig_class)} → tcpl {int(tcpl_class)}')
    
    bars = ax2.bar(range(len(agreement_data)), agreement_data, 
                   color=['lightcoral', 'lightblue', 'lightgreen', 'lightyellow'][:len(agreement_data)])
    ax2.set_xlabel('Classification Transition')
    ax2.set_ylabel('Number of Compounds')
    ax2.set_title(f'Classification Agreement\nOverall Accuracy: {accuracy:.3f}')
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(height)}', ha='center', va='bottom')
    
    # Score distributions by classification agreement
    ax3 = axes[1, 0]
    agree_mask = valid_data['y_class'] == valid_data['tcpl_binary_compliant']
    disagree_mask = valid_data['y_class'] != valid_data['tcpl_binary_compliant']
    
    ax3.hist([valid_data[agree_mask]['y'], valid_data[disagree_mask]['y']], 
             bins=30, alpha=0.7, label=['Agreement', 'Disagreement'],
             color=['lightgreen', 'lightcoral'], edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Original Toxicity Score (y)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Original Score Distribution by Agreement')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # tcpl score distributions by classification agreement
    ax4 = axes[1, 1]
    ax4.hist([valid_data[agree_mask]['S_c_tcpl_compliant'], 
              valid_data[disagree_mask]['S_c_tcpl_compliant']], 
             bins=30, alpha=0.7, label=['Agreement', 'Disagreement'],
             color=['lightgreen', 'lightcoral'], edgecolor='black', linewidth=0.5)
    ax4.set_xlabel('EPA ToxCast/tcpl Score (S_c)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('tcpl Score Distribution by Agreement')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/binary_classification_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/binary_classification_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    return accuracy, cm

def plot_ternary_classification_analysis(valid_data, output_dir):
    """Analyze ternary classification from tcpl"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('EPA ToxCast/tcpl Ternary Classification Analysis', fontsize=18, fontweight='bold')
    
    # Ternary distribution
    ax1 = axes[0, 0]
    ternary_counts = valid_data['tcpl_ternary_compliant'].value_counts().sort_index()
    colors = ['lightgreen', 'lightyellow', 'lightcoral']
    bars = ax1.bar(['Low (0)', 'Medium (1)', 'High (2)'], ternary_counts.values, 
                   color=colors, edgecolor='black', linewidth=1)
    ax1.set_ylabel('Number of Compounds')
    ax1.set_title('EPA ToxCast/tcpl Ternary Classification Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Add percentage labels
    total = ternary_counts.sum()
    for bar, count in zip(bars, ternary_counts.values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{count}\n({count/total*100:.1f}%)', ha='center', va='bottom')
    
    # Original binary vs tcpl ternary
    ax2 = axes[0, 1]
    cross_tab = pd.crosstab(valid_data['y_class'], valid_data['tcpl_ternary_compliant'])
    sns.heatmap(cross_tab, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=['tcpl Low', 'tcpl Medium', 'tcpl High'],
                yticklabels=['Original Low', 'Original High'])
    ax2.set_title('Original Binary vs tcpl Ternary')
    ax2.set_xlabel('EPA ToxCast/tcpl Ternary Classification')
    ax2.set_ylabel('Original Binary Classification')
    
    # Score distributions by ternary class
    ax3 = axes[1, 0]
    ternary_data = [valid_data[valid_data['tcpl_ternary_compliant'] == cls]['y'].values 
                    for cls in sorted(valid_data['tcpl_ternary_compliant'].unique())]
    ternary_labels = [f'tcpl Class {int(cls)}' for cls in sorted(valid_data['tcpl_ternary_compliant'].unique())]
    
    bp = ax3.boxplot(ternary_data, labels=ternary_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_ylabel('Original Toxicity Score (y)')
    ax3.set_title('Original Scores by tcpl Ternary Classification')
    ax3.grid(True, alpha=0.3)
    
    # tcpl score distributions by ternary class
    ax4 = axes[1, 1]
    tcpl_ternary_data = [valid_data[valid_data['tcpl_ternary_compliant'] == cls]['S_c_tcpl_compliant'].values 
                         for cls in sorted(valid_data['tcpl_ternary_compliant'].unique())]
    
    bp2 = ax4.boxplot(tcpl_ternary_data, labels=ternary_labels, patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors[:len(bp2['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.set_ylabel('EPA ToxCast/tcpl Score (S_c)')
    ax4.set_title('tcpl Scores by Ternary Classification')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ternary_classification_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/ternary_classification_analysis.pdf', bbox_inches='tight')
    plt.close()
    
    return cross_tab

def generate_summary_report(data, valid_data, corr_pearson, corr_spearman, accuracy, cm, cross_tab, output_dir):
    """Generate a summary report of the comparison analysis"""
    
    report = f"""
# Toxicity Label Comparison Analysis Report

## Dataset Summary
- Total records in dataset: {len(pd.read_csv('data/final/processed_final8k213_tcpl_labeled_final.csv')):,}
- Records with both label types: {len(valid_data):,}
- Coverage: {len(valid_data)/len(data)*100:.1f}%

## Continuous Score Correlation
- Pearson correlation: {corr_pearson:.4f}
- Spearman correlation: {corr_spearman:.4f}

## Binary Classification Agreement
- Overall accuracy: {accuracy:.4f}
- Confusion Matrix:
  - True Negatives: {cm[0,0]:,}
  - False Positives: {cm[0,1]:,}
  - False Negatives: {cm[1,0]:,}
  - True Positives: {cm[1,1]:,}

## Ternary Classification Distribution
{cross_tab.to_string()}

## Key Findings
1. The correlation between original and tcpl scores shows {'moderate' if abs(corr_spearman) > 0.3 else 'weak'} agreement
2. Binary classification accuracy of {accuracy:.1%} indicates {'good' if accuracy > 0.7 else 'moderate'} consistency
3. EPA ToxCast/tcpl provides more nuanced ternary classification
4. tcpl methodology offers regulatory-compliant toxicity assessment

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(f'{output_dir}/comparison_analysis_report.txt', 'w') as f:
        f.write(report)
    
    print("Analysis complete! Generated files:")
    print(f"  - {output_dir}/continuous_score_comparison.png")
    print(f"  - {output_dir}/binary_classification_comparison.png") 
    print(f"  - {output_dir}/ternary_classification_analysis.png")
    print(f"  - {output_dir}/comparison_analysis_report.txt")

def main():
    """Main analysis function"""
    
    print("Starting toxicity label comparison analysis...")
    
    # Load and prepare data
    data, valid_data = load_and_prepare_data()
    
    # Create output directory
    output_dir = create_output_directory()
    
    # Generate comparison plots
    print("Generating continuous score comparison...")
    corr_pearson, corr_spearman = plot_continuous_score_comparison(valid_data, output_dir)
    
    print("Generating binary classification comparison...")
    accuracy, cm = plot_binary_classification_comparison(valid_data, output_dir)
    
    print("Generating ternary classification analysis...")
    cross_tab = plot_ternary_classification_analysis(valid_data, output_dir)
    
    # Generate summary report
    print("Generating summary report...")
    generate_summary_report(data, valid_data, corr_pearson, corr_spearman, accuracy, cm, cross_tab, output_dir)
    
    print("Analysis completed successfully!")

if __name__ == "__main__":
    main()
