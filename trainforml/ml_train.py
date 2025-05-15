#代码A
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import os
import logging
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    classification_report, confusion_matrix, roc_curve, auc,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score,
    log_loss, roc_auc_score, average_precision_score,
    precision_recall_curve
)
from sklearn.calibration import calibration_curve
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import xgboost as xgb
from catboost import CatBoostClassifier
import shap
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore')

def set_figure_style():
    """Set consistent style for all plots"""
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'axes.linewidth': 2,
        'grid.linewidth': 1.0,
        'lines.linewidth': 2,
        'xtick.major.width': 2,
        'ytick.major.width': 2,
        'xtick.major.size': 5.0,
        'ytick.major.size': 5.0,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial']
    })

class MLPipeline:
    def __init__(self, output_dir='ml_outputs'):
        # 更全面的警告处理
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', message='.*DataFrame is highly fragmented.*')
        warnings.filterwarnings('ignore', message='.*Precision and F-score are ill-defined.*')
        warnings.filterwarnings('ignore', message='.*No data for colormapping provided.*')
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = f"{output_dir}_{self.timestamp}"
        self.setup_directories()
        self.setup_logging()
        set_figure_style()
        sns.set_theme(style="whitegrid")
        
    def setup_directories(self):
        """Create necessary output directories"""
        self.plots_dir = os.path.join(self.output_dir, 'plots')
        self.logs_dir = os.path.join(self.output_dir, 'logs')
        self.model_dir = os.path.join(self.output_dir, 'models')
        self.shap_dir = os.path.join(self.output_dir, 'shap_analysis')
        self.feature_importance_dir = os.path.join(self.output_dir, 'feature_importance')
        self.data_dir = os.path.join(self.output_dir, 'plot_data')  # 新增数据保存目录
        
        for directory in [self.output_dir, self.plots_dir, self.logs_dir, 
                        self.model_dir, self.shap_dir, self.feature_importance_dir,
                        self.data_dir]:
            os.makedirs(directory, exist_ok=True)
            
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            filename=os.path.join(self.logs_dir, 'pipeline.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def load_data(self, file_path):
        """Load and preprocess data"""
        try:
            # Handle different file formats
            if file_path.endswith('.csv'):
                self.dataset = pd.read_csv(file_path)
            elif file_path.endswith(('.xls', '.xlsx')):
                self.dataset = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format")
                
            logging.info(f"Successfully loaded data from {file_path}")
            
            # Prepare training data
            self.X = self.dataset.iloc[:, :-1]
            self.y = self.dataset.iloc[:, -1]
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
            )
            
            # Scale features - 保存scaler对象以便后续还原数据尺度
            self.scaler = StandardScaler()
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            
            logging.info("Data preprocessing completed")
            
        except Exception as e:
            logging.error(f"Error in data loading: {str(e)}")
            raise


    def initialize_models(self):
        """Initialize all models with their parameter grids"""
        param_grids = {
            'MLP': {
                'hidden_layer_sizes': [(50,), (100,), (50,25), (100,50)],
                'activation': ['relu', 'tanh'],
                'solver': ['adam'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01],
                'max_iter': [2000],
                'early_stopping': [True],
                'n_iter_no_change': [10]
            },
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced']
            },
            'XGBoost': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8],
                'colsample_bytree': [0.8],
                'scale_pos_weight': [1]
            },
            'AdaBoost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1.0],
                'algorithm': ['SAMME']
            },
            'CatBoost': {
                'iterations': [1000],
                'depth': [6, 8, 10],
                'learning_rate': [0.01, 0.1],
                'l2_leaf_reg': [1, 3, 5],
                'border_count': [128],
                'bagging_temperature': [1.0]
            },
            'Logistic': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [5000]
            },
            'SVM': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf'],
                'probability': [True]
            },
            'NaiveBayes': {},
            'KNN': {  # 新增KNN模型的参数网格
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        }
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        self.models = {
            'MLP': GridSearchCV(MLPClassifier(random_state=42), param_grids['MLP'], 
                            cv=cv, scoring='f1_macro', n_jobs=-1, verbose=1),
            'Random Forest': GridSearchCV(RandomForestClassifier(random_state=42), 
                                        param_grids['Random Forest'], cv=cv, scoring='f1_macro', n_jobs=-1, verbose=1),
            'XGBoost': GridSearchCV(xgb.XGBClassifier(random_state=42), 
                                param_grids['XGBoost'], cv=cv, scoring='f1_macro', n_jobs=-1, verbose=1),
            'AdaBoost': GridSearchCV(AdaBoostClassifier(random_state=42), 
                                param_grids['AdaBoost'], cv=cv, scoring='f1_macro', n_jobs=-1, verbose=1),
            'CatBoost': GridSearchCV(CatBoostClassifier(random_state=42, verbose=100), 
                                param_grids['CatBoost'], cv=cv, scoring='f1_macro', n_jobs=-1, verbose=2),
            'Logistic': GridSearchCV(LogisticRegression(random_state=42), 
                                param_grids['Logistic'], cv=cv, scoring='f1_macro', n_jobs=-1, verbose=1),
            'SVM': GridSearchCV(SVC(random_state=42), param_grids['SVM'], 
                            cv=cv, scoring='f1_macro', n_jobs=-1, verbose=1),
            'NaiveBayes': GridSearchCV(GaussianNB(), param_grids['NaiveBayes'], 
                                    cv=cv, scoring='f1_macro', n_jobs=-1, verbose=1),
            'KNN': GridSearchCV(KNeighborsClassifier(), param_grids['KNN'],  # 新增KNN模型
                            cv=cv, scoring='f1_macro', n_jobs=-1, verbose=1)
        }

    def calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive set of evaluation metrics"""
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Balanced_Accuracy': balanced_accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='weighted'),
            'Recall': recall_score(y_true, y_pred, average='weighted'),
            'F1_Score': f1_score(y_true, y_pred, average='weighted'),
            'MCC': matthews_corrcoef(y_true, y_pred),
            'Cohen_Kappa': cohen_kappa_score(y_true, y_pred),
            'Log_Loss': log_loss(y_true, y_pred_proba),
            'ROC_AUC': roc_auc_score(y_true, y_pred_proba[:, 1]),
            'PR_AUC': average_precision_score(y_true, y_pred_proba[:, 1])
        }
        return metrics

    def train_and_evaluate(self):
        """Train all models and perform evaluation"""
        self.results = {}
        self.cv_results = {}
        self.predictions = {}
        self.probabilities = {}
        
        for name, model in self.models.items():
            logging.info(f"Training {name}...")
            try:
                model.fit(self.X_train_scaled, self.y_train)
                
                # Make predictions
                y_train_pred = model.predict(self.X_train_scaled)
                self.predictions[name] = model.predict(self.X_test_scaled)
                self.probabilities[name] = model.predict_proba(self.X_test_scaled)
                
                # Calculate metrics
                train_metrics = self.calculate_metrics(
                    self.y_train, 
                    y_train_pred,
                    model.predict_proba(self.X_train_scaled)
                )
                
                test_metrics = self.calculate_metrics(
                    self.y_test, 
                    self.predictions[name],
                    self.probabilities[name]
                )
                
                self.results[name] = {
                    'Best_Params': model.best_params_,
                    'Training_Metrics': train_metrics,
                    'Testing_Metrics': test_metrics
                }
                
                self.cv_results[name] = pd.DataFrame(model.cv_results_)
                
                # Save model if possible
                if hasattr(model.best_estimator_, 'save_model'):
                    model.best_estimator_.save_model(
                        os.path.join(self.model_dir, f'{name}_best_model.txt')
                    )
                
                logging.info(f"Completed training {name}")
                
            except Exception as e:
                logging.error(f"Error in training {name}: {str(e)}")

    def analyze_feature_importance(self):
        """Analyze and plot feature importance for supported models"""
        self.feature_importance = {}
        
        for name, model in self.models.items():
            if hasattr(model.best_estimator_, 'feature_importances_'):
                importances = model.best_estimator_.feature_importances_
                feature_importance = pd.DataFrame({
                    'feature': self.X.columns,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                self.feature_importance[name] = feature_importance
                
                # Plot top 20 features
                plt.figure(figsize=(12, 8))
                sns.barplot(data=feature_importance.head(20), x='importance', y='feature')
                plt.title(f'Top 20 Most Important Features - {name}')
                plt.tight_layout()
                plt.savefig(os.path.join(self.feature_importance_dir, f'feature_importance_{name}.png'))
                plt.close()
                
                # Save CSV
                feature_importance.to_csv(
                    os.path.join(self.feature_importance_dir, f'feature_importance_{name}.csv')
                )
                
                # Plot importance distribution
                plt.figure(figsize=(10, 6))
                plt.hist(importances, bins=50)
                plt.title(f'Feature Importance Distribution - {name}')
                plt.xlabel('Importance Score')
                plt.ylabel('Count')
                plt.savefig(
                    os.path.join(self.feature_importance_dir, f'feature_importance_dist_{name}.png')
                )
                plt.close()

    def plot_results(self):
        """Generate all visualization plots"""
        self.plot_model_performance()
        self.plot_roc_curves()
        self.plot_pr_curves()
        self.plot_confusion_matrices()
        self.plot_calibration_curves()
        
    def plot_model_performance(self):
        """Plot comparison of model performance metrics"""
        results_df = pd.DataFrame({name: res['Testing_Metrics'] 
                                for name, res in self.results.items()}).T
        
        # 保存性能指标数据
        results_df.to_csv(os.path.join(self.data_dir, 'model_performance.csv'))
        
        plt.figure(figsize=(15, 8))
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']
        x = np.arange(len(results_df.index))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            plt.bar(x + i*width, results_df[metric], width, label=metric)
        
        plt.title('Model Performance Comparison', pad=20)
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.xticks(x + width*2, results_df.index, rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'model_performance_comparison.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_confusion_matrices(self):
            """Plot confusion matrices for all models"""
            n_models = len(self.models)
            n_cols = 3
            n_rows = (n_models + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6*n_rows))
            axes = axes.ravel()
            cmap = sns.color_palette("Blues")
            
            for idx, (name, model) in enumerate(self.models.items()):
                if idx < len(axes):
                    cm = confusion_matrix(self.y_test, self.predictions[name])
                    sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx], cmap=cmap)
                    axes[idx].set_title(f'{name} Confusion Matrix')
                    axes[idx].set_xlabel('Predicted')
                    axes[idx].set_ylabel('True')
            
            for idx in range(len(self.models), len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'confusion_matrices.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close()

    def plot_roc_curves(self):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        roc_data = {}
        
        for name in self.models.keys():
            try:
                fpr, tpr, _ = roc_curve(self.y_test, self.probabilities[name][:, 1])
                roc_auc = auc(fpr, tpr)
                roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
                
                # 保存每个模型的ROC数据
                pd.DataFrame({'fpr': fpr, 'tpr': tpr}).to_csv(
                    os.path.join(self.data_dir, f'roc_curve_{name}.csv'), index=False
                )
                
                plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
            except Exception as e:
                logging.error(f"Error plotting ROC curve for {name}: {str(e)}")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, 'roc_curves.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_calibration_curves(self):
        """Plot calibration curves for all models"""
        plt.figure(figsize=(10, 8))
        calibration_data = {}
        
        for name in self.models.keys():
            try:
                prob_pos = self.probabilities[name][:, 1]
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    self.y_test, prob_pos, n_bins=10
                )
                calibration_data[name] = {
                    'fraction_of_positives': fraction_of_positives,
                    'mean_predicted_value': mean_predicted_value
                }
                
                # 保存校准曲线数据
                pd.DataFrame({
                    'mean_predicted_value': mean_predicted_value,
                    'fraction_of_positives': fraction_of_positives
                }).to_csv(os.path.join(self.data_dir, f'calibration_curve_{name}.csv'), index=False)
                
                plt.plot(mean_predicted_value, fraction_of_positives, 
                        's-', label=name)
            except Exception as e:
                logging.error(f"Error plotting calibration curve for {name}: {str(e)}")
        
        plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title('Calibration Curves')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, 'calibration_curves.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_pr_curves(self):
        """Plot Precision-Recall curves for all models"""
        plt.figure(figsize=(10, 8))
        pr_data = {}
        
        for name in self.models.keys():
            precision, recall, _ = precision_recall_curve(
                self.y_test, 
                self.probabilities[name][:, 1]
            )
            avg_precision = average_precision_score(
                self.y_test, 
                self.probabilities[name][:, 1]
            )
            pr_data[name] = {
                'precision': precision,
                'recall': recall,
                'avg_precision': avg_precision
            }
            
            # 保存PR曲线数据
            pd.DataFrame({
                'precision': precision,
                'recall': recall
            }).to_csv(os.path.join(self.data_dir, f'pr_curve_{name}.csv'), index=False)
            
            plt.plot(recall, precision, 
                    label=f'{name} (AP = {avg_precision:.2f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, 'precision_recall_curves.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

    def perform_shap_analysis(self):
        """执行SHAP分析，使用自定义方法确保正确显示原始数据尺度"""
        try:
            # 确保输出目录存在
            os.makedirs(self.shap_dir, exist_ok=True)
            os.makedirs(os.path.join(self.shap_dir, 'html_plots'), exist_ok=True)
            
            # 准备用于SHAP计算的标准化数据
            X_display_scaled = pd.DataFrame(
                self.X_train_scaled,
                columns=self.X.columns
            ).iloc[:, 1:]  # 排除第一列
            
            # 准备用于显示的原始尺度数据 - 确保索引匹配
            X_display_original = self.X_train.iloc[:, 1:].reset_index(drop=True)  # 排除第一列并重置索引
            
            # 获取XGBoost的最佳模型并重新训练
            best_params = self.models['XGBoost'].best_params_
            xgb_model = xgb.XGBClassifier(**best_params, random_state=42)
            xgb_model.fit(X_display_scaled, self.y_train)
            
            # 创建解释器
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X_display_scaled)
            
            # 如果是二分类问题且shap_values是列表，取第一个元素
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # 4. 保存SHAP值到CSV - 使用特征的原始名称
            shap_df = pd.DataFrame(shap_values, columns=X_display_scaled.columns)
            shap_df.to_csv(os.path.join(self.shap_dir, 'shap_values.csv'), index=False)
            
            # 1. Summary Plot - 使用原始数据
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_display_original, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(self.shap_dir, 'shap_summary.png'), bbox_inches='tight', dpi=300)
            plt.close()
            
            # 2. Bar Plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_display_scaled, plot_type='bar', show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(self.shap_dir, 'shap_importance.png'), bbox_inches='tight', dpi=300)
            plt.close()
            
            # 3. Individual Feature Dependence Plots - 使用自定义方法
            for feature in X_display_original.columns:
                try:
                    feature_idx = X_display_scaled.columns.get_loc(feature)
                    
                    # 创建包含原始尺度值和对应SHAP值的DataFrame
                    dependence_data = pd.DataFrame({
                        'feature_value': X_display_original[feature].values,
                        'shap_value': shap_values[:, feature_idx]
                    })
                    
                    # 保存依赖关系数据
                    dependence_data.to_csv(os.path.join(self.shap_dir, f'dependence_{feature}_data.csv'), 
                                        index=False)
                    
                    # 绘制自定义依赖图，确保x轴显示原始尺度
                    plt.figure(figsize=(10, 6))
                    
                    # 为每个点着色（可以选择另一个特征进行着色）
                    other_feature = None
                    for other_feat in X_display_original.columns:
                        if other_feat != feature and X_display_original[other_feat].nunique() > 5:
                            other_feature = other_feat
                            break
                    
                    # 如果找到了适合着色的特征，使用它
                    if other_feature:
                        other_feat_idx = X_display_original.columns.get_loc(other_feature)
                        # 归一化值用于着色
                        norm_values = (X_display_original[other_feature] - X_display_original[other_feature].min()) / \
                                    (X_display_original[other_feature].max() - X_display_original[other_feature].min())
                        
                        plt.scatter(X_display_original[feature], shap_values[:, feature_idx], 
                                c=norm_values, cmap='viridis', alpha=0.7)
                        plt.colorbar(label=other_feature)
                    else:
                        # 否则使用默认蓝色
                        plt.scatter(X_display_original[feature], shap_values[:, feature_idx], 
                                color='blue', alpha=0.7)
                    
                    # 添加平滑线来显示趋势
                    if len(X_display_original) > 10:  # 只有当有足够的数据点时
                        # 按特征值排序用于平滑线
                        sorted_idx = X_display_original[feature].argsort()
                        window_size = min(30, max(5, len(X_display_original) // 10))
                        
                        # 使用移动平均计算平滑线
                        x_sorted = X_display_original[feature].iloc[sorted_idx]
                        y_sorted = shap_values[sorted_idx, feature_idx]
                        
                        # 创建移动平均线
                        from scipy.ndimage import gaussian_filter1d
                        x_unique, indices = np.unique(x_sorted, return_index=True)
                        if len(x_unique) > 1:  # 确保有多个唯一x值
                            y_mean = np.array([np.mean(y_sorted[x_sorted == x]) for x in x_unique])
                            
                            # 应用高斯平滑
                            if len(y_mean) > 3:  # 至少需要几个点才能平滑
                                y_smooth = gaussian_filter1d(y_mean, sigma=1)
                                plt.plot(x_unique, y_smooth, color='red', linewidth=2)
                    
                    # 添加水平参考线在y=0
                    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.6)
                    
                    # 添加标题和标签
                    plt.title(f'SHAP Dependence Plot for {feature}')
                    plt.xlabel(feature)
                    plt.ylabel(f'SHAP value for {feature}')
                    plt.grid(True, alpha=0.3)
                    
                    # 保存图片
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.shap_dir, f'dependence_{feature}.png'), 
                            bbox_inches='tight', dpi=300)
                    plt.close()
                    
                except Exception as e:
                    logging.error(f"Error creating dependence plot for {feature}: {str(e)}")
                    continue
            
            # 5. 只创建重要特征的交互图
            # 获取最重要的特征（基于平均绝对SHAP值）
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            top_indices = np.argsort(-mean_abs_shap)[:10]  # 取前10个最重要特征
            top_features = [X_display_original.columns[i] for i in top_indices]
            
            # 为重要特征绘制交互图
            for i, feat1 in enumerate(top_features):
                for j in range(i+1, len(top_features)):
                    feat2 = top_features[j]
                    try:
                        plt.figure(figsize=(10, 6))
                        
                        # 获取特征索引
                        feat1_idx = X_display_original.columns.get_loc(feat1)
                        feat2_idx = X_display_original.columns.get_loc(feat2)
                        
                        # 创建散点图，X轴是特征1，Y轴是特征1的SHAP值，颜色表示特征2
                        sc = plt.scatter(X_display_original[feat1], shap_values[:, feat1_idx],
                                    c=X_display_original[feat2], cmap='viridis', alpha=0.7)
                        
                        # 添加颜色条
                        cbar = plt.colorbar(sc)
                        cbar.set_label(feat2)
                        
                        # 添加平滑线
                        # (这里可以添加类似上面的平滑线代码)
                        
                        # 添加水平参考线
                        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.6)
                        
                        # 添加标题和标签
                        plt.title(f'SHAP Interaction: {feat1} vs {feat2}')
                        plt.xlabel(feat1)
                        plt.ylabel(f'SHAP value for {feat1}')
                        plt.grid(True, alpha=0.3)
                        
                        # 保存图片
                        plt.tight_layout()
                        plt.savefig(
                            os.path.join(self.shap_dir, f'interaction_{feat1}_vs_{feat2}.png'),
                            bbox_inches='tight', dpi=300
                        )
                        plt.close()
                    except Exception as e:
                        logging.error(f"Error creating interaction plot for {feat1} vs {feat2}: {str(e)}")
                        continue
            
            logging.info("SHAP analysis completed successfully")
            
        except Exception as e:
            logging.error(f"Error in SHAP analysis: {str(e)}")
            raise

    def save_results(self):
        """Save all results and generate comprehensive report"""
        # Prepare results dataframe
        results_df = pd.DataFrame({name: {
            **{'Best_Params': res['Best_Params']},
            **{f'Train_{k}': v for k, v in res['Training_Metrics'].items()},
            **{f'Test_{k}': v for k, v in res['Testing_Metrics'].items()}
        } for name, res in self.results.items()}).T
        
        # Save to Excel with multiple sheets
        with pd.ExcelWriter(os.path.join(self.output_dir, 'complete_results.xlsx')) as writer:
            results_df.to_excel(writer, sheet_name='Model_Performance')
            for name, cv_result in self.cv_results.items():
                cv_result.to_excel(writer, sheet_name=f'CV_Results_{name}')
        
        # Generate comprehensive report
        with open(os.path.join(self.output_dir, 'comprehensive_report.txt'), 'w') as f:
            f.write("Machine Learning Pipeline Comprehensive Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Dataset Information
            f.write("1. Dataset Information\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total samples: {len(self.X)}\n")
            f.write(f"Features: {len(self.X.columns)}\n")
            f.write(f"Feature names: {', '.join(self.X.columns)}\n")
            f.write(f"Class distribution:\n{self.y.value_counts()}\n\n")
            
            # Model Performance
            f.write("2. Model Performance Summary\n")
            f.write("-" * 30 + "\n")
            f.write(str(results_df))
            f.write("\n\n")
            
            # Feature Importance
            if hasattr(self, 'feature_importance'):
                f.write("3. Feature Importance Analysis\n")
                f.write("-" * 30 + "\n")
                for name, importance_df in self.feature_importance.items():
                    f.write(f"\n{name} Top 10 Features:\n")
                    f.write(str(importance_df.head(10)))
                    f.write("\n")
            
            # Detailed Model Results
            f.write("\n4. Detailed Model Results\n")
            f.write("-" * 30 + "\n")
            for name, result in self.results.items():
                f.write(f"\n{name} Model:\n")
                f.write(f"Best Parameters: {result['Best_Params']}\n")
                f.write("\nTraining Metrics:\n")
                for metric, value in result['Training_Metrics'].items():
                    f.write(f"- {metric}: {value:.4f}\n")
                f.write("\nTesting Metrics:\n")
                for metric, value in result['Testing_Metrics'].items():
                    f.write(f"- {metric}: {value:.4f}\n")
                f.write("\nClassification Report:\n")
                f.write(classification_report(self.y_test, self.predictions[name]))
                f.write("\n" + "="*50 + "\n")

    def run_pipeline(self, data_path):
        """Execute the complete machine learning pipeline"""
        try:
            logging.info("Starting ML pipeline")
            
            # Load and preprocess data
            self.load_data(data_path)
            logging.info("Data loaded successfully")
            
            # Initialize models
            self.initialize_models()
            logging.info("Models initialized")
            
            # Train and evaluate models
            self.train_and_evaluate()
            logging.info("Model training and evaluation completed")
            
            # Analyze feature importance
            self.analyze_feature_importance()
            logging.info("Feature importance analysis completed")
            
            # Generate visualizations
            self.plot_results()
            logging.info("Visualization plots generated")
            
            # Perform SHAP analysis
            self.perform_shap_analysis()
            logging.info("SHAP analysis completed")
            
            # Save results
            self.save_results()
            logging.info("Results saved")
            
            logging.info("Pipeline completed successfully")
            
            # Return results summary
            return pd.DataFrame(self.results).T
            
        except Exception as e:
            logging.error(f"Pipeline failed: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    pipeline = MLPipeline(output_dir='2-3999ml_results')
    results = pipeline.run_pipeline('3999enhanced_optimized_labels.csv')
    
    print("\nModel Performance Summary:")
    print(results)