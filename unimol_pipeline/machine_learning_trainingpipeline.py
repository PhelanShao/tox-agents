# machine_learning
import json
import os
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
import warnings

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None
    warnings.warn("CatBoost is not installed; CatBoost model will be skipped.", ImportWarning)

try:
    import shap
except ImportError:
    shap = None
    warnings.warn("shap is not installed; SHAP analysis will be skipped.", ImportWarning)

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
    def __init__(
        self,
        output_dir='ml_outputs',
        target_column=None,
        drop_columns=None,
        feature_metadata=None,
        feature_selection=None,
    ):
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', message='.*DataFrame is highly fragmented.*')
        warnings.filterwarnings('ignore', message='.*Precision and F-score are ill-defined.*')
        warnings.filterwarnings('ignore', message='.*No data for colormapping provided.*')

        output_dir = Path(output_dir)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if output_dir.name:
            self.output_dir = output_dir.parent / f"{output_dir.name}_{self.timestamp}"
        else:
            self.output_dir = Path.cwd() / f"ml_outputs_{self.timestamp}"

        self.target_column = target_column
        self.drop_columns = set(drop_columns or [])
        self.feature_metadata = feature_metadata or {}
        # feature selection configuration
        self.feature_selection = {
            'drop_constant': True,            # drop zero-variance features
            'fp_min_support': 10,             # min count or fraction for FP bits
            'corr_threshold': 0.95,           # Pearson threshold among non-fingerprint features
            'l1_select': False,               # optionally enable L1-based selection
            'l1_top_k': None,                 # keep top-k by |coef| if provided; otherwise non-zero
            'l1_C': 0.1,                      # regularization strength for L1 LR
        }
        if feature_selection:
            self.feature_selection.update(feature_selection)

        self.train_indices = []
        self.test_indices = []

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
        self.data_dir = os.path.join(self.output_dir, 'plot_data')
        
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

    def load_data(self, data_source):
        """Load and preprocess data"""
        try:
            source_description = 'DataFrame input'
            if isinstance(data_source, pd.DataFrame):
                self.dataset = data_source.copy()
            else:
                file_path = Path(data_source)
                if not file_path.exists():
                    raise FileNotFoundError(f"Data file not found: {file_path}")

                if file_path.suffix.lower() == '.csv':
                    self.dataset = pd.read_csv(file_path)
                elif file_path.suffix.lower() in {'.xls', '.xlsx'}:
                    self.dataset = pd.read_excel(file_path)
                else:
                    raise ValueError("Unsupported file format")
                source_description = str(file_path)

            logging.info(f"Successfully loaded data from {source_description}")

            if self.target_column and self.target_column in self.dataset.columns:
                target_col = self.target_column
            else:
                target_col = self.dataset.columns[-1]
                self.target_column = target_col

            self.y = self.dataset[target_col]
            feature_df = self.dataset.drop(columns=[target_col], errors='ignore')

            if self.drop_columns:
                feature_df = feature_df.drop(
                    columns=[col for col in self.drop_columns if col in feature_df.columns],
                    errors='ignore'
                )

            numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
            non_numeric = set(feature_df.columns) - set(numeric_cols)
            if non_numeric:
                logging.warning(f"Dropping non-numeric feature columns: {sorted(non_numeric)}")
            feature_df = feature_df[numeric_cols]

            self.X = feature_df

            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
            )

            self.train_indices = list(self.X_train.index)
            self.test_indices = list(self.X_test.index)

            # Apply feature selection based on training set only
            self.apply_feature_selection()

            self.scaler = StandardScaler()
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)

            logging.info("Data preprocessing completed")

        except Exception as e:
            logging.error(f"Error in data loading: {str(e)}")
            raise

    def _is_fingerprint_column(self, col: str) -> bool:
        return isinstance(col, str) and col.startswith('FP_')

    def apply_feature_selection(self):
        """Apply a transparent, training-only feature screening pipeline.

        Steps:
        - Drop constant/near-constant features (train set)
        - Filter fingerprint bits by minimum support (train set)
        - Remove highly correlated non-fingerprint features (train set)
        - Optional L1-logistic selection (train set)
        Saves a report under output_dir.
        """
        fs_cfg = self.feature_selection

        # Ensure unique column names to avoid ambiguous DataFrame indexing
        cols = pd.Index(self.X_train.columns)
        if cols.has_duplicates:
            rename_map_train = {}
            rename_map_test = {}
            seen = {}
            new_cols_train = []
            for c in self.X_train.columns:
                if c in seen:
                    seen[c] += 1
                    new_c = f"{c}__dup{seen[c]}"
                else:
                    seen[c] = 0
                    new_c = c
                new_cols_train.append(new_c)
            self.X_train.columns = new_cols_train

            # apply same deterministic renaming to test columns
            seen = {}
            new_cols_test = []
            for c in self.X_test.columns:
                if c in seen:
                    seen[c] += 1
                    new_c = f"{c}__dup{seen[c]}"
                else:
                    seen[c] = 0
                    new_c = c
                new_cols_test.append(new_c)
            self.X_test.columns = new_cols_test

            try:
                with open(Path(self.output_dir) / 'duplicate_columns_renamed.txt', 'w', encoding='utf-8') as f:
                    for old_name, new_name in zip(cols, self.X_train.columns):
                        if old_name != new_name:
                            f.write(f"{old_name} -> {new_name}\n")
            except Exception:
                pass

        kept = list(self.X_train.columns)
        dropped_constant = []
        dropped_fp_low_support = []
        dropped_correlated = []  # tuples (col_drop, col_keep, corr)

        # 1) Constant features
        if fs_cfg.get('drop_constant', True):
            for col in list(kept):
                if self.X_train[col].nunique(dropna=False) <= 1:
                    kept.remove(col)
                    dropped_constant.append(col)

        # 2) Fingerprint support filter
        fp_cols = [c for c in kept if self._is_fingerprint_column(c)]
        if fp_cols:
            support = (self.X_train[fp_cols] > 0).sum(axis=0)
            min_sup = fs_cfg.get('fp_min_support', 10)
            if isinstance(min_sup, float) and 0 < min_sup < 1:
                min_count = int(np.ceil(min_sup * len(self.X_train)))
            else:
                min_count = int(min_sup)
            low_support = support[support < min_count].index.tolist()
            for col in low_support:
                if col in kept:
                    kept.remove(col)
                    dropped_fp_low_support.append((col, int(support[col])))

        # 3) Correlation filter among non-fingerprint features
        physchem_cols = [c for c in kept if not self._is_fingerprint_column(c)]
        corr_thr = fs_cfg.get('corr_threshold', 0.95)
        if physchem_cols and len(physchem_cols) > 1 and corr_thr < 1.0:
            df_phys = self.X_train[physchem_cols]
            corr = df_phys.corr().abs()
            # correlation to target (point-biserial equals Pearson for binary y)
            y_arr = self.y_train.values.astype(float)
            corr_to_y = {}
            for col in physchem_cols:
                x = df_phys[col].values.astype(float)
                if np.std(x) == 0:
                    corr_to_y[col] = 0.0
                else:
                    c = np.corrcoef(x, y_arr)[0, 1]
                    corr_to_y[col] = 0.0 if np.isnan(c) else float(abs(c))

            cols = list(physchem_cols)
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    cval = corr.iloc[i, j]
                    if cval >= corr_thr:
                        c1, c2 = cols[i], cols[j]
                        # drop the one less associated with target
                        drop = c1 if corr_to_y.get(c1, 0.0) < corr_to_y.get(c2, 0.0) else c2
                        keep = c2 if drop == c1 else c1
                        if drop in kept:
                            kept.remove(drop)
                            dropped_correlated.append((drop, keep, float(cval)))

        # 4) Optional L1-based selection
        if fs_cfg.get('l1_select', False) and kept:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            Xtr = self.X_train[kept].values
            scaler = StandardScaler()
            Xtr_sc = scaler.fit_transform(Xtr)
            lr = LogisticRegression(penalty='l1', solver='liblinear', C=fs_cfg.get('l1_C', 0.1), max_iter=5000)
            try:
                lr.fit(Xtr_sc, self.y_train.values)
                coefs = np.abs(lr.coef_.ravel())
                if fs_cfg.get('l1_top_k'):
                    k = int(fs_cfg['l1_top_k'])
                    idx = np.argsort(coefs)[::-1][:k]
                    kept = [kept[i] for i in sorted(idx)]
                else:
                    idx = np.where(coefs > 0)[0]
                    if len(idx) == 0:
                        # fallback to top 256
                        idx = np.argsort(coefs)[::-1][: min(256, len(kept))]
                    kept = [kept[i] for i in sorted(idx)]
            except Exception as e:
                logging.warning(f"L1 selection skipped due to error: {e}")

        # Apply column subset
        self.X_train = self.X_train[kept]
        self.X_test = self.X_test[kept]
        self.feature_columns = kept

        # Persist report
        report = {
            'n_features_initial': int(self.X.shape[1]),
            'n_features_after_selection': int(len(kept)),
            'dropped_constant': dropped_constant,
            'dropped_fp_low_support': [{'column': c, 'support': s} for c, s in dropped_fp_low_support],
            'correlation_threshold': corr_thr,
            'dropped_correlated': [
                {'dropped': d, 'kept': k, 'corr': c} for d, k, c in dropped_correlated
            ],
            'l1_select': bool(fs_cfg.get('l1_select', False)),
            'l1_top_k': fs_cfg.get('l1_top_k'),
            'l1_C': fs_cfg.get('l1_C'),
        }
        try:
            with open(Path(self.output_dir) / 'feature_selection_report.json', 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            # Simple text lists for quick inspection
            (Path(self.output_dir) / 'selected_features.txt').write_text('\n'.join(kept))
        except Exception as e:
            logging.warning(f"Failed saving feature selection artifacts: {e}")


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
        }

        if CatBoostClassifier is not None:
            self.models['CatBoost'] = GridSearchCV(
                CatBoostClassifier(random_state=42, verbose=100),
                param_grids['CatBoost'],
                cv=cv,
                scoring='f1_macro',
                n_jobs=-1,
                verbose=2
            )

        self.models.update({
            'Logistic': GridSearchCV(LogisticRegression(random_state=42), 
                                param_grids['Logistic'], cv=cv, scoring='f1_macro', n_jobs=-1, verbose=1),
            'SVM': GridSearchCV(SVC(random_state=42), param_grids['SVM'], 
                            cv=cv, scoring='f1_macro', n_jobs=-1, verbose=1),
            'NaiveBayes': GridSearchCV(GaussianNB(), param_grids['NaiveBayes'], 
                                    cv=cv, scoring='f1_macro', n_jobs=-1, verbose=1),
            'KNN': GridSearchCV(KNeighborsClassifier(), param_grids['KNN'],  # 新增KNN模型
                            cv=cv, scoring='f1_macro', n_jobs=-1, verbose=1)
        })

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
        self.fitted_models = {}

        for name, model in self.models.items():
            logging.info(f"Training {name}...")
            try:
                model.fit(self.X_train_scaled, self.y_train)

                # Make predictions
                y_train_pred = model.predict(self.X_train_scaled)
                self.predictions[name] = model.predict(self.X_test_scaled)
                self.probabilities[name] = model.predict_proba(self.X_test_scaled)
                self.fitted_models[name] = model
                
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

        if not self.fitted_models:
            logging.warning("No fitted models available for feature importance analysis")
            return

        for name, model in self.fitted_models.items():
            if hasattr(model.best_estimator_, 'feature_importances_'):
                importances = model.best_estimator_.feature_importances_
                feature_importance = pd.DataFrame({
                    'feature': self.feature_columns,
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
        if not self.results:
            logging.warning("No successful models to plot; skipping visualization generation")
            return

        self.plot_model_performance()
        self.plot_roc_curves()
        self.plot_pr_curves()
        self.plot_confusion_matrices()
        self.plot_calibration_curves()
        
    def plot_model_performance(self):
        """Plot comparison of model performance metrics"""
        results_df = pd.DataFrame({name: res['Testing_Metrics'] 
                                for name, res in self.results.items()}).T
        if results_df.empty:
            logging.warning("Model performance DataFrame is empty; skipping performance plot")
            return

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
            n_models = len(self.predictions)
            if n_models == 0:
                logging.warning("No predictions available; skipping confusion matrix plotting")
                return

            n_cols = 3
            n_rows = (n_models + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6*n_rows))
            axes = np.atleast_1d(axes).ravel()
            cmap = sns.color_palette("Blues")

            for idx, (name, preds) in enumerate(self.predictions.items()):
                if idx < len(axes):
                    cm = confusion_matrix(self.y_test, preds)
                    sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx], cmap=cmap)
                    axes[idx].set_title(f'{name} Confusion Matrix')
                    axes[idx].set_xlabel('Predicted')
                    axes[idx].set_ylabel('True')

            for idx in range(len(self.predictions), len(axes)):
                axes[idx].set_visible(False)

            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'confusion_matrices.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close()

    def plot_roc_curves(self):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        roc_data = {}
        
        if not self.probabilities:
            logging.warning("No probabilities stored; skipping ROC curves")
            plt.close()
            return

        for name, probs in self.probabilities.items():
            try:
                fpr, tpr, _ = roc_curve(self.y_test, probs[:, 1])
                roc_auc = auc(fpr, tpr)
                roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}

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
        
        if not self.probabilities:
            logging.warning("No probabilities stored; skipping calibration curves")
            plt.close()
            return

        for name, probs in self.probabilities.items():
            try:
                prob_pos = probs[:, 1]
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    self.y_test, prob_pos, n_bins=10
                )
                calibration_data[name] = {
                    'fraction_of_positives': fraction_of_positives,
                    'mean_predicted_value': mean_predicted_value
                }
                
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
        
        if not self.probabilities:
            logging.warning("No probabilities stored; skipping PR curves")
            plt.close()
            return

        for name, probs in self.probabilities.items():
            precision, recall, _ = precision_recall_curve(
                self.y_test, 
                probs[:, 1]
            )
            avg_precision = average_precision_score(
                self.y_test, 
                probs[:, 1]
            )
            pr_data[name] = {
                'precision': precision,
                'recall': recall,
                'avg_precision': avg_precision
            }
            
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

    def _first_sample_with_feature(self, column_name):
        column_series = self.X[column_name]
        matches = column_series[column_series > 0]
        if matches.empty:
            return None
        return matches.index[0]

    def _fingerprint_fragments_for_bit(self, sample_id, bit_idx, fingerprint_meta):
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except ImportError:
            logging.error("RDKit is required for fingerprint interpretability but is not available")
            return []

        smiles = fingerprint_meta.get('sample_smiles', {}).get(str(sample_id))
        if not smiles:
            return []

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        bit_info = {}
        AllChem.GetMorganFingerprintAsBitVect(
            mol,
            fingerprint_meta.get('radius', 2),
            nBits=fingerprint_meta.get('n_bits', 2048),
            useChirality=fingerprint_meta.get('use_chirality', True),
            bitInfo=bit_info,
        )

        fragments = []
        for atom_idx, radius in bit_info.get(bit_idx, []):
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
            atom_indices = {atom_idx}
            for bond_idx in env:
                bond = mol.GetBondWithIdx(bond_idx)
                atom_indices.add(bond.GetBeginAtomIdx())
                atom_indices.add(bond.GetEndAtomIdx())
            atoms_tuple = tuple(sorted(atom_indices))
            try:
                if env:
                    fragment = Chem.MolFragmentToSmiles(
                        mol,
                        atoms=atoms_tuple,
                        bondsToUse=env,
                        canonical=True,
                    )
                else:
                    fragment = Chem.MolFragmentToSmiles(
                        mol,
                        atoms=atoms_tuple,
                        canonical=True,
                    )
                fragments.append(fragment)
            except Exception as frag_err:
                logging.error(
                    "Failed to extract fragment for bit %s in sample %s: %s",
                    bit_idx,
                    sample_id,
                    frag_err,
                )
        return sorted(set(fragments))

    def generate_fingerprint_interpretability(self, top_n=15, per_sample_top=5):
        fingerprint_meta = self.feature_metadata.get('fingerprint')
        if not fingerprint_meta:
            logging.info("No fingerprint metadata provided; skipping fingerprint interpretability")
            return

        logistic_cv = self.fitted_models.get('Logistic')
        if logistic_cv is None:
            logging.warning("Logistic model unavailable for fingerprint interpretability output")
            return

        logistic_model = getattr(logistic_cv, 'best_estimator_', None)
        if logistic_model is None or not hasattr(logistic_model, 'coef_'):
            logging.warning("Unable to extract coefficients from logistic model for interpretability")
            return

        coef = logistic_model.coef_
        if coef.ndim > 1:
            coef = coef[0]
        intercept = float(getattr(logistic_model, 'intercept_', [0.0])[0])

        # interpret features present after selection
        feature_names = list(self.X_train.columns)
        column_to_index = {col: idx for idx, col in enumerate(feature_names)}
        column_to_bit = fingerprint_meta.get('column_to_bit', {})
        fp_columns = [col for col in feature_names if col in column_to_bit]
        if not fp_columns:
            logging.warning("No fingerprint feature columns detected; skipping interpretability")
            return

        fp_indices = np.array([column_to_index[col] for col in fp_columns])
        fp_bits = np.array([column_to_bit[col] for col in fp_columns])

        scaler_scale = getattr(self.scaler, 'scale_', None)
        scaler_mean = getattr(self.scaler, 'mean_', None)
        if scaler_scale is not None:
            scaler_scale = np.asarray(scaler_scale)
        if scaler_mean is not None:
            scaler_mean = np.asarray(scaler_mean)

        if scaler_scale is not None:
            scale_subset = np.where(scaler_scale[fp_indices] == 0, 1.0, scaler_scale[fp_indices])
            coef_fp_raw = coef[fp_indices] / scale_subset
            if scaler_mean is not None:
                mean_subset = scaler_mean[fp_indices]
                intercept_raw = intercept - float(np.sum(coef[fp_indices] * (mean_subset / scale_subset)))
            else:
                intercept_raw = intercept
        else:
            coef_fp_raw = coef[fp_indices]
            intercept_raw = intercept

        coef_fp_scaled = coef[fp_indices]
        support_series = self.X[fp_columns].sum().astype(int)

        global_rows = []
        for col, bit, coef_raw_val, coef_scaled_val in zip(fp_columns, fp_bits, coef_fp_raw, coef_fp_scaled):
            global_rows.append(
                {
                    'bit': int(bit),
                    'column': col,
                    'coefficient_raw': float(coef_raw_val),
                    'coefficient_scaled': float(coef_scaled_val),
                    'support': int(support_series[col]),
                }
            )

        global_df = pd.DataFrame(global_rows)
        fragment_cache = {}

        def cached_fragments(sample_id, bit):
            key = (str(sample_id), int(bit))
            if key not in fragment_cache:
                fragment_cache[key] = self._fingerprint_fragments_for_bit(sample_id, bit, fingerprint_meta)
            return fragment_cache[key]

        top_positive = []
        for _, row in global_df.sort_values('coefficient_raw', ascending=False).head(top_n).iterrows():
            sample_id = self._first_sample_with_feature(row['column'])
            fragments = cached_fragments(sample_id, row['bit']) if sample_id is not None else []
            top_positive.append(
                {
                    'bit': int(row['bit']),
                    'column': row['column'],
                    'coefficient_raw': float(row['coefficient_raw']),
                    'coefficient_scaled': float(row['coefficient_scaled']),
                    'support': int(row['support']),
                    'example_sample': str(sample_id) if sample_id is not None else None,
                    'example_fragments': fragments,
                }
            )

        top_negative = []
        for _, row in global_df.sort_values('coefficient_raw', ascending=True).head(top_n).iterrows():
            sample_id = self._first_sample_with_feature(row['column'])
            fragments = cached_fragments(sample_id, row['bit']) if sample_id is not None else []
            top_negative.append(
                {
                    'bit': int(row['bit']),
                    'column': row['column'],
                    'coefficient_raw': float(row['coefficient_raw']),
                    'coefficient_scaled': float(row['coefficient_scaled']),
                    'support': int(row['support']),
                    'example_sample': str(sample_id) if sample_id is not None else None,
                    'example_fragments': fragments,
                }
            )

        sample_smiles = fingerprint_meta.get('sample_smiles', {})
        local_entries = []
        probabilities = None
        if hasattr(logistic_model, 'predict_proba'):
            probabilities = logistic_model.predict_proba(self.X_test_scaled)[:, 1]
        else:
            logits = logistic_model.decision_function(self.X_test_scaled)
            probabilities = 1.0 / (1.0 + np.exp(-logits))

        fp_columns_np = np.array(fp_columns)
        fp_bits_np = np.array(fp_bits)
        coef_fp_raw_np = np.array(coef_fp_raw)

        for idx, sample_id in enumerate(self.X_test.index):
            row_values = self.X_test.loc[sample_id, fp_columns_np].values.astype(float)
            active_mask = row_values > 0
            if not np.any(active_mask):
                continue

            contributions = coef_fp_raw_np[active_mask] * row_values[active_mask]
            bits_active = fp_bits_np[active_mask]
            cols_active = fp_columns_np[active_mask]
            order = np.argsort(np.abs(contributions))[::-1][:per_sample_top]

            top_bits = []
            for pos in order:
                contribution_value = float(contributions[pos])
                if abs(contribution_value) < 1e-9:
                    continue
                bit_idx = int(bits_active[pos])
                column_name = cols_active[pos]
                fragments = cached_fragments(sample_id, bit_idx)
                top_bits.append(
                    {
                        'bit': bit_idx,
                        'column': column_name,
                        'contribution': contribution_value,
                        'fragments': fragments,
                    }
                )

            local_entries.append(
                {
                    'sample_id': str(sample_id),
                    'smiles': sample_smiles.get(str(sample_id)),
                    'true_label': int(self.y.loc[sample_id]) if sample_id in self.y.index else None,
                    'predicted_probability': float(probabilities[idx]),
                    'top_contributions': top_bits,
                }
            )

        interpretation = {
            'logistic': {
                'global': {
                    'top_positive': top_positive,
                    'top_negative': top_negative,
                },
                'local': local_entries,
                'metadata': {
                    'intercept_raw': float(intercept_raw),
                    'radius': fingerprint_meta.get('radius'),
                    'n_bits': fingerprint_meta.get('n_bits'),
                },
            }
        }

        output_path = Path(self.output_dir) / 'fingerprint_interpretability.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(interpretation, f, indent=2)

        logging.info("Fingerprint interpretability artifacts saved to %s", output_path)

    def perform_shap_analysis(self):
        if shap is None:
            logging.warning("shap library not available; skipping SHAP analysis")
            return

        try:
            os.makedirs(self.shap_dir, exist_ok=True)
            os.makedirs(os.path.join(self.shap_dir, 'html_plots'), exist_ok=True)

            if 'XGBoost' not in self.fitted_models:
                logging.warning("XGBoost model unavailable; skipping SHAP analysis")
                return
            
            X_display_scaled = pd.DataFrame(
                self.X_train_scaled,
                columns=self.feature_columns
            )
            
            X_display_original = self.X_train.reset_index(drop=True)
            
            best_params = self.fitted_models['XGBoost'].best_params_
            xgb_model = xgb.XGBClassifier(**best_params, random_state=42)
            xgb_model.fit(X_display_scaled, self.y_train)
            
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X_display_scaled)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            shap_df = pd.DataFrame(shap_values, columns=X_display_scaled.columns)
            shap_df.to_csv(os.path.join(self.shap_dir, 'shap_values.csv'), index=False)
            
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
            
            for feature in X_display_original.columns:
                try:
                    feature_idx = X_display_scaled.columns.get_loc(feature)
                    
                    dependence_data = pd.DataFrame({
                        'feature_value': X_display_original[feature].values,
                        'shap_value': shap_values[:, feature_idx]
                    })
                    
                    dependence_data.to_csv(os.path.join(self.shap_dir, f'dependence_{feature}_data.csv'), 
                                        index=False)
                    
                    plt.figure(figsize=(10, 6))
                    
                    other_feature = None
                    for other_feat in X_display_original.columns:
                        if other_feat != feature and X_display_original[other_feat].nunique() > 5:
                            other_feature = other_feat
                            break
                    
                    if other_feature:
                        other_feat_idx = X_display_original.columns.get_loc(other_feature)
                        norm_values = (X_display_original[other_feature] - X_display_original[other_feature].min()) / \
                                    (X_display_original[other_feature].max() - X_display_original[other_feature].min())
                        
                        plt.scatter(X_display_original[feature], shap_values[:, feature_idx], 
                                c=norm_values, cmap='viridis', alpha=0.7)
                        plt.colorbar(label=other_feature)
                    else:
                        plt.scatter(X_display_original[feature], shap_values[:, feature_idx], 
                                color='blue', alpha=0.7)
                    
                    if len(X_display_original) > 10:  
                        sorted_idx = X_display_original[feature].argsort()
                        window_size = min(30, max(5, len(X_display_original) // 10))
                        
                        x_sorted = X_display_original[feature].iloc[sorted_idx]
                        y_sorted = shap_values[sorted_idx, feature_idx]
                        
                        from scipy.ndimage import gaussian_filter1d
                        x_unique, indices = np.unique(x_sorted, return_index=True)
                        if len(x_unique) > 1:  
                            y_mean = np.array([np.mean(y_sorted[x_sorted == x]) for x in x_unique])
                            if len(y_mean) > 3:  
                                y_smooth = gaussian_filter1d(y_mean, sigma=1)
                                plt.plot(x_unique, y_smooth, color='red', linewidth=2)
                    
                    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.6)           
                    plt.title(f'SHAP Dependence Plot for {feature}')
                    plt.xlabel(feature)
                    plt.ylabel(f'SHAP value for {feature}')
                    plt.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.shap_dir, f'dependence_{feature}.png'), 
                            bbox_inches='tight', dpi=300)
                    plt.close()
                    
                except Exception as e:
                    logging.error(f"Error creating dependence plot for {feature}: {str(e)}")
                    continue
            
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            top_indices = np.argsort(-mean_abs_shap)[:10]  
            top_features = [X_display_original.columns[i] for i in top_indices]
            
            for i, feat1 in enumerate(top_features):
                for j in range(i+1, len(top_features)):
                    feat2 = top_features[j]
                    try:
                        plt.figure(figsize=(10, 6))
                        
                        feat1_idx = X_display_original.columns.get_loc(feat1)
                        feat2_idx = X_display_original.columns.get_loc(feat2)
                        
                        sc = plt.scatter(X_display_original[feat1], shap_values[:, feat1_idx],
                                    c=X_display_original[feat2], cmap='viridis', alpha=0.7)
                        
                        cbar = plt.colorbar(sc)
                        cbar.set_label(feat2)
                                          
                        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.6)
                        
                        plt.title(f'SHAP Interaction: {feat1} vs {feat2}')
                        plt.xlabel(feat1)
                        plt.ylabel(f'SHAP value for {feat1}')
                        plt.grid(True, alpha=0.3)
                        
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
        
        excel_path = Path(self.output_dir) / 'complete_results.xlsx'
        try:
            with pd.ExcelWriter(excel_path) as writer:
                results_df.to_excel(writer, sheet_name='Model_Performance')
                for name, cv_result in self.cv_results.items():
                    cv_result.to_excel(writer, sheet_name=f'CV_Results_{name}')
        except ModuleNotFoundError:
            logging.warning('openpyxl not installed; exporting results as CSV instead of Excel')
            results_df.to_csv(Path(self.output_dir) / 'model_performance_summary.csv')
            for name, cv_result in self.cv_results.items():
                cv_result.to_csv(Path(self.output_dir) / f'cv_results_{name}.csv', index=False)
        except Exception as exc:
            logging.error(f'Failed to save Excel results: {exc}')
            raise

        # Generate comprehensive report
        with open(Path(self.output_dir) / 'comprehensive_report.txt', 'w') as f:
            f.write("Machine Learning Pipeline Comprehensive Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Dataset Information
            f.write("1. Dataset Information\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total samples: {len(self.X)}\n")
            # report on selected features
            selected_cols = getattr(self, 'feature_columns', list(self.X.columns))
            f.write(f"Features: {len(selected_cols)}\n")
            f.write(f"Feature names: {', '.join(selected_cols)}\n")
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
            
            # Generate fingerprint interpretability outputs
            self.generate_fingerprint_interpretability()
            
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
    # Example usage using local dataset
    base_dir = Path(__file__).resolve().parent
    data_file = base_dir / '7100enhanced_optimized_labels_mapped_from_7330.csv'
    if not data_file.exists():
        raise FileNotFoundError(f"Expected dataset not found: {data_file}")

    pipeline = MLPipeline(output_dir=base_dir / '7100ml_results_mapped')
    results = pipeline.run_pipeline(data_file)

    print("\nModel Performance Summary:")
    print(results)
