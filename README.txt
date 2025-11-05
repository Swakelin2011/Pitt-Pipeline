# PJK Prediction Pipeline

A comprehensive, publication-ready machine learning pipeline for PJK (Proximal Junctional Kyphosis) prediction with Bayesian optimization and ensemble methods.

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Key Features

- **Proper Train/Test Split**: Prevents data leakage for publication-ready results
- **Automated Feature Selection**: Multiple strategies including AUC-optimized selection
- **Bayesian Hyperparameter Optimization**: Uses scikit-optimize for intelligent parameter tuning
- **Ensemble Methods**: Voting, Stacking, and Weighted ensembles for improved performance
- **Feature Engineering**: Automated interaction term generation with filtering
- **Comprehensive Model Comparison**: Tests 10+ classifiers automatically
- **Variable Type Inference**: Automatically detects continuous, ordinal, and categorical variables
- **Interactive Reclassification**: Allows manual correction of variable types

## 📋 Requirements

### Core Dependencies
- Python >= 3.8
- numpy >= 1.20.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- joblib >= 1.0.0
- scipy >= 1.7.0

### Optional Dependencies (Recommended)
- xgboost >= 1.5.0 (for XGBoost models)
- lightgbm >= 3.3.0 (for LightGBM models)
- catboost >= 1.0.0 (for CatBoost models)
- scikit-optimize >= 0.9.0 (for Bayesian optimization)

## 🚀 Installation

### Basic Installation
```bash
pip install pjk-prediction-pipeline
```

### Full Installation (with all optional dependencies)
```bash
pip install pjk-prediction-pipeline[full]
```

### Installation Options
```bash
# Install with Bayesian optimization only
pip install pjk-prediction-pipeline[bayes]

# Install with gradient boosting frameworks only
pip install pjk-prediction-pipeline[boosting]

# Install for development
pip install pjk-prediction-pipeline[dev]
```

### From Source
```bash
git clone https://github.com/yourusername/pjk-prediction-pipeline.git
cd pjk-prediction-pipeline
pip install -e .
```

## 📖 Quick Start

### Basic Usage

```python
from pjk_pipeline import complete_pjk_pipeline_with_bayes_opt
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# Define target and features
target_col = 'pjk_diagnosis'
base_features = ['age', 'bmi', 'bone_density', 'surgery_type', ...]

# Run the pipeline
results = complete_pjk_pipeline_with_bayes_opt(
    df=df,
    target_col=target_col,
    base_features=base_features,
    max_features=12,
    use_bayes_opt=True,
    use_ensemble=True,
    test_size=0.2,
    random_state=42
)

# View results
print(f"Test AUC: {results['test_auc']:.4f}")
print(f"Best Model: {results['best_model_name']}")
print(f"Selected Features: {results['selected_features']}")
```

### Quick Comparison (Simplified Interface)

```python
from pjk_pipeline import quick_compare

results = quick_compare(
    df=df,
    target_col='pjk_diagnosis',
    base_features=predictor_list,
    max_features=10
)
```

## 🔧 Advanced Usage

### Bayesian Optimization

```python
results = complete_pjk_pipeline_with_bayes_opt(
    df=df,
    target_col=target_col,
    base_features=base_features,
    # Bayesian optimization settings
    use_bayes_opt=True,
    n_bayes_calls=50,  # Number of optimization iterations
    cv_folds=5,
    optimize_models=['RandomForest', 'GradientBoosting', 'XGBoost'],
    test_size=0.2,
    random_state=42
)
```

### Ensemble Methods

```python
results = complete_pjk_pipeline_with_bayes_opt(
    df=df,
    target_col=target_col,
    base_features=base_features,
    # Ensemble settings
    use_ensemble=True,
    ensemble_top_n=5,  # Use top 5 models for ensemble
    test_size=0.2,
    random_state=42
)

# Check if best model is an ensemble
if results['is_best_ensemble']:
    print("Best model is an ensemble!")
```

### Custom Feature Selection

```python
results = complete_pjk_pipeline_with_bayes_opt(
    df=df,
    target_col=target_col,
    base_features=base_features,
    # Feature selection settings
    selection_method='auc_optimized',  # or 'hybrid'
    max_features=15,
    epv=20,  # Events per variable
    min_target_corr=0.1,
    max_inter_corr=0.8,
    force_include=['age', 'bmi'],  # Always include these features
    test_size=0.2,
    random_state=42
)
```

### Comparing Strategies

```python
strategies = [
    {'use_ensemble': False, 'use_bayes_opt': False, 'name': 'Baseline'},
    {'use_ensemble': False, 'use_bayes_opt': True, 'name': 'Bayes Opt Only'},
    {'use_ensemble': True, 'use_bayes_opt': False, 'name': 'Ensemble Only'},
    {'use_ensemble': True, 'use_bayes_opt': True, 'name': 'Both'},
]

for strategy in strategies:
    result = complete_pjk_pipeline_with_bayes_opt(
        df=df,
        target_col=target_col,
        base_features=base_features,
        max_features=12,
        **{k: v for k, v in strategy.items() if k != 'name'}
    )
    print(f"{strategy['name']}: Test AUC = {result['test_auc']:.4f}")
```

## 📊 Results Dictionary

The pipeline returns a comprehensive results dictionary:

```python
{
    'X_train': pd.DataFrame,           # Training features
    'X_test': pd.DataFrame,            # Test features
    'y_train': pd.Series,              # Training target
    'y_test': pd.Series,               # Test target
    'selected_features': list,         # Selected feature names
    'best_model': object,              # Fitted best model
    'best_model_name': str,            # Name of best model
    'cv_auc_train': float,             # Cross-validation AUC on training
    'test_auc': float,                 # Final AUC on test set
    'y_test_pred_proba': np.array,     # Predicted probabilities
    'feature_importance': dict,        # Feature importance scores
    'model_params': dict,              # Model parameters
    'training_size': int,              # Number of training samples
    'test_size': int,                  # Number of test samples
    'train_events': int,               # Number of events in training
    'test_events': int,                # Number of events in test
    'bayes_opt_used': bool,            # Whether Bayes opt was used
    'optimized_params': dict,          # Optimized hyperparameters
    'ensemble_used': bool,             # Whether ensembles were tested
    'is_best_ensemble': bool,          # Whether best model is ensemble
    'ensemble_info': dict,             # Detailed ensemble results
}
```

## 🔍 Methodology

### Pipeline Steps

1. **Train/Test Split**: Data is split before any processing to prevent leakage
2. **Variable Type Inference**: Automatically detects continuous, ordinal, categorical variables
3. **Feature Engineering**: Generates interaction terms (multiplication, division)
4. **Feature Filtering**: Removes low-correlation and highly correlated features
5. **Feature Selection**: Multiple methods (univariate, model-based, AUC-optimized)
6. **Hyperparameter Optimization**: Optional Bayesian optimization for selected models
7. **Model Comparison**: Tests 10+ classifiers with cross-validation
8. **Ensemble Creation**: Optional voting, stacking, and weighted ensembles
9. **Final Evaluation**: Best model evaluated on held-out test set

### Supported Models

**Standard Models:**
- Random Forest
- Gradient Boosting
- Extra Trees
- AdaBoost
- Logistic Regression
- SVM
- Decision Tree
- Naive Bayes
- K-Nearest Neighbors

**Gradient Boosting (if installed):**
- XGBoost
- LightGBM
- CatBoost

**Ensemble Methods:**
- Voting Classifier (soft voting)
- Stacking Classifier (with Logistic Regression meta-learner)
- Weighted Ensemble (performance-based weights)

## 📈 Performance Tips

1. **Start Simple**: Use `quick_compare()` for initial exploration
2. **Bayesian Optimization**: Enable for 10-20% performance improvement (slower)
3. **Ensemble Methods**: Can provide 5-15% boost but increases computation time
4. **Feature Selection**: `auc_optimized` method often works best for classification
5. **Sample Size**: Ensure adequate events per variable (EPV >= 10-20)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this package in your research, please cite:

```bibtex
@software{pjk_pipeline,
  title = {PJK Prediction Pipeline: Bayesian Optimization and Ensemble Methods},
  author = {Your Name},
  year = {2024},
  version = {1.0.0},
  url = {https://github.com/yourusername/pjk-prediction-pipeline}
}
```

## 📞 Contact

- Author: Your Name
- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)

## 🙏 Acknowledgments

This pipeline builds upon excellent open-source libraries:
- scikit-learn
- scikit-optimize
- XGBoost, LightGBM, CatBoost
- pandas, numpy, scipy

## 📝 Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and updates.
