"""
PJK Prediction Pipeline
=======================

A comprehensive machine learning pipeline for PJK (Proximal Junctional Kyphosis) 
prediction with Bayesian optimization and ensemble methods.

Key Features:
-------------
- Proper train/test split to prevent data leakage
- Automated feature selection with multiple strategies
- Bayesian hyperparameter optimization for improved performance
- Ensemble methods (Voting, Stacking, Weighted)
- Comprehensive model comparison
- Feature interaction generation
- Variable type inference and reclassification
- Publication-ready results

Basic Usage:
-----------
>>> from pjk_pipeline import complete_pjk_pipeline_with_bayes_opt
>>> 
>>> results = complete_pjk_pipeline_with_bayes_opt(
...     df=model_frame,
...     target_col='pjk_diagnosis',
...     base_features=predictor_list,
...     max_features=12,
...     use_bayes_opt=True,
...     use_ensemble=True,
...     test_size=0.2,
...     random_state=42
... )
>>> 
>>> print(f"Test AUC: {results['test_auc']:.4f}")
>>> print(f"Best Model: {results['best_model_name']}")
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
__license__ = "MIT"

# Import main pipeline function
from pitt_pipeline import complete_pjk_pipeline_with_bayes_opt

# Import utility functions (if you split them into separate modules later)
# from .feature_selection import ensemble_feature_selection, auc_optimized_selection
# from .feature_engineering import generate_interaction_terms, filter_features_for_interactions
# from .optimization import optimize_model_hyperparameters, get_optimization_space
# from .ensemble import create_ensemble_models
# from .utils import infer_variable_types, confirm_variable_types

# Define what's available when using "from pjk_pipeline import *"
__all__ = [
    "complete_pjk_pipeline_with_bayes_opt",
    "__version__",
]

# Check for optional dependencies and provide helpful messages
def _check_optional_dependencies():
    """Check availability of optional dependencies and provide info."""
    optional_deps = {
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM", 
        "catboost": "CatBoost",
        "skopt": "Scikit-Optimize (for Bayesian optimization)",
    }
    
    missing = []
    for module, name in optional_deps.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(name)
    
    if missing:
        print(f"ℹ️  Optional dependencies not installed: {', '.join(missing)}")
        print("   Install with: pip install pjk-prediction-pipeline[full]")
        print("   Or individually:")
        print("   - Bayesian optimization: pip install scikit-optimize")
        print("   - Gradient boosting: pip install xgboost lightgbm catboost")

# Run dependency check on import (can be disabled by setting environment variable)
import os
if os.getenv("PJK_PIPELINE_SILENT_IMPORT", "0") != "1":
    _check_optional_dependencies()


# Convenience function for quick model comparison
def quick_compare(df, target_col, base_features, **kwargs):
    """
    Quick model comparison with sensible defaults.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with features and target
    target_col : str
        Name of the target column
    base_features : list
        List of feature column names
    **kwargs : dict
        Additional arguments passed to complete_pjk_pipeline_with_bayes_opt
        
    Returns
    -------
    dict
        Results dictionary with model performance and details
        
    Example
    -------
    >>> results = quick_compare(
    ...     df=data,
    ...     target_col='outcome',
    ...     base_features=feature_list,
    ...     max_features=10
    ... )
    """
    defaults = {
        'selection_method': 'auc_optimized',
        'use_bayes_opt': False,  # Faster default
        'use_ensemble': False,
        'test_size': 0.2,
        'random_state': 42,
    }
    defaults.update(kwargs)
    
    return complete_pjk_pipeline_with_bayes_opt(
        df=df,
        target_col=target_col,
        base_features=base_features,
        **defaults
    )


# Add to __all__
__all__.append("quick_compare")


# Package metadata
PACKAGE_INFO = {
    "name": "pjk-prediction-pipeline",
    "version": __version__,
    "description": "Publication-ready PJK prediction pipeline with Bayesian optimization and ensemble methods",
    "author": __author__,
    "license": __license__,
    "url": "https://github.com/yourusername/pjk-prediction-pipeline",
    "documentation": "https://pjk-prediction-pipeline.readthedocs.io",
}


def get_package_info():
    """Return package metadata as a dictionary."""
    return PACKAGE_INFO.copy()


def print_citation():
    """Print citation information for the package."""
    citation = """
    If you use this package in your research, please cite:
    
    @software{pjk_pipeline,
      title = {PJK Prediction Pipeline: Bayesian Optimization and Ensemble Methods},
      author = {Your Name},
      year = {2024},
      version = {""" + __version__ + """},
      url = {https://github.com/yourusername/pjk-prediction-pipeline}
    }
    """
    print(citation)


__all__.extend(["get_package_info", "print_citation"])
