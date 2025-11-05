"""
Setup configuration for the PJK Prediction Pipeline package
"""
from setuptools import setup, find_packages

# Read the long description from README if it exists
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "A comprehensive machine learning pipeline for PJK prediction with Bayesian optimization and ensemble methods."

setup(
    name="pitt_pipeline",
    version="1.0.0",
    author="Your Name",  # Update with actual author
    author_email="your.email@example.com",  # Update with actual email
    description="Publication-ready PJK prediction pipeline with Bayesian optimization and ensemble methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Swakelin2011/pitt-pipeline",  # Update with actual URL
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    
    # Core dependencies
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "joblib>=1.0.0",
        "scipy>=1.7.0",
    ],
    
    # Optional dependencies for enhanced functionality
    extras_require={
        "full": [
            "xgboost>=1.5.0",
            "lightgbm>=3.3.0",
            "catboost>=1.0.0",
            "scikit-optimize>=0.9.0",
        ],
        "bayes": [
            "scikit-optimize>=0.9.0",
        ],
        "boosting": [
            "xgboost>=1.5.0",
            "lightgbm>=3.3.0",
            "catboost>=1.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "ipython>=8.0.0",
            "jupyter>=1.0.0",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.18.0",
        ],
    },
    
    # Entry points for command-line scripts (optional)
    entry_points={
        "console_scripts": [
            "pjk-pipeline=pjk_pipeline.cli:main",  # If you add CLI later
        ],
    },
    
    # Include additional files
    include_package_data=True,
    package_data={
        "pjk_pipeline": ["*.txt", "*.md"],
    },
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/yourusername/pjk-prediction-pipeline/issues",
        "Source": "https://github.com/yourusername/pjk-prediction-pipeline",
        "Documentation": "https://pjk-prediction-pipeline.readthedocs.io",
    },
    
    # Keywords for PyPI search
    keywords=[
        "machine-learning",
        "medical-prediction",
        "bayesian-optimization",
        "ensemble-methods",
        "feature-selection",
        "classification",
        "healthcare",
        "pjk",
    ],
    
    # Zip safe
    zip_safe=False,
)
