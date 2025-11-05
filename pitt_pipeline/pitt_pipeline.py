def complete_pjk_pipeline_with_bayes_opt(df, target_col, base_features, max_features=None, epv=20,  
                                         min_target_corr=0.1, max_inter_corr=0.8, max_interaction_features=20,
                                         selection_method='hybrid', force_include=None, use_ensemble=False,
                                         test_size=0.2, random_state=42, stratify=True,
                                         # NEW Bayesian Optimization parameters
                                         use_bayes_opt=True, n_bayes_calls=50, cv_folds=5,
                                         optimize_models=['RandomForest', 'GradientBoosting', 'XGBoost'],
                                         # NEW Ensemble parameters
                                         ensemble_top_n=5):
    """
    Complete PJK prediction pipeline with Bayesian Optimization AND Ensemble methods.
    
    Key Changes for Publication:
    - Splits data into train/test BEFORE any feature selection or model training
    - All feature selection happens only on training data
    - Bayesian optimization for hyperparameter tuning on selected models
    - Ensemble methods for improved performance
    - Final model performance reported on held-out test set
    
    New Parameters:
    - use_bayes_opt: Boolean, whether to apply Bayesian optimization
    - n_bayes_calls: Number of optimization iterations (default=50)
    - cv_folds: Number of cross-validation folds for optimization (default=5)
    - optimize_models: List of model names to optimize (others use default params)
    - use_ensemble: Boolean, whether to create ensemble models
    - ensemble_top_n: Number of top models to include in ensembles (default=5)
    """
    
    import pandas as pd
    import numpy as np
    from itertools import combinations
    from joblib import Parallel, delayed
    from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
    from sklearn.base import clone
    from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                                ExtraTreesClassifier, AdaBoostClassifier, 
                                VotingClassifier, StackingClassifier)
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression, LassoCV
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    
    try:
        from xgboost import XGBClassifier
    except ImportError:
        XGBClassifier = None
        print("XGBoost not available - install with: pip install xgboost")
    
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        LGBMClassifier = None
        print("LightGBM not available - install with: pip install lightgbm")
    
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        CatBoostClassifier = None
        print("CatBoost not available - install with: pip install catboost")

    # FIXED: Bayesian Optimization imports
    try:
        from skopt import gp_minimize
        from skopt.space import Real, Integer, Categorical
        from skopt.utils import use_named_args
        bayes_opt_available = True
        print("✅ Scikit-optimize available for Bayesian optimization")
    except ImportError:
        bayes_opt_available = False
        if use_bayes_opt:
            print("❌ ERROR: scikit-optimize not available. Install with: pip install scikit-optimize")
            print("Continuing without Bayesian optimization...")
            use_bayes_opt = False

    def infer_variable_types(df, max_cat_unique=5, max_ordinal_unique=10):
        """Automatically infer variable types"""
        continuous_vars = []
        ordinal_vars = []
        categorical_vars = []

        for col in df.columns:
            n_unique = df[col].nunique(dropna=True)
            dtype = df[col].dtypes

            if pd.api.types.is_numeric_dtype(df[col]):
                if n_unique <= max_cat_unique:
                    categorical_vars.append(col)
                elif n_unique <= max_ordinal_unique:
                    ordinal_vars.append(col)
                else:
                    continuous_vars.append(col)
            else:
                categorical_vars.append(col)

        return continuous_vars, ordinal_vars, categorical_vars
    
    def get_new_classification(var, df):
        """Get new classification for a specific variable with detailed info"""
        print(f"\n{'='*50}")
        print(f"Reclassifying: {var}")
        
        if df is not None and var in df.columns:
            values = df[var].dropna()
            if len(values) > 0:
                unique_vals = sorted(values.unique())
                n_unique = len(unique_vals)
                
                print(f"Number of unique values: {n_unique}")
                print(f"Data type: {values.dtype}")
                
                if n_unique <= 10:
                    print(f"All unique values: {unique_vals}")
                else:
                    print(f"Sample values: {unique_vals[:5]} ... {unique_vals[-3:]}")
                    print(f"Range: {values.min():.2f} to {values.max():.2f}")
                    print(f"Mean: {values.mean():.2f}, Std: {values.std():.2f}")
                
                if n_unique <= 15:
                    print("Value counts:")
                    print(values.value_counts().head(10))
        
        while True:
            user_input = input(f"\nNew classification for '{var}' [C=Continuous, O=Ordinal, T=Categorical, S=Skip]: ").strip().upper()
            
            if user_input in ['C', 'O', 'T', 'S']:
                type_names = {'C': 'Continuous', 'O': 'Ordinal', 'T': 'Categorical', 'S': 'Skipped'}
                print(f"✓ {var} reclassified as {type_names[user_input]}")
                return user_input
            else:
                print("Please enter C, O, T, or S")

    def reclassify_specific_variables(continuous, ordinal, categorical, df):
        """Allow user to reclassify specific variables"""
        continuous_new = continuous.copy()
        ordinal_new = ordinal.copy()
        categorical_new = categorical.copy()
        
        all_variables = continuous_new + ordinal_new + categorical_new
        
        print("Available variables to reclassify:")
        for i, var in enumerate(all_variables, 1):
            if var in continuous_new:
                current_type = "Continuous"
            elif var in ordinal_new:
                current_type = "Ordinal"
            else:
                current_type = "Categorical"
            print(f"{i:2d}. {var:<30} (currently: {current_type})")
        
        print("\nEnter variable numbers to reclassify (e.g., '1 3 7' or 'all'), or 'done' to finish:")
        
        while True:
            user_input = input("Variables to reclassify: ").strip()
            
            if user_input.lower() == 'done':
                break
            elif user_input.lower() == 'all':
                vars_to_reclassify = all_variables
            else:
                try:
                    indices = [int(x) for x in user_input.split()]
                    vars_to_reclassify = [all_variables[i-1] for i in indices if 1 <= i <= len(all_variables)]
                    if not vars_to_reclassify:
                        print("No valid variable numbers entered. Try again.")
                        continue
                except (ValueError, IndexError):
                    print("Invalid input. Enter numbers separated by spaces, 'all', or 'done'.")
                    continue
            
            for var in vars_to_reclassify:
                # Remove from current lists
                if var in continuous_new:
                    continuous_new.remove(var)
                elif var in ordinal_new:
                    ordinal_new.remove(var)
                elif var in categorical_new:
                    categorical_new.remove(var)
                
                new_type = get_new_classification(var, df)
                
                # Add to appropriate list
                if new_type == 'C':
                    continuous_new.append(var)
                elif new_type == 'O':
                    ordinal_new.append(var)
                elif new_type == 'T':
                    categorical_new.append(var)
            
            print(f"\n=== Updated Classification ===")
            print(f"Continuous ({len(continuous_new)}):", continuous_new)
            print(f"Ordinal ({len(ordinal_new)}):", ordinal_new)
            print(f"Categorical ({len(categorical_new)}):", categorical_new)
            
            more_changes = input("\nReclassify more variables? (y/n): ").strip().lower()
            if more_changes not in ['y', 'yes']:
                break
        
        return continuous_new, ordinal_new, categorical_new

    def confirm_variable_types(continuous, ordinal, categorical, df=None):
        """Enhanced confirmation with selective reclassification option"""
        print("\n=== Variable Type Summary ===")
        print(f"Continuous ({len(continuous)}):", continuous)
        print(f"Ordinal ({len(ordinal)}):", ordinal)
        print(f"Categorical ({len(categorical)}):", categorical)

        response = input("\nDo the variable types look correct? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            return continuous, ordinal, categorical
        else:
            print("\n=== Selective Reclassification ===")
            return reclassify_specific_variables(continuous, ordinal, categorical, df)
    
    def filter_features_for_interactions(df, target_col, features, 
                                       min_target_corr=0.1, max_inter_corr=0.8, 
                                       max_features=20):
        """Pre-filter features before generating interactions"""
        print(f"\n=== Pre-filtering {len(features)} features for interaction generation ===")
        
        correlations = df[features + [target_col]].corr()
        target_corrs = correlations[target_col].abs()
        
        candidate_features = target_corrs[target_corrs >= min_target_corr].index.tolist()
        if target_col in candidate_features:
            candidate_features.remove(target_col)
        
        print(f"Features with target correlation >= {min_target_corr}: {len(candidate_features)}")
        
        if len(candidate_features) > max_features:
            candidate_features = target_corrs[candidate_features].nlargest(max_features).index.tolist()
            print(f"Keeping top {max_features} features by target correlation")
        
        if len(candidate_features) > 1:
            feature_corr_matrix = correlations.loc[candidate_features, candidate_features]
            
            high_corr_pairs = []
            for i in range(len(candidate_features)):
                for j in range(i+1, len(candidate_features)):
                    feat1, feat2 = candidate_features[i], candidate_features[j]
                    if abs(feature_corr_matrix.loc[feat1, feat2]) > max_inter_corr:
                        if target_corrs[feat1] >= target_corrs[feat2]:
                            high_corr_pairs.append(feat2)
                        else:
                            high_corr_pairs.append(feat1)
            
            filtered_features = [f for f in candidate_features if f not in high_corr_pairs]
            if len(high_corr_pairs) > 0:
                print(f"Removed {len(high_corr_pairs)} highly correlated features")
        else:
            filtered_features = candidate_features
        
        print(f"Final features for interaction generation: {len(filtered_features)}")
        print(f"This will generate ~{len(filtered_features) * (len(filtered_features) - 1)} interaction terms")
        
        return filtered_features

    def generate_interaction_terms(df, features):
        """Generate interaction terms efficiently"""
        interaction_dict = {}

        for f1, f2 in combinations(features, 2):
            # Multiply interaction
            interaction_dict[f'{f1}_x_{f2}'] = df[f1] * df[f2]

            # Determine if either variable is binary
            is_f1_binary = set(df[f1].dropna().unique()).issubset({0, 1})
            is_f2_binary = set(df[f2].dropna().unique()).issubset({0, 1})

            # Division: Only allow if denominator is not binary
            if not is_f2_binary:
                mask_f2 = (df[f2] != 0) & df[f2].notna() & df[f1].notna()
                div_1 = pd.Series(0, index=df.index, dtype=float)
                div_1[mask_f2] = df.loc[mask_f2, f1] / df.loc[mask_f2, f2]
                interaction_dict[f'{f1}_div_{f2}'] = div_1

            if not is_f1_binary:
                mask_f1 = (df[f1] != 0) & df[f1].notna() & df[f2].notna()
                div_2 = pd.Series(0, index=df.index, dtype=float)
                div_2[mask_f1] = df.loc[mask_f1, f2] / df.loc[mask_f1, f1]
                interaction_dict[f'{f2}_div_{f1}'] = div_2

        interaction_df = pd.DataFrame(interaction_dict, index=df.index)
        return interaction_df

    def get_optimization_space(model_name):
        """Define hyperparameter search spaces for different models"""
        spaces = {
            'RandomForest': [
                Integer(50, 500, name='n_estimators'),
                Integer(1, 30, name='max_depth'),
                Integer(2, 20, name='min_samples_split'),
                Integer(1, 10, name='min_samples_leaf'),
                Real(0.5, 1.0, name='max_features'),
                Categorical(['gini', 'entropy'], name='criterion')
            ],
            'GradientBoosting': [
                Integer(50, 300, name='n_estimators'),
                Real(0.01, 0.3, name='learning_rate'),
                Integer(1, 15, name='max_depth'),
                Integer(2, 20, name='min_samples_split'),
                Integer(1, 10, name='min_samples_leaf'),
                Real(0.5, 1.0, name='subsample')
            ],
            'XGBoost': [
                Integer(50, 300, name='n_estimators'),
                Real(0.01, 0.3, name='learning_rate'),
                Integer(3, 10, name='max_depth'),
                Real(0.5, 1.0, name='subsample'),
                Real(0.5, 1.0, name='colsample_bytree'),
                Real(0, 10, name='reg_alpha'),
                Real(0, 10, name='reg_lambda')
            ],
            'LightGBM': [
                Integer(50, 300, name='n_estimators'),
                Real(0.01, 0.3, name='learning_rate'),
                Integer(3, 15, name='max_depth'),
                Integer(10, 300, name='num_leaves'),
                Real(0.5, 1.0, name='subsample'),
                Real(0.5, 1.0, name='colsample_bytree')
            ],
            'ExtraTrees': [
                Integer(50, 500, name='n_estimators'),
                Integer(1, 30, name='max_depth'),
                Integer(2, 20, name='min_samples_split'),
                Integer(1, 10, name='min_samples_leaf'),
                Real(0.5, 1.0, name='max_features')
            ],
            'LogisticRegression': [
                Real(0.0001, 100, prior='log-uniform', name='C'),
                Categorical(['l1', 'l2', 'elasticnet'], name='penalty'),
                Real(0.1, 0.9, name='l1_ratio')  # Only used if penalty='elasticnet'
            ],
            'SVM': [
                Real(0.1, 100, prior='log-uniform', name='C'),
                Real(0.0001, 1, prior='log-uniform', name='gamma'),
                Categorical(['rbf', 'poly', 'sigmoid'], name='kernel')
            ]
        }
        return spaces.get(model_name, [])

    def optimize_model_hyperparameters(model_name, model_class, X_train, y_train, 
                                     n_calls=50, cv_folds=5, random_state=42):
        """Optimize hyperparameters using Bayesian optimization"""
        print(f"\n=== BAYESIAN OPTIMIZATION: {model_name} ===")
        
        space = get_optimization_space(model_name)
        if not space:
            print(f"No optimization space defined for {model_name}, using default parameters")
            return model_class(random_state=random_state), None, None
        
        cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        @use_named_args(space)
        def objective(**params):
            # Handle special cases for different models
            if model_name == 'LogisticRegression':
                if params['penalty'] != 'elasticnet':
                    params.pop('l1_ratio', None)
                if params['penalty'] == 'l1':
                    # l1 penalty requires liblinear or saga solver
                    params['solver'] = 'liblinear'
                model = model_class(random_state=random_state, max_iter=1000, **params)
            elif model_name == 'XGBoost':
                model = model_class(random_state=random_state, eval_metric='logloss', **params)
            elif model_name == 'LightGBM':
                model = model_class(random_state=random_state, verbose=-1, **params)
            elif model_name == 'SVM':
                model = model_class(probability=True, random_state=random_state, **params)
            else:
                model = model_class(random_state=random_state, **params)
            
            try:
                scores = cross_val_score(model, X_train, y_train, scoring='roc_auc', cv=cv_strategy)
                return -scores.mean()  # Negative because gp_minimize minimizes
            except Exception as e:
                print(f"Error with params {params}: {str(e)}")
                return 0  # Return worst score for invalid parameters
        
        print(f"Starting optimization with {n_calls} iterations...")
        result = gp_minimize(objective, space, n_calls=n_calls, random_state=random_state)
        
        # Get best parameters
        best_params = dict(zip([dim.name for dim in space], result.x))
        best_score = -result.fun
        
        print(f"Best CV AUC: {best_score:.4f}")
        print(f"Best parameters: {best_params}")
        
        # Create optimized model
        if model_name == 'LogisticRegression':
            if best_params['penalty'] != 'elasticnet':
                best_params.pop('l1_ratio', None)
            if best_params['penalty'] == 'l1':
                best_params['solver'] = 'liblinear'
            optimized_model = model_class(random_state=random_state, max_iter=1000, **best_params)
        elif model_name == 'XGBoost':
            optimized_model = model_class(random_state=random_state, eval_metric='logloss', **best_params)
        elif model_name == 'LightGBM':
            optimized_model = model_class(random_state=random_state, verbose=-1, **best_params)
        elif model_name == 'SVM':
            optimized_model = model_class(probability=True, random_state=random_state, **best_params)
        else:
            optimized_model = model_class(random_state=random_state, **best_params)
        
        return optimized_model, best_params, best_score

    def create_ensemble_models(X_train, y_train, selected_features, top_models, 
                              optimized_models, use_bayes_opt=True, cv_folds=5):
        """
        Create ensemble models from the top performing individual models
        """
        print(f"\n=== CREATING ENSEMBLE MODELS ===")
        print(f"Building ensembles from top {len(top_models)} models")
        
        cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Prepare models for ensemble
        ensemble_models = []
        for model_name, cv_auc, std_auc, model, was_optimized in top_models:
            if was_optimized and model_name in optimized_models:
                # Use optimized model
                optimized_model, best_params = optimized_models[model_name]
                ensemble_models.append((model_name, clone(optimized_model)))
            else:
                # Use default model
                ensemble_models.append((model_name, clone(model)))
            print(f"  Added {model_name} ({'optimized' if was_optimized else 'default'})")
        
        ensemble_results = []
        
        # 1. VOTING CLASSIFIER (Soft Voting)
        print(f"\n--- Testing Voting Classifier (Soft) ---")
        try:
            voting_clf = VotingClassifier(
                estimators=ensemble_models,
                voting='soft'
            )
            
            voting_scores = cross_val_score(
                voting_clf, X_train[selected_features], y_train,
                scoring='roc_auc', cv=cv_strategy
            )
            voting_auc = voting_scores.mean()
            voting_std = voting_scores.std()
            
            print(f"Voting Classifier: AUC = {voting_auc:.3f} ± {voting_std:.3f}")
            ensemble_results.append(('VotingClassifier', voting_auc, voting_std, voting_clf, False))
            
        except Exception as e:
            print(f"Voting Classifier failed: {str(e)}")
        
        # 2. STACKING CLASSIFIER with Logistic Regression
        print(f"\n--- Testing Stacking Classifier (LogReg Meta) ---")
        try:
            stacking_clf = StackingClassifier(
                estimators=ensemble_models,
                final_estimator=LogisticRegression(max_iter=1000, random_state=42),
                cv=cv_folds,
                passthrough=True  # Include original features in meta-learner
            )
            
            stacking_scores = cross_val_score(
                stacking_clf, X_train[selected_features], y_train,
                scoring='roc_auc', cv=cv_strategy
            )
            stacking_auc = stacking_scores.mean()
            stacking_std = stacking_scores.std()
            
            print(f"Stacking Classifier: AUC = {stacking_auc:.3f} ± {stacking_std:.3f}")
            ensemble_results.append(('StackingClassifier', stacking_auc, stacking_std, stacking_clf, False))
            
        except Exception as e:
            print(f"Stacking Classifier failed: {str(e)}")
        
        # 3. WEIGHTED ENSEMBLE (Custom weights based on CV performance)
        print(f"\n--- Testing Weighted Ensemble ---")
        try:
            # Calculate weights based on CV AUC scores
            model_weights = []
            for model_name, cv_auc, std_auc, model, was_optimized in top_models:
                model_weights.append(cv_auc)
            
            # Normalize weights
            total_weight = sum(model_weights)
            normalized_weights = [w/total_weight for w in model_weights]
            
            print(f"Model weights: {dict(zip([m[0] for m in top_models], normalized_weights))}")
            
            # Create weighted voting classifier
            weighted_voting_clf = VotingClassifier(
                estimators=ensemble_models,
                voting='soft',
                weights=normalized_weights
            )
            
            weighted_scores = cross_val_score(
                weighted_voting_clf, X_train[selected_features], y_train,
                scoring='roc_auc', cv=cv_strategy
            )
            weighted_auc = weighted_scores.mean()
            weighted_std = weighted_scores.std()
            
            print(f"Weighted Ensemble: AUC = {weighted_auc:.3f} ± {weighted_std:.3f}")
            ensemble_results.append(('WeightedEnsemble', weighted_auc, weighted_std, weighted_voting_clf, False))
            
        except Exception as e:
            print(f"Weighted Ensemble failed: {str(e)}")
        
        return ensemble_results

    def auc_optimized_selection(X, y, max_features):
        """Comprehensive AUC-optimized feature selection"""
        from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.naive_bayes import GaussianNB
        
        print("Using comprehensive AUC optimization...")
        
        # Step 1: Multiple selection methods
        methods = {}
        
        # Univariate selection
        selector_f = SelectKBest(score_func=f_classif, k=max_features)
        selector_f.fit(X, y)
        methods['f_score'] = X.columns[selector_f.get_support()].tolist()
        
        # Mutual information
        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_series = pd.Series(mi_scores, index=X.columns)
        methods['mutual_info'] = mi_series.nlargest(max_features).index.tolist()
        
        # Model-based (multiple models)
        for name, model in [('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
                           ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
                           ('nb', GaussianNB())]:
            try:
                model.fit(X, y)
                if hasattr(model, 'feature_importances_'):
                    importances = pd.Series(model.feature_importances_, index=X.columns)
                    methods[name] = importances.nlargest(max_features).index.tolist()
            except:
                continue
        
        # Step 2: Ensemble voting
        feature_votes = {}
        for method_name, features in methods.items():
            for i, feature in enumerate(features):
                if feature not in feature_votes:
                    feature_votes[feature] = 0
                # Weight by rank (higher rank = more votes)
                feature_votes[feature] += (max_features - i)
        
        # Step 3: Select top voted features
        sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
        candidate_features = [f[0] for f in sorted_features[:max_features * 2]]  # Get double for testing
        
        # Step 4: Optimize final set with forward selection
        print(f"Testing combinations from {len(candidate_features)} candidate features...")
        
        selected_features = []
        remaining_candidates = candidate_features.copy()
        
        # Use the best model type for final optimization
        test_model = GaussianNB()  # Since this has been your best performer
        
        for step in range(max_features):
            best_feature = None
            best_auc = 0
            
            if len(selected_features) == 0:
                # First feature - test each individually
                for feature in remaining_candidates[:20]:  # Test top 20 for first feature
                    try:
                        auc = cross_val_score(test_model, X[[feature]], y, scoring='roc_auc', cv=3).mean()
                        if auc > best_auc:
                            best_auc = auc
                            best_feature = feature
                    except:
                        continue
            else:
                # Subsequent features - test additions
                for feature in remaining_candidates[:15]:  # Test top 15 additions
                    trial_features = selected_features + [feature]
                    try:
                        auc = cross_val_score(test_model, X[trial_features], y, scoring='roc_auc', cv=3).mean()
                        if auc > best_auc:
                            best_auc = auc
                            best_feature = feature
                    except:
                        continue
            
            if best_feature is not None:
                selected_features.append(best_feature)
                remaining_candidates.remove(best_feature)
                print(f"Selected feature {len(selected_features)}: {best_feature} (AUC: {best_auc:.4f})")
            else:
                break
        
        return selected_features

    def ensemble_feature_selection(X, y, max_features=None, epv=10, method='hybrid'):
        """Perform feature selection using ensemble approach"""
        
        # Determine target number of features
        if max_features is not None:
            target_features = max_features
            print(f"Target: {target_features} features (user specified)")
        else:
            n_events = y.sum()
            target_features = int(max(1, n_events // epv))
            print(f"Target: {target_features} features based on EPV={epv} and {int(n_events)} events")
        
        # Handle missing values
        print("Handling missing values with median imputation...")
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
        
        if method == 'auc_optimized':
            print("Using AUC-optimized selection (best for maximizing AUC)...")
            selected_features = auc_optimized_selection(X_imputed, y, target_features)
        else:
            # Default to simple selection for other methods
            print(f"Using {method} selection...")
            selector = SelectKBest(score_func=f_classif, k=target_features)
            selector.fit(X_imputed, y)
            selected_features = X.columns[selector.get_support()].tolist()
        
        return selected_features, X_imputed

    def get_feature_importance(model, feature_names):
        """Extract feature importance from fitted model"""
        if hasattr(model, 'feature_importances_'):
            return dict(zip(feature_names, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            return dict(zip(feature_names, np.abs(model.coef_[0])))
        return None

    def select_best_model_and_features_with_bayes_opt(X_train, y_train, max_features=None, epv=10, 
                                                    selection_method='hybrid', force_include=None, 
                                                    use_ensemble=False, use_bayes_opt=True,
                                                    n_bayes_calls=50, cv_folds=5,
                                                    optimize_models=['RandomForest', 'GradientBoosting', 'XGBoost'],
                                                    ensemble_top_n=5):
        """Enhanced model selection with Bayesian optimization AND ensemble methods"""
        
        # Define all available models
        all_models = {
            'RandomForest': RandomForestClassifier,
            'GradientBoosting': GradientBoostingClassifier,
            'ExtraTrees': ExtraTreesClassifier,
            'AdaBoost': AdaBoostClassifier,
            'LogisticRegression': LogisticRegression,
            'SVM': SVC,
            'DecisionTree': DecisionTreeClassifier,
            'NaiveBayes': GaussianNB,
            'KNN': KNeighborsClassifier
        }
        
        if XGBClassifier is not None:
            all_models['XGBoost'] = XGBClassifier
        if LGBMClassifier is not None:
            all_models['LightGBM'] = LGBMClassifier
        if CatBoostClassifier is not None:
            all_models['CatBoost'] = CatBoostClassifier
        
        print(f"\n=== TRAINING SET: Feature Selection + Model Comparison (method: {selection_method}) ===")
        print(f"Training set size: {len(X_train)} samples, {len(X_train.columns)} features")
        print(f"Bayesian optimization: {'ON' if use_bayes_opt else 'OFF'}")
        print(f"Ensemble methods: {'ON' if use_ensemble else 'OFF'}")
        if use_bayes_opt:
            print(f"Models to optimize: {optimize_models}")
        
        # Feature selection on training data only
        selected_features, X_train_processed = ensemble_feature_selection(
            X_train, y_train, max_features=max_features, epv=epv, method=selection_method
        )
        
        # Handle force_include parameter
        if force_include:
            force_include_available = [f for f in force_include if f in X_train.columns]
            if force_include_available:
                for feature in force_include_available:
                    if feature not in selected_features:
                        selected_features.append(feature)
                print(f"Force included features: {force_include_available}")
        
        print(f"\n=== TRAINING SET: Model Comparison on {len(selected_features)} Selected Features ===")
        
        # Use stratified k-fold for more robust CV estimates
        cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        results = []
        optimized_models = {}
        
        # Test individual models
        for name, model_class in all_models.items():
            try:
                if use_bayes_opt and bayes_opt_available and name in optimize_models:
                    # Use Bayesian optimization for specified models
                    optimized_model, best_params, best_score = optimize_model_hyperparameters(
                        name, model_class, X_train_processed[selected_features], y_train,
                        n_calls=n_bayes_calls, cv_folds=cv_folds
                    )
                    optimized_models[name] = (optimized_model, best_params)
                    results.append((name, best_score, 0, optimized_model, True))
                    print(f"{name:<20}: Optimized AUC = {best_score:.3f}")
                else:
                    # Use default parameters
                    if name == 'LogisticRegression':
                        model = model_class(random_state=42, max_iter=1000)
                    elif name == 'XGBoost' and XGBClassifier is not None:
                        model = model_class(eval_metric='logloss', random_state=42)
                    elif name == 'LightGBM' and LGBMClassifier is not None:
                        model = model_class(random_state=42, verbose=-1)
                    elif name == 'CatBoost' and CatBoostClassifier is not None:
                        model = model_class(random_state=42, verbose=0)
                    elif name == 'SVM':
                        model = model_class(probability=True, random_state=42)
                    else:
                        model = model_class(random_state=42)
                    
                    auc_scores = cross_val_score(
                        model, X_train_processed[selected_features], y_train, 
                        scoring='roc_auc', cv=cv_strategy
                    )
                    mean_auc = auc_scores.mean()
                    std_auc = auc_scores.std()
                    results.append((name, mean_auc, std_auc, model, False))
                    print(f"{name:<20}: AUC = {mean_auc:.3f} ± {std_auc:.3f}")
            except Exception as e:
                print(f"{name:<20}: Failed - {str(e)[:50]}...")
        
        results.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n=== TRAINING SET: Top {min(len(results), ensemble_top_n)} Individual Models ===")
        for i, (name, auc, std, model, optimized) in enumerate(results[:ensemble_top_n]):
            opt_marker = " (OPTIMIZED)" if optimized else ""
            if optimized:
                print(f"{i+1}. {name:<20}: AUC = {auc:.3f}{opt_marker}")
            else:
                print(f"{i+1}. {name:<20}: AUC = {auc:.3f} ± {std:.3f}{opt_marker}")
        
        # CREATE ENSEMBLE MODELS if requested
        ensemble_results = []
        ensemble_info = {'ensemble_used': use_ensemble, 'ensemble_results': None, 'is_ensemble': False}
        
        if use_ensemble and len(results) >= 2:
            top_models = results[:min(ensemble_top_n, len(results))]
            ensemble_results = create_ensemble_models(
                X_train_processed, y_train, selected_features, 
                top_models, optimized_models, use_bayes_opt, cv_folds
            )
            
            # Add ensemble results to main results
            results.extend(ensemble_results)
            results.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\n=== FINAL RANKING: Individual + Ensemble Models ===")
            for i, (name, auc, std, model, optimized) in enumerate(results[:10]):  # Show top 10
                opt_marker = " (OPTIMIZED)" if optimized else ""
                ensemble_marker = " [ENSEMBLE]" if name.startswith(('Voting', 'Stacking', 'Weighted')) else ""
                if optimized:
                    print(f"{i+1}. {name:<25}: AUC = {auc:.3f}{opt_marker}{ensemble_marker}")
                else:
                    print(f"{i+1}. {name:<25}: AUC = {auc:.3f} ± {std:.3f}{opt_marker}{ensemble_marker}")
            
            ensemble_info['ensemble_results'] = ensemble_results
        
        # Select best overall model (could be individual or ensemble)
        best_model_name, best_cv_auc, best_std, best_model, was_optimized = results[0]
        
        # Determine optimization info
        optimization_info = None
        if was_optimized and best_model_name in optimized_models:
            optimization_info = optimized_models[best_model_name][1]  # best_params
        
        # Update ensemble info
        ensemble_info['is_ensemble'] = best_model_name.startswith(('Voting', 'Stacking', 'Weighted'))
        
        return (best_model_name, selected_features, best_cv_auc, best_model, 
                X_train_processed, optimization_info, ensemble_info)

    # === MAIN PIPELINE STARTS HERE ===
    print("=== Publication-Ready PJK Prediction Pipeline with Bayesian Optimization + Ensembles ===")
    print("Key improvements: Proper train/test split + hyperparameter optimization + ensemble methods")
    
    # Data preparation
    unique_base_features = list(dict.fromkeys(base_features))
    available_features = [f for f in unique_base_features if f in df.columns]
    print(f"Using {len(available_features)} unique variables for analysis")
    
    y = df[target_col]
    X_base = df[available_features]
    
    # ===== TRAIN/TEST SPLIT =====
    print(f"\n=== TRAIN/TEST SPLIT ===")
    print(f"Original dataset: {len(df)} samples")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    X_train_base, X_test_base, y_train, y_test = train_test_split(
        X_base, y, test_size=test_size, random_state=random_state, 
        stratify=y if stratify else None
    )
    
    print(f"Training set: {len(X_train_base)} samples ({len(X_train_base)/len(df)*100:.1f}%)")
    print(f"Test set: {len(X_test_base)} samples ({len(X_test_base)/len(df)*100:.1f}%)")
    print(f"Training target distribution: {y_train.value_counts().to_dict()}")
    print(f"Test target distribution: {y_test.value_counts().to_dict()}")
    
    # ===== CONTINUE WITH EXISTING PIPELINE =====
    # Variable type inference
    continuous, ordinal, categorical = infer_variable_types(X_train_base)
    continuous, ordinal, categorical = confirm_variable_types(continuous, ordinal, categorical, X_train_base)

    # Feature interactions
    features_to_interact = continuous + ordinal
    print(f"Starting with {len(features_to_interact)} continuous/ordinal features")
    
    if len(features_to_interact) > 0:
        train_with_target = pd.concat([X_train_base, y_train], axis=1)
        
        filtered_features = filter_features_for_interactions(
            train_with_target, target_col, features_to_interact,
            min_target_corr=min_target_corr,
            max_inter_corr=max_inter_corr,
            max_features=max_interaction_features
        )
        
        if len(filtered_features) > 1:
            X_train_interactions = generate_interaction_terms(X_train_base, filtered_features)
            X_train_all = pd.concat([X_train_base, X_train_interactions], axis=1)
            
            # Apply same interactions to test data
            X_test_interactions = generate_interaction_terms(X_test_base, filtered_features)
            X_test_all = pd.concat([X_test_base, X_test_interactions], axis=1)
            
            print(f"Generated {len(X_train_interactions.columns)} interaction terms")
        else:
            print("Not enough features passed filtering for interaction generation")
            X_train_all = X_train_base
            X_test_all = X_test_base
    else:
        print("No continuous/ordinal features found for interaction generation")
        X_train_all = X_train_base
        X_test_all = X_test_base
    
    # Handle infinite values
    X_train_all.replace([np.inf, -np.inf], 0, inplace=True)
    X_test_all.replace([np.inf, -np.inf], 0, inplace=True)

    # Feature selection and model training with Bayesian optimization AND ensembles
    if max_features is not None:
        print(f"\nTargeting {max_features} features (user specified)")
    else:
        n_events = np.sum(y_train == 1)
        target_features = max(5, int(n_events // epv))
        max_features = target_features
        print(f"\nTargeting {target_features} features based on EPV={epv} and {n_events} events in training set")

    (best_model_name, selected_features, cv_auc, fitted_model, 
     X_train_processed, optimization_info, ensemble_info) = select_best_model_and_features_with_bayes_opt(
        X_train_all, y_train, max_features=max_features, epv=epv, 
        selection_method=selection_method, force_include=force_include, 
        use_ensemble=use_ensemble, use_bayes_opt=use_bayes_opt, 
        n_bayes_calls=n_bayes_calls, cv_folds=cv_folds,
        optimize_models=optimize_models, ensemble_top_n=ensemble_top_n
    )
    
    # Prepare test data with same preprocessing as training data
    print(f"\n=== APPLYING TRAINING PREPROCESSING TO TEST SET ===")
    
    # Debug: Print column info
    print(f"Training columns: {len(X_train_all.columns)}")
    print(f"Test columns before alignment: {len(X_test_all.columns)}")
    
    # Ensure test set has exactly the same columns as training set
    # Add any missing columns with zeros
    missing_cols = set(X_train_all.columns) - set(X_test_all.columns)
    if missing_cols:
        print(f"Adding {len(missing_cols)} missing columns to test set: {list(missing_cols)}")
        for col in missing_cols:
            X_test_all[col] = 0
    
    # Remove any extra columns from test set
    extra_cols = set(X_test_all.columns) - set(X_train_all.columns)
    if extra_cols:
        print(f"Removing {len(extra_cols)} extra columns from test set: {list(extra_cols)}")
        X_test_all = X_test_all.drop(columns=list(extra_cols))
    
    # Reorder test columns to match training exactly
    X_test_all_ordered = X_test_all.reindex(columns=X_train_all.columns, fill_value=0)
    
    print(f"Test columns after alignment: {len(X_test_all_ordered.columns)}")
    print(f"Column alignment check: {list(X_train_all.columns) == list(X_test_all_ordered.columns)}")
    
    # Apply same imputation to test set
    imputer = SimpleImputer(strategy='median')
    imputer.fit(X_train_all)  # Fit on training data
    
    try:
        X_test_processed = pd.DataFrame(
            imputer.transform(X_test_all_ordered), 
            columns=X_train_all.columns, 
            index=X_test_all_ordered.index
        )
        print("✅ Test preprocessing successful")
    except Exception as e:
        print(f"❌ Error in test preprocessing: {e}")
        # Fallback: Create test set with same structure as training
        print("Using fallback approach...")
        X_test_processed = pd.DataFrame(
            np.zeros((len(X_test_all_ordered), len(X_train_all.columns))),
            columns=X_train_all.columns,
            index=X_test_all_ordered.index
        )
        # Fill with actual values where available
        for col in X_train_all.columns:
            if col in X_test_all_ordered.columns:
                X_test_processed[col] = X_test_all_ordered[col].fillna(X_test_all_ordered[col].median())
        print("✅ Fallback preprocessing successful")
    
    # Final model training and evaluation
    print(f"\n=== FINAL MODEL TRAINING AND EVALUATION ===")
    
    # Train final model on full training set
    fitted_model.fit(X_train_processed[selected_features], y_train)
    
    # Get final performance on test set
    y_test_pred_proba = fitted_model.predict_proba(X_test_processed[selected_features])[:, 1]
    test_auc = roc_auc_score(y_test, y_test_pred_proba)
    
    # Get feature importance
    feature_importance = get_feature_importance(fitted_model, selected_features)
    
    # Get model parameters
    if hasattr(fitted_model, 'get_params'):
        model_params = fitted_model.get_params()
    else:
        model_params = None

    print("\n" + "="*80)
    print("=== FINAL PUBLICATION RESULTS ===")
    print(f"Best Model: {best_model_name}")
    print(f"Model Type: {'Ensemble' if ensemble_info['is_ensemble'] else 'Individual'}")
    print(f"Cross-validation AUC (training): {cv_auc:.4f}")
    print(f"Test Set AUC (final): {test_auc:.4f}")
    print(f"Number of features: {len(selected_features)}")
    print(f"Selected features: {selected_features}")
    print(f"Bayesian optimization used: {use_bayes_opt and bayes_opt_available}")
    if optimization_info:
        print(f"Optimized parameters: {optimization_info}")
    
    if feature_importance:
        print(f"\nFeature Importance (Top 10):")
        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for feat, imp in sorted_importance[:10]:
            print(f"  {feat}: {imp:.4f}")
    
    # Ensemble summary
    if ensemble_info['ensemble_used']:
        print(f"\nEnsemble Summary:")
        print(f"  Ensemble methods tested: {len(ensemble_info['ensemble_results']) if ensemble_info['ensemble_results'] else 0}")
        print(f"  Best model is ensemble: {ensemble_info['is_ensemble']}")
        if ensemble_info['ensemble_results']:
            print(f"  Best ensemble performance:")
            ensemble_results = ensemble_info['ensemble_results']
            ensemble_results.sort(key=lambda x: x[1], reverse=True)
            best_ensemble = ensemble_results[0]
            print(f"    {best_ensemble[0]}: {best_ensemble[1]:.4f} ± {best_ensemble[2]:.4f}")
    
    print("="*80)
    
    # Return comprehensive results for further analysis
    results = {
        'X_train': X_train_processed[selected_features],
        'X_test': X_test_processed[selected_features],
        'y_train': y_train,
        'y_test': y_test,
        'selected_features': selected_features,
        'best_model': fitted_model,
        'best_model_name': best_model_name,
        'cv_auc_train': cv_auc,
        'test_auc': test_auc,
        'y_test_pred_proba': y_test_pred_proba,
        'feature_importance': feature_importance,
        'model_params': model_params,
        'training_size': len(y_train),
        'test_size': len(y_test),
        'train_events': int(np.sum(y_train == 1)),
        'test_events': int(np.sum(y_test == 1)),
        # Bayesian optimization specific results
        'bayes_opt_used': use_bayes_opt and bayes_opt_available,
        'optimized_params': optimization_info,
        'n_bayes_calls': n_bayes_calls if use_bayes_opt else None,
        'optimized_models': optimize_models if use_bayes_opt else None,
        # Ensemble specific results
        'ensemble_info': ensemble_info,
        'ensemble_used': use_ensemble,
        'is_best_ensemble': ensemble_info['is_ensemble']
    }
    
    return results


# USAGE EXAMPLES:

# Example 1: Basic usage with both Bayesian optimization and ensembles
"""
results = complete_pjk_pipeline_with_bayes_opt(
    model_frame, 'pjf diag', predictors,
    max_features=12,
    selection_method='auc_optimized',
    use_ensemble=True,  # Enable ensemble methods
    ensemble_top_n=5,   # Use top 5 models for ensemble
    use_bayes_opt=True, # Enable Bayesian optimization
    n_bayes_calls=50,
    optimize_models=['RandomForest', 'GradientBoosting', 'XGBoost'],
    test_size=0.2,
    random_state=42
)

print(f"Best model: {results['best_model_name']}")
print(f"Is ensemble: {results['is_best_ensemble']}")
print(f"Test AUC: {results['test_auc']:.4f}")
"""

# Example 2: Compare different strategies
"""
strategies = [
    {'use_ensemble': False, 'use_bayes_opt': False, 'name': 'Baseline'},
    {'use_ensemble': False, 'use_bayes_opt': True, 'name': 'Bayes Opt Only'},
    {'use_ensemble': True, 'use_bayes_opt': False, 'name': 'Ensemble Only'},
    {'use_ensemble': True, 'use_bayes_opt': True, 'name': 'Both Bayes + Ensemble'},
]

for strategy in strategies:
    result = complete_pjk_pipeline_with_bayes_opt(
        model_frame, 'pjf diag', predictors,
        max_features=12,
        selection_method='auc_optimized',
        **{k: v for k, v in strategy.items() if k != 'name'}
    )
    print(f"{strategy['name']}: Test AUC = {result['test_auc']:.4f}")
"""