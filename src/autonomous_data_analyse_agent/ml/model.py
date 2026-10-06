import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_validate
import warnings

def run_baseline_model(df: pd.DataFrame, target_col: str, task_type: str = 'classification') -> dict:
    warnings.filterwarnings('ignore')
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
        
    df_clean = df.copy()
    n_rows = len(df_clean)
    cols_to_drop = []
    
    # Target Leakage Prevention: ID/Constant Columns
    for col in df_clean.columns:
        if col == target_col: continue
        n_unique = df_clean[col].nunique()
        if n_unique <= 1:
            cols_to_drop.append(col)
        elif (pd.api.types.is_object_dtype(df_clean[col]) or pd.api.types.is_integer_dtype(df_clean[col])) and n_unique / n_rows > 0.8:
            cols_to_drop.append(col)
            
    df_clean.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    # Target Leakage Prevention: Correlation > 0.98
    num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in num_cols:
        corr = df_clean[num_cols].corr()
        target_corr = corr[target_col].abs()
        high_corr_cols = target_corr[target_corr > 0.98].index.tolist()
        high_corr_cols.remove(target_col)
        cols_to_drop.extend(high_corr_cols)
        df_clean.drop(columns=high_corr_cols, inplace=True, errors='ignore')
        
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    # Strictly Encapsulated Pipeline
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    
    numeric_transformer = SimpleImputer(strategy='median')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        
    if task_type == 'classification':
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        scoring = ['f1_weighted', 'roc_auc'] if len(np.unique(y)) == 2 else ['f1_weighted']
    else:
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        scoring = ['neg_mean_absolute_error', 'r2']
        
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    
    cv_results = cross_validate(pipeline, X, y, cv=5, scoring=scoring, return_estimator=True)
    
    metrics = {}
    for score in scoring:
        # Convert neg_mean_absolute_error to positive MAE for easier reading
        if score == 'neg_mean_absolute_error':
            val = -cv_results[f'test_{score}']
            metrics['mae'] = f"{np.mean(val):.4f} ± {np.std(val):.4f}"
        else:
            val = cv_results[f'test_{score}']
            metrics[score.replace('test_', '')] = f"{np.mean(val):.4f} ± {np.std(val):.4f}"
        
    # Feature Importances via Tree Weights
    best_estimator = cv_results['estimator'][0]
    feat_imp = {}
    try:
        cat_features = best_estimator.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features)
        all_features = list(numeric_features) + list(cat_features)
        importances = best_estimator.named_steps['model'].feature_importances_
        feat_imp = pd.Series(importances, index=all_features).sort_values(ascending=False).head(10).to_dict()
    except Exception as e:
        pass
        
    return {"metrics": metrics, "feature_importances": feat_imp, "dropped_leakage_cols": cols_to_drop}
