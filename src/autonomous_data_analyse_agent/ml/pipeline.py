import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, List

def safe_train_test_split(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Safely splits the data into train and test sets BEFORE preprocessing.
    Automatically detects and drops likely identifier columns to prevent target leakage.
    
    Returns: X_train, X_test, y_train, y_test
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
        
    df_clean = df.copy()
    
    # Heuristic 1: Drop high-cardinality ID columns to prevent target leakage
    # (e.g., if a column has >95% unique values and is likely just a row identifier)
    cols_to_drop = []
    n_rows = len(df_clean)
    
    for col in df_clean.columns:
        if col == target_col:
            continue
            
        n_unique = df_clean[col].nunique()
        # If unique values are almost equal to total rows (and we have a decent sample size)
        if n_unique == n_rows or (n_unique > n_rows * 0.95 and n_rows > 100):
            cols_to_drop.append(col)
            
    if cols_to_drop:
        print(f"🔒 [SAFETY PROTOCOL] Automatically dropped potential identifier/leakage columns: {cols_to_drop}")
        df_clean = df_clean.drop(columns=cols_to_drop)
        
    # Perform strict split
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test
