import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from autonomous_data_analyse_agent.data.validate import DataQualityValidator
from autonomous_data_analyse_agent.ml.model import run_baseline_model

def create_synthetic_data():
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'id': range(n), # Leakage: ID
        'constant_col': ['A'] * n, # Leakage: Constant
        'feature1': np.random.normal(0, 1, n),
        'feature2': np.random.choice(['X', 'Y', 'Z'], n),
        'feature3': np.random.uniform(0, 100, n),
        'target': np.random.choice([0, 1], n)
    })
    # Add high correlation leakage
    df['leakage_target_proxy'] = df['target'] + np.random.normal(0, 0.01, n)
    # Add some missing values
    df.loc[10:50, 'feature1'] = np.nan
    return df

def test_pipeline():
    print("--- GENERATING SYNTHETIC DATA ---")
    df = create_synthetic_data()
    print(f"Shape: {df.shape}")
    
    print("\n--- PHASE 2: VALIDATION & PROFILING ---")
    profile = DataQualityValidator.audit(df)
    print("Zero Variance Cols:", profile["zero_variance_cols"])
    print("High Cardinality:", profile["high_cardinality_nominals"])
    print("Missingness Flags:", {k: v for k, v in profile["missingness"].items() if v["flag_high"]})
    
    print("\n--- PHASE 3: LEAKAGE-FREE ML ---")
    ml_report = run_baseline_model(df, target_col='target', task_type='classification')
    print("Dropped Leakage Cols:", ml_report["dropped_leakage_cols"])
    print("Metrics:", ml_report["metrics"])
    print("Top Feature Importances:", ml_report["feature_importances"])
    
    print("\n✅ PIPELINE TEST COMPLETE.")

if __name__ == "__main__":
    test_pipeline()
