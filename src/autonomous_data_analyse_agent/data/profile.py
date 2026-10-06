import pandas as pd
import json
import numpy as np

def generate_profile(df: pd.DataFrame) -> str:
    """
    Generates an LLM-friendly statistical profile of a dataframe.
    Never includes raw rows.
    """
    profile = {
        "dataset_shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": {}
    }
    
    for col in df.columns:
        col_type = str(df[col].dtype)
        col_stats = {
            "type": col_type,
            "missing_values": int(df[col].isnull().sum()),
            "missing_percentage": round(float(df[col].isnull().sum() / len(df)) * 100, 2)
        }
        
        # Numeric statistics
        if pd.api.types.is_numeric_dtype(df[col]):
            desc = df[col].describe()
            col_stats.update({
                "mean": round(float(desc.get("mean", 0)), 2),
                "std": round(float(desc.get("std", 0)), 2),
                "min": round(float(desc.get("min", 0)), 2),
                "25%": round(float(desc.get("25%", 0)), 2),
                "50%": round(float(desc.get("50%", 0)), 2),
                "75%": round(float(desc.get("75%", 0)), 2),
                "max": round(float(desc.get("max", 0)), 2),
                "unique_values_count": int(df[col].nunique())
            })
            
            # Simple distribution using histogram to pass to LLM
            # Catching cases where max == min (no variance)
            if col_stats["max"] > col_stats["min"]:
                hist, bin_edges = np.histogram(df[col].dropna(), bins=5)
                col_stats["distribution_bins"] = [round(b, 2) for b in bin_edges.tolist()]
                col_stats["distribution_counts"] = hist.tolist()
            else:
                col_stats["distribution_bins"] = []
                col_stats["distribution_counts"] = []
                
        # Categorical statistics
        elif pd.api.types.is_object_dtype(df[col]):
            col_stats.update({
                "unique_values_count": int(df[col].nunique()),
            })
            # Only include value counts if cardinality is low to avoid blowing up the context window
            if col_stats["unique_values_count"] <= 20:
                counts = df[col].value_counts().head(20).to_dict()
                col_stats["top_values"] = {str(k): int(v) for k, v in counts.items()}
            else:
                col_stats["top_values"] = "Too many unique values to display"
                
        profile["columns"][col] = col_stats
        
    return json.dumps(profile, indent=2)
