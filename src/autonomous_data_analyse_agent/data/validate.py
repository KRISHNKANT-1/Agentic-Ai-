import pandas as pd
import numpy as np

class DataQualityValidator:
    @staticmethod
    def audit(df: pd.DataFrame) -> dict:
        profile = {
            "total_rows": len(df),
            "column_count": len(df.columns),
            "schema": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "duplicates": {"count": int(df.duplicated().sum()), "percentage": float(df.duplicated().mean())},
            "missingness": {},
            "zero_variance_cols": [],
            "high_cardinality_nominals": [],
            "numerical_distributions": {},
            "outliers_iqr": {}
        }
        
        for col in df.columns:
            missing_pct = df[col].isnull().mean()
            profile["missingness"][col] = {"rate": float(missing_pct), "flag_high": bool(missing_pct > 0.4)}
            
            if df[col].nunique() <= 1:
                profile["zero_variance_cols"].append(col)
                
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                if df[col].nunique() / len(df) > 0.8:
                    profile["high_cardinality_nominals"].append(col)
                    
            if pd.api.types.is_numeric_dtype(df[col]):
                desc = df[col].describe()
                skew = float(df[col].skew()) if len(df[col].dropna()) > 2 else 0.0
                profile["numerical_distributions"][col] = {
                    "min": float(desc.get("min", 0)),
                    "25%": float(desc.get("25%", 0)),
                    "median": float(desc.get("50%", 0)),
                    "75%": float(desc.get("75%", 0)),
                    "max": float(desc.get("max", 0)),
                    "std": float(desc.get("std", 0)),
                    "skewness": skew
                }
                
                # IQR Outliers
                Q1 = desc.get("25%", 0)
                Q3 = desc.get("75%", 0)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                profile["outliers_iqr"][col] = int(len(outliers))
                
        return profile
