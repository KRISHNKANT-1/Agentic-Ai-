import pandas as pd
from typing import Dict, Any
import sqlalchemy
import os

MAX_ROWS = 50000

def ingest_data(source: str, encoding: str = 'utf-8', delimiter: str = ',') -> Dict[str, Any]:
    """
    Universal ingestion with memory safety limits.
    Handles CSV, Parquet, JSON, Excel, and SQL database connection strings.
    """
    df = None
    total_rows = 0
    
    try:
        # --- 1. Handle SQL Databases ---
        # Format expectation: "sqlite:///database.db::SELECT * FROM my_table"
        if source.startswith(("postgresql://", "mysql://", "sqlite://")):
            if "::" in source:
                conn_str, query = source.split("::", 1)
                engine = sqlalchemy.create_engine(conn_str)
                # Enforce safety limit on the SQL query
                safe_query = f"{query} LIMIT {MAX_ROWS}"
                df = pd.read_sql(safe_query, engine)
                total_rows = len(df) # We only know what we pulled unless we run a COUNT()
            else:
                raise ValueError("For DB sources, append '::SELECT * FROM table' to the connection string.")
                
        # --- 2. Handle Files (Local or URL) ---
        else:
            # Check extension to determine parsing engine
            ext = os.path.splitext(source)[1].lower()
            
            if ext == '.parquet':
                df = pd.read_parquet(source)
                total_rows = len(df)
                if total_rows > MAX_ROWS:
                    df = df.head(MAX_ROWS)
                    
            elif ext in ['.xlsx', '.xls']:
                df = pd.read_excel(source, nrows=MAX_ROWS)
                total_rows = "Unknown (Excel)" # Loading full excel to count is memory-heavy
                
            elif ext == '.json':
                df = pd.read_json(source)
                total_rows = len(df)
                if total_rows > MAX_ROWS:
                    df = df.head(MAX_ROWS)
                    
            else:
                # Default to CSV / TXT / TSV
                # Fast row count for text files
                if os.path.exists(source):
                    with open(source, 'r', encoding=encoding, errors='ignore') as f:
                        total_rows = sum(1 for _ in f)
                    if total_rows > 0:
                        total_rows -= 1 # adjust for header
                else:
                    total_rows = "Unknown (URL)"
                    
                df = pd.read_csv(source, encoding=encoding, sep=delimiter, nrows=MAX_ROWS, on_bad_lines='skip')
                
    except Exception as e:
        raise RuntimeError(f"Ingestion Error: Failed to ingest data from {source}. Details: {str(e)}")

    return {
        'dataframe': df,
        'total_rows_in_file': total_rows,
        'rows_ingested': len(df) if df is not None else 0
    }
