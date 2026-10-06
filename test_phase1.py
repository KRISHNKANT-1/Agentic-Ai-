import sys
import os

# Add the src folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from autonomous_data_analyse_agent.data.detect import detect_file_properties
from autonomous_data_analyse_agent.data.ingest import ingest_data
import pandas as pd

file_path = "/Users/krishnkant/Downloads/sample_submission (1).csv"

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    sys.exit(1)

print("--- STEP 1: DETECT ---")
props = detect_file_properties(file_path)
print(f"Detected Encoding: {props['encoding']}")
print(f"Detected Delimiter: {repr(props['delimiter'])}")

print("\n--- STEP 2: INGEST ---")
result = ingest_data(file_path, encoding=props['encoding'], delimiter=props['delimiter'])
df = result['dataframe']

print(f"Total rows in file: {result['total_rows_in_file']}")
print(f"Rows ingested: {result['rows_ingested']}")

print("\n--- STEP 3: SCHEMA & TYPES ---")
print(df.dtypes)

print("\n--- STEP 4: FIRST 5 ROWS ---")
print(df.head())

print("\n--- STEP 5: MISSING VALUES & DUPLICATES ---")
print(f"Missing Values:\n{df.isnull().sum()}")
print(f"Duplicate Rows: {df.duplicated().sum()}")
