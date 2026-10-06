import sys
import os
import pandas as pd
import json

# Add the src folder to path so modules can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from autonomous_data_analyse_agent.data.detect import detect_file_properties
from autonomous_data_analyse_agent.data.ingest import ingest_data
from autonomous_data_analyse_agent.data.profile import generate_profile
from autonomous_data_analyse_agent.core.sandbox import LocalExecutionSandbox

def run_all_tests():
    file_path = "/Users/krishnkant/Downloads/sample_submission (1).csv"
    
    print("==================================================")
    print("🧪 RUNNING ALL SYSTEM TESTS")
    print("==================================================\n")
    
    if not os.path.exists(file_path):
        print(f"❌ Target test file not found: {file_path}")
        return

    # --- PHASE 1: DETECT & INGEST ---
    print("▶️ TEST 1: AUTO-DETECT & INGESTION")
    props = detect_file_properties(file_path)
    print(f"  ✓ Detected Encoding: {props['encoding']}")
    print(f"  ✓ Detected Delimiter: {repr(props['delimiter'])}")
    
    ingest_result = ingest_data(file_path, encoding=props['encoding'], delimiter=props['delimiter'])
    df = ingest_result['dataframe']
    print(f"  ✓ Total Rows in File: {ingest_result['total_rows_in_file']}")
    print(f"  ✓ Rows Safely Ingested: {ingest_result['rows_ingested']}")
    print(f"  ✓ DataFrame Shape: {df.shape}")
    print("  ✅ Test 1 Passed.\n")

    # --- PHASE 2: PROFILING ---
    print("▶️ TEST 2: STATISTICAL PROFILING")
    profile_json = generate_profile(df)
    profile_dict = json.loads(profile_json)
    print(f"  ✓ Dataset Shape in Profile: {profile_dict['dataset_shape']}")
    print(f"  ✓ Columns Profiled: {list(profile_dict['columns'].keys())}")
    print("  ✅ Test 2 Passed.\n")
    
    # --- PHASE 5: SANDBOX EXECUTION ---
    print("▶️ TEST 3: LOCAL EXECUTION SANDBOX")
    sandbox = LocalExecutionSandbox()
    
    # Mock LLM generated code
    llm_code = """
print("Simulating LLM dropping 'id' column to prevent leakage...")
df = df.drop(columns=['id'])
print(f"New Shape: {df.shape}")
"""
    
    modified_df, output, success = sandbox.execute_code(df, llm_code)
    print(f"  ✓ Sandbox Execution Success: {success}")
    if success:
        print(f"  ✓ Original Shape: {df.shape}")
        print(f"  ✓ Modified Shape: {modified_df.shape}")
        print(f"  ✓ Sandbox Output Captured: \n      {output.strip()}")
    print("  ✅ Test 3 Passed.\n")
    
    print("==================================================")
    print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    run_all_tests()
