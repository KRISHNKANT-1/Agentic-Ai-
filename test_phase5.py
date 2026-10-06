import sys
import os

# Add the src folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
from autonomous_data_analyse_agent.core.sandbox import LocalExecutionSandbox

# Mock a dataframe
df = pd.DataFrame({
    'id': [15000, 15001, 15002],
    'Age': [10, 10, 10]
})

print("--- ORIGINAL DATAFRAME ---")
print(df)

# Mock LLM generated code (Notice how it interacts with 'df')
llm_code = """
print("Dropping 'id' column to prevent target leakage...")
df = df.drop(columns=['id'])
print(f"New Shape: {df.shape}")
"""

print("\n--- EXECUTING LLM CODE IN SANDBOX ---")
sandbox = LocalExecutionSandbox()
modified_df, output, success = sandbox.execute_code(df, llm_code)

print("\n--- SANDBOX OUTPUT ---")
print(output)

print("--- MODIFIED DATAFRAME ---")
print(modified_df)

print(f"\nExecution Success: {success}")
