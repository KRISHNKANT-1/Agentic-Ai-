import sys
import os
import json

# Add the src folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from autonomous_data_analyse_agent.data.detect import detect_file_properties
from autonomous_data_analyse_agent.data.ingest import ingest_data
from autonomous_data_analyse_agent.data.profile import generate_profile

file_path = "/Users/krishnkant/Downloads/sample_submission (1).csv"

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    sys.exit(1)

# Ingest
props = detect_file_properties(file_path)
result = ingest_data(file_path, encoding=props['encoding'], delimiter=props['delimiter'])
df = result['dataframe']

# Profile
print("Generating LLM-Friendly Statistical Profile...")
profile_json = generate_profile(df)
print(profile_json)
