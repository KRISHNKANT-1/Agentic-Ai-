import argparse
import sys
import os
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from autonomous_data_analyse_agent.data.detect import detect_file_properties
from autonomous_data_analyse_agent.data.ingest import ingest_data
from autonomous_data_analyse_agent.data.profile import generate_profile
from autonomous_data_analyse_agent.core.agent import AutonomousDataAgent
from autonomous_data_analyse_agent.core.sandbox import LocalExecutionSandbox

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Autonomous Data Analysis Agent")
    parser.add_argument("file_path", help="Path to the dataset (e.g., CSV)")
    parser.add_argument("--objective", type=str, default="Analyze the dataset for anomalies and clean it.", help="Target analytical objective")
    parser.add_argument("--provider", type=str, choices=["google", "openai", "groq"], default="google", help="LLM Provider")
    parser.add_argument("--model", type=str, default=None, help="Model name (defaults to env var based on provider)")
    parser.add_argument("--max_steps", type=int, default=5, help="Maximum number of ReAct loop iterations")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file_path):
        print(f"Error: File '{args.file_path}' not found.")
        return

    print(f"🚀 Initializing Agent with {args.provider.upper()} ({args.model})...")
    
    print("\n[1/4] 🔍 Detecting file properties...")
    props = detect_file_properties(args.file_path)
    
    print("\n[2/4] 📥 Ingesting data safely...")
    ingest_result = ingest_data(args.file_path, encoding=props['encoding'], delimiter=props['delimiter'])
    df = ingest_result['dataframe']
    
    print("\n[3/4] 📊 Generating privacy-preserving statistical profile...")
    profile_json = generate_profile(df)
    
    print("\n[4/4] 🧠 Entering Autonomous ReAct Loop...")
    try:
        agent = AutonomousDataAgent(provider=args.provider, model_name=args.model)
        sandbox = LocalExecutionSandbox()
        
        # Initial message to kick off the loop
        prompt = f"User Objective: {args.objective}\n\nInitial Dataset Profile (JSON):\n{profile_json}\n\nWhat is your first step?"
        
        step = 1
        while step <= args.max_steps:
            print(f"\n--- 🤖 AGENT THINKING (Step {step}/{args.max_steps}) ---")
            response = agent.chat(prompt)
            print(response)
            
            # Check if agent wants to execute code
            code_to_run = agent.extract_code(response)
            if not code_to_run:
                print("\n✅ Agent has completed the objective (No code generated).")
                break
                
            print("\n--- 💻 SANDBOX EXECUTING CODE ---")
            df, output, success = sandbox.execute_code(df, code_to_run)
            
            print(f"Execution Success: {success}")
            print(f"Output:\n{output.strip() if output.strip() else '<No standard output>'}")
            
            # Feed the output back to the agent for the next iteration
            prompt = f"Sandbox Execution Success: {success}\nOutput:\n{output}\n\nWhat is your next step based on these results?"
            step += 1
            
        if step > args.max_steps:
            print(f"\n⚠️ Max iterations ({args.max_steps}) reached. Halting loop.")
            
    except Exception as e:
        print(f"\n❌ Pipeline Error. Ensure your API keys are valid. Details: {e}")

if __name__ == "__main__":
    main()
