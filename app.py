import streamlit as st
import pandas as pd
import json
import time
import os
import sys
from dotenv import load_dotenv

# Add src to path for backend integration
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from autonomous_data_analyse_agent.data.detect import detect_file_properties
from autonomous_data_analyse_agent.data.ingest import ingest_data
from autonomous_data_analyse_agent.data.validate import DataQualityValidator
from autonomous_data_analyse_agent.core.agent import AutonomousDataAgent
from autonomous_data_analyse_agent.core.sandbox import LocalExecutionSandbox

# --- 1. UI AESTHETIC CONFIGURATION (Matched to Video) ---
st.set_page_config(page_title="SPEKTRAL | Data Agent", page_icon="⬛", layout="wide")

# Custom CSS to match the dark, futuristic, monospace aesthetic from the video
st.markdown("""
<style>
    /* Global Background and Text */
    .stApp {
        background-color: #0a0a0a;
        color: #e0e0e0;
        font-family: 'Courier New', Courier, monospace;
    }
    
    /* Headers matching 'IMAGINATION ENGINEERING' style */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #ffffff;
    }
    
    /* Terminal style blocks */
    .stCodeBlock, pre {
        background-color: #111111 !important;
        border: 1px solid #333333;
        color: #00ff00 !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Minimalist buttons */
    .stButton>button {
        background-color: transparent;
        color: #ffffff;
        border: 1px solid #ffffff;
        border-radius: 0px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        background-color: #ffffff;
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. BACKEND INITIALIZATION ---
load_dotenv(override=True)
if "sandbox" not in st.session_state:
    st.session_state.sandbox = LocalExecutionSandbox()
    st.session_state.df = None
    st.session_state.profile = None
    st.session_state.messages = []

# --- 3. UI LAYOUT ---
st.title("SPEKTRAL // AUTONOMOUS_DATA_AGENT")
st.markdown("---")

# Sidebar for Ingestion & Config
with st.sidebar:
    st.markdown("### [00] SYSTEM_CONFIG")
    provider_choice = st.selectbox("LLM Provider", ["google", "openai", "groq", "ollama"])
    
    # Dynamically instantiate or update the agent based on sidebar choice
    if "agent" not in st.session_state or st.session_state.agent.provider != provider_choice:
        try:
            st.session_state.agent = AutonomousDataAgent(provider=provider_choice)
        except Exception as e:
            st.error(f"Failed to load {provider_choice.upper()} agent. Error: {str(e)}")
            
    st.markdown("---")
    st.markdown("### [01] INGESTION_PROTOCOL")
    data_source = st.selectbox("Select Source", ["Local File Upload", "URL / Dataset Link", "Raw Paste"])
    
    uploaded_file = None
    if data_source == "Local File Upload":
        uploaded_file = st.file_uploader("Upload CSV/TXT", type=['csv', 'txt'])
    elif data_source == "URL / Dataset Link":
        url = st.text_input("Enter URL")
    else:
        raw_text = st.text_area("Paste Raw Data")
        
    if st.button("INITIALIZE_CONNECTION"):
        if uploaded_file:
            # Save uploaded file temporarily to use our backend logic
            temp_path = f"/tmp/{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            with st.spinner("Initializing Secure Connection..."):
                # Terminal animation effect from video
                status = st.empty()
                status.code("[01] Verifying Network Protocols...\n>> Checking Transport Layer\n>> Validating Schema")
                time.sleep(1)
                
                # Backend execution
                props = detect_file_properties(temp_path)
                status.code(f"[02] Establishing Link...\n>> Encoding: {props['encoding']}\n>> Delimiter: '{props['delimiter']}'")
                time.sleep(1)
                
                ingest_result = ingest_data(temp_path, encoding=props['encoding'], delimiter=props['delimiter'])
                st.session_state.df = ingest_result['dataframe']
                
                status.code("[03] Optimizing Performance...\n>> Generating Statistical Profile\n|||||||||||||||||||||| 100%")
                time.sleep(1)
                
                st.session_state.profile = json.dumps(DataQualityValidator.audit(st.session_state.df))
                status.empty()
                
            st.success("DATA_SECURED")

# --- 4. MAIN DASHBOARD ---
if st.session_state.df is not None:
    st.markdown("### DATA_PREVIEW & METRICS")
    
    # Calculate cool metrics
    prof = json.loads(st.session_state.profile)
    missing_flags = len(prof.get("missingness_flags", {}))
    zero_var = len(prof.get("zero_variance_cols", []))
    total_flags = missing_flags + zero_var
    
    # Display visual metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Rows", st.session_state.df.shape[0])
    m2.metric("Columns", st.session_state.df.shape[1])
    m3.metric("Data Quality Alerts", total_flags, delta="-100% Secure" if total_flags == 0 else f"{total_flags} Issues", delta_color="inverse")
    
    st.markdown("---")
    st.markdown(f"**Data Preview: {st.session_state.df.shape[0]} Rows × {st.session_state.df.shape[1]} Columns**")
    st.dataframe(st.session_state.df, use_container_width=True)
    
    with st.expander("VIEW RAW SYSTEM PROFILE (JSON)"):
        st.json(prof)
    
    # UI Export Feature
    csv = st.session_state.df.to_csv(index=False).encode('utf-8')
    st.download_button("EXPORT_CLEANED_DATA [CSV]", data=csv, file_name="cleaned_dataset.csv", mime="text/csv")
        
    st.markdown("---")
    st.markdown("---")
    st.markdown("### [02] QUICK_ACTIONS")
    q1, q2, q3, q4, q5 = st.columns(5)
    
    if "direct_output" not in st.session_state:
        st.session_state.direct_output = None
        
    action_type = None
    if q1.button("🧹 CLEANING_AGENT"): action_type = "clean"
    if q2.button("🔬 ANALYSE_AGENT"): action_type = "analyse"
    if q3.button("📊 VISUALIZE_AGENT"): action_type = "visualize"
    if q4.button("💡 INSIGHT_AGENT"): action_type = "insight"
    if q5.button("📝 REPORT_AGENT"): action_type = "report"
        
    # Execute native python action instantly
    if action_type:
        with st.status("Executing Native Python Action...", expanded=True) as status_box:
            import io
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            df = st.session_state.df
            img_bytes = None
            code_output = ""
            response = ""
            
            try:
                if action_type == "clean":
                    # Drop >40% missing and impute median
                    st.session_state.df = df.dropna(thresh=int(df.shape[0]*0.6), axis=1)
                    st.session_state.df = st.session_state.df.fillna(st.session_state.df.median(numeric_only=True))
                    st.session_state.profile = json.dumps(DataQualityValidator.audit(st.session_state.df))
                    response = "### 🧹 CLEANING_AGENT\nSuccessfully dropped columns with >40% missing data and imputed numeric blanks with the median."
                    code_output = str(st.session_state.df.head())
                    
                elif action_type == "analyse":
                    numeric_df = df.select_dtypes(include='number')
                    corr = numeric_df.corr().abs().unstack().sort_values(ascending=False).drop_duplicates()
                    top_corr = corr[corr < 1.0].head(5)
                    response = "### 🔬 ANALYSE_AGENT\n**Top 5 Highly Correlated Features:**\n\nThese features move together. High correlation can indicate predictive power, but can also cause multicollinearity in machine learning models."
                    code_output = top_corr.to_string()
                    
                elif action_type == "visualize":
                    numeric_df = df.select_dtypes(include='number')
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', facecolor='black', edgecolor='none')
                    img_bytes = buf.getvalue()
                    plt.close()
                    response = "### 📊 VISUALIZE_AGENT\nGenerated native Seaborn correlation matrix heatmap."
                    
                elif action_type == "insight":
                    prof_data = json.loads(st.session_state.profile)
                    issues = prof_data.get("missingness_flags", {})
                    zeros = prof_data.get("zero_variance_cols", [])
                    response = f"### 💡 INSIGHT_AGENT\n**Quick System Diagnostics:**\n- 🚨 Found **{len(zeros)}** constant/zero-variance columns (these provide zero predictive power).\n- 🚨 Found **{len(issues)}** columns with missing data.\n\n**>> RECOMMENDATION:** Run the CLEANING_AGENT to automatically sanitize the dataset before modeling."
                    
                elif action_type == "report":
                    response = f"### 📝 EXECUTIVE REPORT\n\n- **Total Database Rows:** {df.shape[0]}\n- **Total Database Columns:** {df.shape[1]}\n- **Memory Footprint:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n**Schema Types:**"
                    code_output = df.dtypes.to_string()
                    
                st.session_state.direct_output = {
                    "response": response,
                    "code_output": code_output if code_output else None,
                    "image_bytes": img_bytes
                }
                status_box.update(label="Action Complete!", state="complete", expanded=False)
            except Exception as e:
                status_box.update(label="Critical Error", state="error", expanded=True)
                st.error(f"⚠️ FAULT DETECTED: {str(e)}")

    if st.session_state.direct_output:
        with st.container():
            st.markdown("#### >> DIRECT_ACTION_RESULT")
            st.markdown(st.session_state.direct_output["response"])
            if st.session_state.direct_output.get("code_output"):
                st.code(st.session_state.direct_output["code_output"], language="bash")
            if st.session_state.direct_output.get("image_bytes"):
                st.image(st.session_state.direct_output["image_bytes"])
            if st.button("✖️ Clear Result"):
                st.session_state.direct_output = None
                st.rerun()

    st.markdown("---")
    st.markdown("### [03] AGENT_CONSOLE")
    
    # Render chat history
    for msg in st.session_state.messages:
        avatar = "👤" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            import re
            clean_content = re.sub(r'\*\*ACTION:\*\*.*?```python.*?```', '', msg["content"], flags=re.DOTALL)
            clean_content = re.sub(r'```python.*?```', '', clean_content, flags=re.DOTALL)
            st.markdown(clean_content)
            
            if "code_output" in msg and msg["code_output"]:
                st.success(f"Result:\n{msg['code_output']}")
            if "image_bytes" in msg and msg["image_bytes"]:
                if isinstance(msg["image_bytes"], list):
                    for img in msg["image_bytes"]:
                        st.image(img)
                else:
                    st.image(msg["image_bytes"])
                
    # Chat Input override
    if prompt := st.chat_input("Enter custom analytical objective..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
            
        with st.chat_message("assistant", avatar="🤖"):
            with st.status("Analyzing Objective...", expanded=True) as status_box:
                try:
                    full_prompt = f"User Objective: {prompt}\n\nCurrent Profile:\n{st.session_state.profile}\n\nWhat is your next step?"
                    status_box.update(label="Querying Neural Network...", state="running")
                    response = st.session_state.agent.chat(full_prompt)
                    
                    # Check for code execution
                    code_to_run = st.session_state.agent.extract_code(response)
                    if code_to_run:
                        status_box.update(label="Executing Sandbox Code...", state="running")
                        st.session_state.df, output, success = st.session_state.sandbox.execute_code(st.session_state.df, code_to_run)
                        status_box.update(label="Process Complete!", state="complete", expanded=False)
                        
                        import re
                        clean_response = re.sub(r'\*\*ACTION:\*\*.*?```python.*?```', '', response, flags=re.DOTALL)
                        clean_response = re.sub(r'```python.*?```', '', clean_response, flags=re.DOTALL)
                        
                        st.markdown(clean_response)
                        
                        if output and not success:
                            st.error(f"Execution Error:\n{output}")
                        elif output and success:
                            st.success(f"Result:\n{output}")
                        
                        # Intercept and display any charts generated by the LLM
                        import glob
                        png_files = glob.glob("*.png")
                        img_bytes_list = []
                        for img_file in png_files:
                            st.image(img_file)
                            with open(img_file, "rb") as f:
                                img_bytes_list.append(f.read())
                            os.remove(img_file)
                        
                        # Update profile for next turn
                        st.session_state.profile = json.dumps(DataQualityValidator.audit(st.session_state.df))
                        
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response,
                            "code_output": output,
                            "image_bytes": img_bytes_list
                        })
                    else:
                        status_box.update(label="Response Generated", state="complete", expanded=False)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    status_box.update(label="Critical Error", state="error", expanded=True)
                    st.error("⚠️ SYSTEM FAULT DETECTED")
                    st.warning(f"Error Details: {str(e)}")
                    st.info("💡 Quick Fix: Change the 'LLM Provider' in the left sidebar to 'OpenAI' or 'Groq' to instantly bypass Google's limits!")
else:
    st.markdown("### AWAITING_DATA_SOURCE...")
    st.markdown("Please initialize a connection via the sidebar to begin analysis.")
