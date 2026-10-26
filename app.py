import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import os
import sys
import io
import re
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# Add src to path for backend integration
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from autonomous_data_analyse_agent.data.detect import detect_file_properties
from autonomous_data_analyse_agent.data.ingest import ingest_data
from autonomous_data_analyse_agent.data.validate import DataQualityValidator
from autonomous_data_analyse_agent.core.agent import AutonomousDataAgent
from autonomous_data_analyse_agent.core.sandbox import LocalExecutionSandbox

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SPEKTRAL AI | Autonomous Data Analyst",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. LUXURY MODERN DESIGN SYSTEM (CSS) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Theme & Typography */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #0B0F19;
        color: #F3F4F6;
    }

    /* Ambient Background Glow */
    .stApp {
        background: radial-gradient(circle at 15% 10%, rgba(99, 102, 241, 0.08) 0%, transparent 45%),
                    radial-gradient(circle at 85% 80%, rgba(6, 182, 212, 0.06) 0%, transparent 50%),
                    #0B0F19;
    }

    /* Hide Default Chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background-color: transparent !important;}

    /* Typography Hierarchy */
    h1, h2, h3, h4 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    /* Gradient Brand Header */
    .brand-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FFFFFF 0%, #A5B4FC 40%, #38BDF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.03em;
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .brand-subtitle {
        color: #94A3B8;
        font-size: 0.95rem;
        font-weight: 400;
        margin-bottom: 1.5rem;
    }

    /* Custom Glassmorphism Containers */
    .glass-card {
        background: rgba(17, 24, 39, 0.7);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        transition: all 0.2s ease-in-out;
    }

    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.3);
        box-shadow: 0 14px 40px -10px rgba(99, 102, 241, 0.15);
    }

    .feature-card {
        background: rgba(17, 24, 39, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 20px;
        height: 100%;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .feature-card:hover {
        transform: translateY(-3px);
        border-color: rgba(99, 102, 241, 0.4);
        background: rgba(23, 31, 51, 0.75);
    }

    /* Status Badges */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .badge-cyan {
        background: rgba(6, 182, 212, 0.15);
        color: #38BDF8;
        border: 1px solid rgba(6, 182, 212, 0.3);
    }

    .badge-indigo {
        background: rgba(99, 102, 241, 0.15);
        color: #A5B4FC;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }

    .badge-emerald {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .badge-amber {
        background: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: rgba(17, 24, 39, 0.75) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        padding: 16px 20px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
    }

    div[data-testid="stMetricLabel"] p {
        color: #94A3B8 !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    div[data-testid="stMetricValue"] div {
        color: #FFFFFF !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    /* Modern Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(17, 24, 39, 0.6);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }

    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 8px;
        color: #94A3B8;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 0 20px;
        border: none !important;
        background-color: transparent;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.25) 0%, rgba(6, 182, 212, 0.15) 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(99, 102, 241, 0.4) !important;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 8px 18px !important;
        letter-spacing: 0.01em !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3) !important;
        transition: all 0.2s ease !important;
    }

    .stButton>button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.45) !important;
        border-color: rgba(255, 255, 255, 0.3) !important;
    }

    /* Secondary / Action Buttons */
    .action-btn button {
        background: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: none !important;
    }
    
    .action-btn button:hover {
        background: rgba(49, 46, 129, 0.6) !important;
        border-color: #6366F1 !important;
    }

    /* Terminal & Code Blocks */
    .stCodeBlock, pre, code {
        font-family: 'JetBrains Mono', monospace !important;
        border-radius: 10px !important;
        background-color: #0D1117 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0A0D14 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* Chat Messages */
    .stChatMessage {
        background-color: rgba(17, 24, 39, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 14px !important;
        padding: 16px !important;
        margin-bottom: 12px !important;
    }

    /* Custom Dataframe Container */
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. SAMPLE DATASETS GENERATOR (FOR INSTANT DEMOS) ---
def get_sample_dataset(name: str) -> pd.DataFrame:
    """Generates clean, realistic demo datasets for instant evaluation."""
    np.random.seed(42)
    if name == "Customer Churn & Behavior":
        n = 300
        tenure = np.random.randint(1, 72, n)
        monthly = np.round(np.random.uniform(25.0, 120.0, n), 2)
        churn_prob = np.clip(0.6 - (tenure / 120.0) + (monthly / 250.0), 0.05, 0.95)
        churn = (np.random.rand(n) < churn_prob).astype(int)
        
        df = pd.DataFrame({
            "CustomerID": [f"USR-{1000 + i}" for i in range(n)],
            "Age": np.random.randint(18, 70, n),
            "Tenure_Months": tenure,
            "Monthly_Charges": monthly,
            "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.5, 0.3, 0.2]),
            "Internet_Service": np.random.choice(["Fiber optic", "DSL", "No"], n, p=[0.45, 0.35, 0.20]),
            "Support_Tickets": np.random.poisson(1.8, n),
            "Total_Spend": np.round(tenure * monthly + np.random.normal(0, 50, n), 2),
            "Churn": churn
        })
        # Add a couple of realistic blanks to demonstrate cleaning
        mask = np.random.rand(n) < 0.05
        df.loc[mask, "Total_Spend"] = np.nan
        return df

    elif name == "Titanic Survival Demographics":
        n = 280
        pclass = np.random.choice([1, 2, 3], n, p=[0.25, 0.25, 0.5])
        gender = np.random.choice(["female", "male"], n, p=[0.38, 0.62])
        age = np.round(np.random.normal(29, 14, n), 1)
        age = np.clip(age, 1, 75)
        fare = np.round(np.where(pclass == 1, np.random.exponential(60, n) + 30,
                         np.where(pclass == 2, np.random.exponential(20, n) + 12,
                                  np.random.exponential(8, n) + 7)), 2)
        survived = np.where((gender == "female") & (pclass < 3), 1,
                    np.where((gender == "male") & (pclass == 3), 0,
                             np.random.choice([0, 1], n, p=[0.6, 0.4])))
        df = pd.DataFrame({
            "PassengerId": range(1, n + 1),
            "Survived": survived,
            "Pclass": pclass,
            "Sex": gender,
            "Age": age,
            "SibSp": np.random.choice([0, 1, 2], n, p=[0.7, 0.2, 0.1]),
            "Parch": np.random.choice([0, 1, 2], n, p=[0.75, 0.15, 0.1]),
            "Fare": fare,
            "Embarked": np.random.choice(["S", "C", "Q"], n, p=[0.7, 0.2, 0.1])
        })
        # Add realistic missing values in Age
        mask = np.random.rand(n) < 0.10
        df.loc[mask, "Age"] = np.nan
        return df

    else:  # E-Commerce Sales & Revenue
        n = 350
        categories = ["Electronics", "Fashion", "Home & Kitchen", "Books", "Beauty"]
        units = np.random.randint(1, 10, n)
        unit_price = np.round(np.random.uniform(10.0, 350.0, n), 2)
        discount = np.random.choice([0.0, 0.05, 0.10, 0.20, 0.30], n, p=[0.4, 0.2, 0.2, 0.15, 0.05])
        
        df = pd.DataFrame({
            "Transaction_ID": [f"TX-{5000 + i}" for i in range(n)],
            "Category": np.random.choice(categories, n),
            "Units_Sold": units,
            "Unit_Price": unit_price,
            "Discount_Rate": discount,
            "Gross_Revenue": np.round(units * unit_price, 2),
            "Net_Revenue": np.round(units * unit_price * (1 - discount), 2),
            "Customer_Rating": np.round(np.random.uniform(2.5, 5.0, n), 1),
            "Repeat_Customer": np.random.choice([0, 1], n, p=[0.35, 0.65])
        })
        return df

# --- 4. BACKEND INITIALIZATION & SESSION STATE ---
load_dotenv(override=True)

if "sandbox" not in st.session_state:
    st.session_state.sandbox = LocalExecutionSandbox()
    st.session_state.df = None
    st.session_state.dataset_name = "Untitled Dataset"
    st.session_state.profile = None
    st.session_state.messages = []
    st.session_state.direct_output = None

# --- 5. TOP NAVBAR / BRAND HEADER ---
col_logo, col_stat = st.columns([3, 1])
with col_logo:
    st.markdown('<div class="brand-title">⚡ SPEKTRAL AI <span style="font-size: 1rem; color: #6366F1; font-weight: 600;">// AUTONOMOUS DATA ENGINE</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Autonomous Data Science Agent with Multi-LLM Orchestration & Sandboxed Python Execution</div>', unsafe_allow_html=True)

with col_stat:
    st.markdown(
        """
        <div style="display: flex; justify-content: flex-end; gap: 8px; align-items: center; padding-top: 8px;">
            <span class="badge-pill badge-emerald">● System Online</span>
            <span class="badge-pill badge-indigo">🛡️ Sandboxed</span>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- 6. SIDEBAR: CONTROL CENTER ---
with st.sidebar:
    st.markdown("### ⚙️ System Configuration")
    
    provider_raw = st.selectbox(
        "AI Engine Provider",
        ["groq (Cloud - Ultra Fast ⚡)", "google (Gemini 3.5)", "ollama (Local Server)", "openai (GPT-4o)"],
        help="Select the AI brain for orchestration, code generation, and hypothesis testing."
    )
    provider_choice = provider_raw.split(" ")[0].lower()
    
    # Dynamically instantiate or update agent based on selection
    if "agent" not in st.session_state or getattr(st.session_state.agent, 'provider', None) != provider_choice:
        try:
            st.session_state.agent = AutonomousDataAgent(provider=provider_choice)
            st.success(f"Connected to {provider_choice.upper()} Agent")
        except Exception as e:
            st.error(f"Failed to load {provider_choice.upper()} agent: {str(e)}")

    st.markdown("---")
    st.markdown("### 📥 Ingestion Protocol")
    
    data_source = st.selectbox(
        "Data Ingestion Source",
        ["⚡ Demo Datasets (Instant)", "📁 Local File Upload", "🌐 URL / Dataset Link", "📋 Raw Text / CSV Paste"]
    )
    
    load_triggered = False
    new_df = None
    dataset_label = "Dataset"

    if data_source == "⚡ Demo Datasets (Instant)":
        sample_choice = st.selectbox(
            "Select Benchmark Dataset",
            ["Customer Churn & Behavior", "Titanic Survival Demographics", "E-Commerce Sales & Revenue"]
        )
        if st.button("🚀 Load Benchmark Dataset", use_container_width=True):
            with st.spinner("Generating statistical benchmark..."):
                new_df = get_sample_dataset(sample_choice)
                dataset_label = sample_choice
                load_triggered = True

    elif data_source == "📁 Local File Upload":
        uploaded_file = st.file_uploader("Upload CSV, TXT, or JSON", type=['csv', 'txt', 'json'])
        if uploaded_file and st.button("⚡ Process & Secure Data", use_container_width=True):
            temp_path = f"/tmp/{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            with st.spinner("Analyzing encoding & schemas..."):
                props = detect_file_properties(temp_path)
                ingest_result = ingest_data(temp_path, encoding=props['encoding'], delimiter=props['delimiter'])
                new_df = ingest_result['dataframe']
                dataset_label = uploaded_file.name
                load_triggered = True

    elif data_source == "🌐 URL / Dataset Link":
        url_input = st.text_input("Dataset URL", placeholder="https://raw.githubusercontent.com/.../data.csv")
        if url_input and st.button("🌐 Ingest Remote Dataset", use_container_width=True):
            try:
                with st.spinner("Fetching remote stream..."):
                    new_df = pd.read_csv(url_input)
                    dataset_label = url_input.split("/")[-1] or "Remote Dataset"
                    load_triggered = True
            except Exception as e:
                st.error(f"Failed to fetch dataset from URL: {str(e)}")

    else:
        raw_text = st.text_area("Paste CSV Data", height=130, placeholder="col1,col2,col3\n1,2,3\n4,5,6")
        if raw_text and st.button("📋 Parse Raw Text", use_container_width=True):
            try:
                new_df = pd.read_csv(io.StringIO(raw_text))
                dataset_label = "Pasted Dataset"
                load_triggered = True
            except Exception as e:
                st.error(f"Failed to parse text: {str(e)}")

    # Handle dataset loading transition
    if load_triggered and new_df is not None:
        st.session_state.df = new_df
        st.session_state.dataset_name = dataset_label
        st.session_state.profile = json.dumps(DataQualityValidator.audit(st.session_state.df))
        st.session_state.direct_output = None
        st.rerun()

    # Active dataset card in sidebar
    if st.session_state.df is not None:
        st.markdown("---")
        st.markdown("### 📊 Active Workspace")
        st.markdown(
            f"""
            <div style="background: rgba(17, 24, 39, 0.8); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 14px; margin-bottom: 12px;">
                <div style="font-weight: 700; color: #FFFFFF; font-size: 0.95rem; margin-bottom: 6px;">📄 {st.session_state.dataset_name}</div>
                <div style="color: #94A3B8; font-size: 0.82rem;">Rows: <b style="color: #F3F4F6;">{st.session_state.df.shape[0]:,}</b></div>
                <div style="color: #94A3B8; font-size: 0.82rem;">Columns: <b style="color: #F3F4F6;">{st.session_state.df.shape[1]}</b></div>
                <div style="color: #94A3B8; font-size: 0.82rem;">Memory: <b style="color: #F3F4F6;">{st.session_state.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB</b></div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("🔄 Unload / Reset Workspace", use_container_width=True):
            st.session_state.df = None
            st.session_state.profile = None
            st.session_state.messages = []
            st.session_state.direct_output = None
            st.rerun()

# --- 7. MAIN VIEWPORT ---

# SCENARIO A: NO DATASET LOADED -> HERO LANDING EXPERIENCE
if st.session_state.df is None:
    st.markdown(
        """
        <div class="glass-card" style="text-align: center; padding: 48px 24px; margin-bottom: 30px; background: radial-gradient(circle at 50% 20%, rgba(99, 102, 241, 0.15) 0%, rgba(17, 24, 39, 0.8) 60%);">
            <span class="badge-pill badge-indigo" style="margin-bottom: 16px;">✨ Autonomous Intelligence Platform</span>
            <h1 style="font-size: 2.8rem; margin: 12px 0; background: linear-gradient(135deg, #FFFFFF 0%, #E0E7FF 60%, #38BDF8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Transform Raw Data into Actionable Insights
            </h1>
            <p style="color: #94A3B8; max-width: 700px; margin: 0 auto 28px auto; font-size: 1.1rem; line-height: 1.6;">
                Spektral is an autonomous agent that ingests, cleans, explores, visualizes, and models complex datasets. 
                Equipped with sandboxed Python code execution and multi-model LLM reasoning.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("#### 🚀 Quick Start: Try a Benchmark Dataset (1-Click Evaluation)")
    demo_c1, demo_c2, demo_c3 = st.columns(3)
    
    with demo_c1:
        st.markdown(
            """
            <div class="feature-card">
                <div style="font-size: 1.6rem; margin-bottom: 8px;">👥</div>
                <div style="font-weight: 700; font-size: 1.05rem; color: #FFFFFF; margin-bottom: 6px;">Customer Churn</div>
                <div style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 14px; min-height: 48px;">
                    Predict churn behavior across demographics, tenure, charges, and support tickets.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Load Customer Churn Dataset", key="load_churn", use_container_width=True):
            st.session_state.df = get_sample_dataset("Customer Churn & Behavior")
            st.session_state.dataset_name = "Customer Churn & Behavior"
            st.session_state.profile = json.dumps(DataQualityValidator.audit(st.session_state.df))
            st.rerun()

    with demo_c2:
        st.markdown(
            """
            <div class="feature-card">
                <div style="font-size: 1.6rem; margin-bottom: 8px;">🛳️</div>
                <div style="font-weight: 700; font-size: 1.05rem; color: #FFFFFF; margin-bottom: 6px;">Titanic Demographics</div>
                <div style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 14px; min-height: 48px;">
                    Classic survival analysis with passenger classes, ages, family counts, and fares.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Load Titanic Dataset", key="load_titanic", use_container_width=True):
            st.session_state.df = get_sample_dataset("Titanic Survival Demographics")
            st.session_state.dataset_name = "Titanic Survival Demographics"
            st.session_state.profile = json.dumps(DataQualityValidator.audit(st.session_state.df))
            st.rerun()

    with demo_c3:
        st.markdown(
            """
            <div class="feature-card">
                <div style="font-size: 1.6rem; margin-bottom: 8px;">📈</div>
                <div style="font-weight: 700; font-size: 1.05rem; color: #FFFFFF; margin-bottom: 6px;">E-Commerce Sales</div>
                <div style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 14px; min-height: 48px;">
                    Multi-category transactions, unit pricing, discount rates, gross and net revenues.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Load Sales Dataset", key="load_sales", use_container_width=True):
            st.session_state.df = get_sample_dataset("E-Commerce Sales & Revenue")
            st.session_state.dataset_name = "E-Commerce Sales & Revenue"
            st.session_state.profile = json.dumps(DataQualityValidator.audit(st.session_state.df))
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🛡️ Autonomous Architectural Capabilities")
    
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.markdown(
            """
            <div class="feature-card">
                <span class="badge-pill badge-indigo" style="margin-bottom: 8px;">01 • Sandbox</span>
                <div style="font-weight: 700; color: #FFFFFF; margin: 8px 0;">Isolated Code Execution</div>
                <div style="color: #94A3B8; font-size: 0.84rem; line-height: 1.5;">
                    Dynamically executes generated Python code inside an isolated memory container without risk.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with f2:
        st.markdown(
            """
            <div class="feature-card">
                <span class="badge-pill badge-cyan" style="margin-bottom: 8px;">02 • Profiling</span>
                <div style="font-weight: 700; color: #FFFFFF; margin: 8px 0;">Automated Data Audit</div>
                <div style="color: #94A3B8; font-size: 0.84rem; line-height: 1.5;">
                    Instantly flags missing values, zero-variance columns, IQR outliers, and cardinality risks.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with f3:
        st.markdown(
            """
            <div class="feature-card">
                <span class="badge-pill badge-emerald" style="margin-bottom: 8px;">03 • Visuals</span>
                <div style="font-weight: 700; color: #FFFFFF; margin: 8px 0;">Smart Visualization</div>
                <div style="color: #94A3B8; font-size: 0.84rem; line-height: 1.5;">
                    Generates native publication-grade heatmaps, feature distributions, and trend analyses.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with f4:
        st.markdown(
            """
            <div class="feature-card">
                <span class="badge-pill badge-amber" style="margin-bottom: 8px;">04 • Intelligence</span>
                <div style="font-weight: 700; color: #FFFFFF; margin: 8px 0;">Multi-LLM Reasoning</div>
                <div style="color: #94A3B8; font-size: 0.84rem; line-height: 1.5;">
                    Seamlessly orchestrates Gemini, OpenAI, Groq, or local Ollama models with context pruning.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# SCENARIO B: DATASET IS LOADED -> FULL ANALYTICS STUDIO
else:
    prof = json.loads(st.session_state.profile) if st.session_state.profile else DataQualityValidator.audit(st.session_state.df)
    
    # Calculate accurate metrics
    missingness = prof.get("missingness", {})
    missing_cols_count = sum(1 for col, data in missingness.items() if data.get("rate", 0) > 0)
    zero_var_count = len(prof.get("zero_variance_cols", []))
    total_outliers = sum(prof.get("outliers_iqr", {}).values())
    total_flags = missing_cols_count + zero_var_count
    
    total_cells = st.session_state.df.shape[0] * st.session_state.df.shape[1]
    missing_cells = st.session_state.df.isnull().sum().sum()
    completeness_rate = round((1.0 - (missing_cells / total_cells if total_cells > 0 else 0)) * 100, 1)

    # --- TOP KPI METRICS BAR ---
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Total Records", f"{st.session_state.df.shape[0]:,}", help="Total number of observations (rows)")
    kpi2.metric("Features", f"{st.session_state.df.shape[1]}", help="Total dimensions (columns)")
    kpi3.metric("Completeness", f"{completeness_rate}%", delta=f"{missing_cells} blanks" if missing_cells > 0 else "100% Intact")
    kpi4.metric("Quality Alerts", total_flags, delta="-All Clean" if total_flags == 0 else f"{total_flags} Attention Flags", delta_color="inverse")
    kpi5.metric("Outliers (IQR)", f"{total_outliers:,}", help="Identified values outside 1.5x Interquartile Range")

    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # --- TAB NAVIGATION ---
    tab_overview, tab_actions, tab_agent = st.tabs([
        "📊 Dataset Intelligence & Quality", 
        "⚡ Autonomous Quick Agents", 
        "💬 Interactive AI Analyst Console"
    ])

    # -------------------------------------------------------------
    # TAB 1: DATASET INTELLIGENCE & QUALITY AUDIT
    # -------------------------------------------------------------
    with tab_overview:
        col_preview, col_audit = st.columns([7, 5])
        
        with col_preview:
            st.markdown("##### 🔍 Interactive Data Preview")
            st.dataframe(st.session_state.df.head(100), use_container_width=True, height=360)
            
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                csv_bytes = st.session_state.df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 Download Current Dataset (CSV)",
                    data=csv_bytes,
                    file_name=f"{st.session_state.dataset_name.lower().replace(' ', '_')}_export.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with d_col2:
                with st.expander("🛠️ Numerical Summary Stats"):
                    st.dataframe(st.session_state.df.describe(), use_container_width=True)

        with col_audit:
            st.markdown("##### 🛡️ Automated Data Quality Audit")
            
            # Health Score Card
            health_color = "#10B981" if total_flags == 0 else ("#F59E0B" if total_flags < 4 else "#EF4444")
            st.markdown(
                f"""
                <div style="background: rgba(17, 24, 39, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 16px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <span style="font-weight: 700; color: #FFFFFF;">Data Health Score</span>
                        <span style="font-weight: 800; color: {health_color}; font-size: 1.2rem;">{completeness_rate}%</span>
                    </div>
                    <div style="background: rgba(255,255,255,0.1); border-radius: 999px; height: 8px; width: 100%; overflow: hidden;">
                        <div style="background: {health_color}; height: 100%; width: {completeness_rate}%;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Missing Value Breakdown
            missing_summary = [
                {"Column": col, "Missing Rate": f"{info['rate']*100:.1f}%", "Severity": "High (>40%)" if info['flag_high'] else "Moderate"}
                for col, info in missingness.items() if info['rate'] > 0
            ]
            if missing_summary:
                st.markdown("**Columns with Missing Values:**")
                st.dataframe(pd.DataFrame(missing_summary), use_container_width=True, height=130)
            else:
                st.success("✅ Zero missing values detected across all columns!")
                
            # Zero Variance & Outliers Notice
            if zero_var_count > 0:
                st.warning(f"⚠️ **Zero-Variance Columns:** {', '.join(prof.get('zero_variance_cols', []))} (provide zero predictive signal)")
            
            with st.expander("📄 View Full JSON Schema Audit"):
                st.json(prof)

    # -------------------------------------------------------------
    # TAB 2: AUTONOMOUS QUICK AGENTS (ONE-CLICK ACTIONS)
    # -------------------------------------------------------------
    with tab_actions:
        st.markdown("##### ⚡ Specialized Autonomous Agents")
        st.markdown("<p style='color: #94A3B8; font-size: 0.9rem;'>Trigger specialized native Python algorithms with pre-built heuristics for cleaning, correlations, and visual telemetry.</p>", unsafe_allow_html=True)

        a1, a2, a3, a4, a5 = st.columns(5)
        
        triggered_action = None
        with a1:
            if st.button("🧹 CLEANING AGENT\nAuto-Impute & Prune", use_container_width=True):
                triggered_action = "clean"
        with a2:
            if st.button("🔬 ANALYSE AGENT\nCollinearity Scan", use_container_width=True):
                triggered_action = "analyse"
        with a3:
            if st.button("📊 VISUALIZE AGENT\nCorrelation Heatmap", use_container_width=True):
                triggered_action = "visualize"
        with a4:
            if st.button("💡 INSIGHT AGENT\nRisk Diagnostics", use_container_width=True):
                triggered_action = "insight"
        with a5:
            if st.button("📝 REPORT AGENT\nExecutive Summary", use_container_width=True):
                triggered_action = "report"

        # Execute selected agent
        if triggered_action:
            with st.status(f"🚀 Deploying {triggered_action.upper()} Agent...", expanded=True) as status_box:
                df = st.session_state.df
                img_bytes = None
                code_output = ""
                response = ""

                try:
                    if triggered_action == "clean":
                        before_shape = df.shape
                        # Drop columns with >40% missing values
                        clean_df = df.dropna(thresh=int(df.shape[0] * 0.6), axis=1)
                        # Fill numeric missing values with median
                        num_cols = clean_df.select_dtypes(include='number').columns
                        clean_df[num_cols] = clean_df[num_cols].fillna(clean_df[num_cols].median())
                        
                        st.session_state.df = clean_df
                        st.session_state.profile = json.dumps(DataQualityValidator.audit(st.session_state.df))
                        
                        response = f"### 🧹 Cleaning Agent Execution Succeeded\n\n- **Before Clean:** {before_shape[0]} rows × {before_shape[1]} cols\n- **After Clean:** {clean_df.shape[0]} rows × {clean_df.shape[1]} cols\n- **Transformation:** Dropped features exceeding 40% missing threshold; imputed numeric gaps using robust column medians."
                        code_output = str(clean_df.head(5))

                    elif triggered_action == "analyse":
                        num_df = df.select_dtypes(include='number')
                        if num_df.shape[1] > 1:
                            corr_matrix = num_df.corr().abs()
                            unstacked = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()
                            top_pairs = unstacked[unstacked < 1.0].head(6)
                            
                            response = "### 🔬 Feature Collinearity & Dependency Audit\n\nHighlighted the strongest statistical feature correlations. High correlations (>0.75) may cause multicollinearity in linear models and indicate redundant features."
                            code_output = top_pairs.to_string()
                        else:
                            response = "### 🔬 Collinearity Scan\nNot enough numeric columns found to construct correlation pairs."

                    elif triggered_action == "visualize":
                        num_df = df.select_dtypes(include='number')
                        if num_df.shape[1] > 1:
                            # Apply dark theme aesthetic
                            plt.style.use('dark_background')
                            fig, ax = plt.subplots(figsize=(9, 6.5), facecolor='#0B0F19')
                            ax.set_facecolor('#0B0F19')
                            
                            sns.heatmap(
                                num_df.corr(), 
                                annot=True, 
                                cmap='mako', 
                                fmt=".2f", 
                                ax=ax, 
                                cbar_kws={'label': 'Correlation Coefficient'},
                                linewidths=0.5,
                                linecolor='#1F2937'
                            )
                            plt.title("Spektral Correlation Topology Matrix", color="#FFFFFF", fontsize=14, fontweight='bold', pad=12)
                            plt.tight_layout()
                            
                            buf = io.BytesIO()
                            plt.savefig(buf, format='png', facecolor='#0B0F19', edgecolor='none', dpi=200)
                            img_bytes = buf.getvalue()
                            plt.close()
                            response = "### 📊 Visual Correlation Heatmap Generated\nNative dark-mode Seaborn correlation topology successfully rendered."
                        else:
                            response = "### 📊 Visualizer Notice\nDataset requires at least 2 numeric features to render a correlation heatmap."

                    elif triggered_action == "insight":
                        prof_data = json.loads(st.session_state.profile)
                        zeros = prof_data.get("zero_variance_cols", [])
                        miss = prof_data.get("missingness", {})
                        miss_cols = [k for k, v in miss.items() if v.get("rate", 0) > 0]
                        outliers = prof_data.get("outliers_iqr", {})
                        high_outlier_cols = [k for k, v in outliers.items() if v > 5]

                        response = (
                            f"### 💡 Automated Data Science Diagnostic Brief\n\n"
                            f"- **Constant Features:** {len(zeros)} detected {f'(`{zeros}`)' if zeros else ''} (recommended to drop).\n"
                            f"- **Missing Value Columns:** {len(miss_cols)} detected {f'(`{miss_cols[:4]}`)' if miss_cols else ''}.\n"
                            f"- **Heavy Outlier Signals:** {len(high_outlier_cols)} columns show significant IQR variance.\n\n"
                            f"**Recommended Strategy:** Run `CLEANING AGENT` first, or ask the AI Analyst below to engineer interaction features."
                        )

                    elif triggered_action == "report":
                        mem_mb = df.memory_usage(deep=True).sum() / 1024**2
                        response = (
                            f"### 📝 Executive Architecture Report\n\n"
                            f"| Metric | Value |\n"
                            f"| :--- | :--- |\n"
                            f"| **Active Dataset** | `{st.session_state.dataset_name}` |\n"
                            f"| **Total Observations** | {df.shape[0]:,} |\n"
                            f"| **Total Dimensions** | {df.shape[1]} |\n"
                            f"| **Memory Footprint** | {mem_mb:.2f} MB |\n"
                            f"| **Numeric Features** | {len(df.select_dtypes(include='number').columns)} |\n"
                            f"| **Categorical Features** | {len(df.select_dtypes(exclude='number').columns)} |\n"
                        )
                        code_output = df.dtypes.to_string()

                    st.session_state.direct_output = {
                        "response": response,
                        "code_output": code_output if code_output else None,
                        "image_bytes": img_bytes
                    }
                    status_box.update(label=f"✅ {triggered_action.upper()} Agent Completed!", state="complete", expanded=False)

                except Exception as e:
                    status_box.update(label="❌ Action Error", state="error", expanded=True)
                    st.error(f"Execution Error: {str(e)}")

        # Render Direct Action Output
        if st.session_state.direct_output:
            st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)
            with st.container():
                st.markdown(
                    """
                    <div style="background: rgba(17, 24, 39, 0.85); border: 1px solid rgba(99, 102, 241, 0.35); border-radius: 14px; padding: 22px;">
                    """,
                    unsafe_allow_html=True
                )
                st.markdown(st.session_state.direct_output["response"])
                
                if st.session_state.direct_output.get("image_bytes"):
                    st.image(st.session_state.direct_output["image_bytes"], use_container_width=True)
                    
                if st.session_state.direct_output.get("code_output"):
                    st.code(st.session_state.direct_output["code_output"], language="bash")
                    
                if st.button("✖️ Dismiss Result Panel"):
                    st.session_state.direct_output = None
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # TAB 3: INTERACTIVE AI AGENT CONSOLE (CHAT & SANDBOX)
    # -------------------------------------------------------------
    with tab_agent:
        col_chat_title, col_chat_reset = st.columns([4, 1])
        with col_chat_title:
            st.markdown("##### 🤖 Sandboxed Conversational Analyst")
            st.markdown("<p style='color: #94A3B8; font-size: 0.9rem;'>Instruct the AI in plain English. The agent formulates hypotheses, generates Python code, runs it in the sandbox, and visualizes output.</p>", unsafe_allow_html=True)
        with col_chat_reset:
            if st.button("🗑️ Reset Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        # Clean any orphaned user messages from previous broken turns
        cleaned_messages = []
        i = 0
        while i < len(st.session_state.messages):
            msg = st.session_state.messages[i]
            if msg["role"] == "user":
                if i + 1 < len(st.session_state.messages) and st.session_state.messages[i + 1]["role"] == "assistant":
                    cleaned_messages.append(msg)
                    cleaned_messages.append(st.session_state.messages[i + 1])
                    i += 2
                else:
                    i += 1
            else:
                cleaned_messages.append(msg)
                i += 1
        st.session_state.messages = cleaned_messages

        # Quick Suggested Prompts (Specific chips requested by user)
        st.markdown("**Suggested Analytical Objectives:**")
        q1, q2, q3, q4, q5 = st.columns(5)
        selected_prompt = None
        with q1:
            if st.button("📊 Distributions", use_container_width=True, help="Plot histograms and distribution curves for all primary numeric features and save to chart.png"):
                selected_prompt = "Plot histograms and distribution curves for all primary numeric features and save to chart.png."
        with q2:
            if st.button("🔍 Outliers (IQR)", use_container_width=True, help="Detect outliers across numeric columns using IQR, print their counts and explain anomalies"):
                selected_prompt = "Detect outliers across numeric columns using IQR, print their counts and explain anomalies."
        with q3:
            if st.button("🤖 RandomForest", use_container_width=True, help="Prepare features, split data with safe_train_test_split, train a RandomForest baseline model and evaluate performance"):
                selected_prompt = "Prepare features, split data with safe_train_test_split, train a RandomForest baseline model and evaluate performance."
        with q4:
            if st.button("📈 Key Drivers", use_container_width=True, help="Identify the top 3 features with highest predictive significance or correlation in this dataset"):
                selected_prompt = "Identify the top 3 features with highest predictive significance or correlation in this dataset."
        with q5:
            if st.button("🎯 Top 3 Features", use_container_width=True, help="Identify the top 3 features"):
                selected_prompt = "Identify the top 3 features"

        # Chat history display
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                avatar = "👤" if msg["role"] == "user" else "⚡"
                with st.chat_message(msg["role"], avatar=avatar):
                    clean_content = re.sub(r'\*\*ACTION:\*\*.*?```python.*?```', '', msg["content"], flags=re.DOTALL)
                    clean_content = re.sub(r'```python.*?```', '', clean_content, flags=re.DOTALL)
                    st.markdown(clean_content)
                    
                    if msg.get("code_output"):
                        st.code(msg["code_output"], language="python")
                        
                    if msg.get("image_bytes"):
                        if isinstance(msg["image_bytes"], list):
                            for img in msg["image_bytes"]:
                                st.image(img, use_container_width=True)
                        else:
                            st.image(msg["image_bytes"], use_container_width=True)

        user_typed_input = st.chat_input("Enter analytical objective (e.g., 'Plot scatter of Price vs Age', 'Train a classifier')...")
        active_query = selected_prompt or user_typed_input

        if active_query:
            with st.chat_message("user", avatar="👤"):
                st.markdown(active_query)

            with st.chat_message("assistant", avatar="⚡"):
                with st.status("🧠 Agent Formulating Strategy & Executing Code...", expanded=True) as status_box:
                    full_prompt = (
                        f"User Objective: {active_query}\n\n"
                        f"Current Profile:\n{st.session_state.profile}\n\n"
                        f"What is your next step?"
                    )
                    try:
                        status_box.update(label="Querying Neural Network...", state="running")
                        response = st.session_state.agent.chat(full_prompt)
                        
                        code_to_run = st.session_state.agent.extract_code(response)
                        img_bytes_list = []
                        output = None
                        success = True

                        if code_to_run:
                            status_box.update(label="Executing Sandboxed Python Code...", state="running")
                            st.session_state.df, output, success = st.session_state.sandbox.execute_code(st.session_state.df, code_to_run)
                            status_box.update(label="Execution Complete!", state="complete", expanded=False)
                            
                            clean_response = re.sub(r'\*\*ACTION:\*\*.*?```python.*?```', '', response, flags=re.DOTALL)
                            clean_response = re.sub(r'```python.*?```', '', clean_response, flags=re.DOTALL)
                            st.markdown(clean_response)
                            
                            if output:
                                if success:
                                    st.markdown("**Sandbox Output:**")
                                    st.code(output, language="python")
                                else:
                                    st.error(f"Sandbox Traceback:\n{output}")

                            # Intercept any generated PNG plots
                            png_files = glob.glob("*.png")
                            for img_file in png_files:
                                st.image(img_file, use_container_width=True)
                                with open(img_file, "rb") as f:
                                    img_bytes_list.append(f.read())
                                os.remove(img_file)

                            # Update profile
                            st.session_state.profile = json.dumps(DataQualityValidator.audit(st.session_state.df))
                        else:
                            status_box.update(label="Response Ready", state="complete", expanded=False)
                            st.markdown(response)

                        # Atomically append BOTH user and assistant message to state
                        st.session_state.messages.append({"role": "user", "content": active_query})
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "code_output": output,
                            "image_bytes": img_bytes_list
                        })
                    except Exception as err:
                        status_box.update(label="⚠️ Provider Alert", state="error", expanded=True)
                        err_str = str(err)
                        if "Connection refused" in err_str or "11434" in err_str:
                            st.error("🚨 Local Ollama server is offline (Connection refused on port 11434).")
                            st.info("💡 To fix: Switch the 'AI Engine Provider' in the left sidebar to **groq** (recommended cloud) or **google**, or run `ollama serve` in your terminal.")
                        else:
                            st.error(f"Execution Error: {err_str}")
                        st.session_state.messages.append({"role": "user", "content": active_query})
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"⚠️ Query encountered an error: `{err_str}`\n\n💡 Tip: If using Ollama, ensure `ollama serve` is running. Otherwise, switch to **groq** in the sidebar.",
                            "code_output": None,
                            "image_bytes": []
                        })

            st.rerun()
