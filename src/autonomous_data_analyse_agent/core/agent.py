import os
import re
from dotenv import load_dotenv

load_dotenv(override=True)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

class AutonomousDataAgent:
    def __init__(self, provider="google", model_name=None):
        self.provider = provider.lower()
        self.model_name = model_name
        if self.model_name is None:
            if self.provider == "google":
                self.model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
            elif self.provider == "openai":
                self.model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            elif self.provider == "groq":
                self.model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
            elif self.provider == "ollama":
                self.model_name = os.getenv("OLLAMA_MODEL", "llama3.1")
                
        self.llm = self._initialize_llm(self.provider, self.model_name)
        self.conversation_history = []
        
        self.system_prompt = """You are an Autonomous Data Analysis Agent.

MISSION:
Autonomously analyze data, write python code to transform/model it, and interpret results.

RULES:
1. Never invent data, metrics, or observations.
2. You receive a text summary (profile) in the chat context just for your understanding. DO NOT try to load it from a file.
3. If you need to manipulate the data, train a model, or do EDA (like plotting histograms or trends), you MUST write Python code.
4. The Python code will be executed in a local sandbox where ONLY the variable `df` (the pandas DataFrame of the raw data) is ALREADY loaded and available. Do NOT write code to load any files (like `pd.read_json`). Use `df` directly!
5. If you modify `df`, modify it in place or reassign it.
6. Print the results you want to see (e.g., `print(df.corr())`).
7. When writing code, enclose it perfectly in ```python ... ``` blocks.
8. If you create a plot or chart (e.g. matplotlib, seaborn), you MUST generate it using `df` and save it as 'chart.png' (e.g. `plt.savefig('chart.png')`). NEVER use `plt.show()`.
9. If no further code execution is needed and the objective is met, provide a final REPORT.

Format your response as:
**THOUGHTS:**
<Your reasoning here>

**ACTION:**
```python
# your code here
```
(Omit the ACTION section if you are done and just writing the final REPORT).
"""
        self.conversation_history.append(SystemMessage(content=self.system_prompt))

    def _initialize_llm(self, provider, model_name):
        if provider == "google":
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.1)
        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            return ChatOpenAI(model=model_name, api_key=api_key, temperature=0.1)
        elif provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            return ChatGroq(model=model_name, api_key=api_key, temperature=0.1)
        elif provider == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(model=model_name, temperature=0.1)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _fallback_invoke(self, conversation):
        """Attempts to invoke with fallback models or providers if primary encounters capacity errors."""
        fallback_models = []
        if self.provider == "google":
            fallback_models = ["gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-flash-latest", "gemini-3.8-flash"]
        elif self.provider == "groq":
            fallback_models = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]

        last_error = None
        for m in fallback_models:
            try:
                temp_llm = self._initialize_llm(self.provider, m)
                res = temp_llm.invoke(conversation)
                self.model_name = m
                self.llm = temp_llm
                return res
            except Exception as e:
                last_error = e
                continue

        # If primary provider fails (e.g. Ollama connection refused, OpenAI credit exhausted), failover to cloud providers
        if os.getenv("GROQ_API_KEY") and self.provider != "groq":
            try:
                temp_llm = ChatGroq(model="openai/gpt-oss-120b", api_key=os.getenv("GROQ_API_KEY"), temperature=0.1)
                return temp_llm.invoke(conversation)
            except Exception:
                pass
        if os.getenv("GOOGLE_API_KEY") and self.provider != "google":
            try:
                temp_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0.1)
                return temp_llm.invoke(conversation)
            except Exception:
                pass

        if last_error:
            raise last_error
        raise RuntimeError("No LLM fallback succeeded.")

    def _generate_offline_response(self, prompt: str) -> str:
        """Autonomous heuristic fallback if external network/LLM APIs are exhausted."""
        prompt_lower = prompt.lower()
        if "histogram" in prompt_lower or "distribution" in prompt_lower:
            return """**THOUGHTS:**
I will plot histograms with distribution curves for all numeric columns in `df` and save the visualization to 'chart.png'.

**ACTION:**
```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('dark_background')
numeric_cols = df.select_dtypes(include='number').columns.tolist()

if numeric_cols:
    n_cols = len(numeric_cols)
    nrows = (n_cols + 1) // 2
    ncols = 2 if n_cols > 1 else 1

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows), facecolor='#0B0F19')
    if n_cols == 1:
        axes_list = [axes]
    else:
        axes_list = axes.flatten()

    for idx, col in enumerate(numeric_cols):
        ax = axes_list[idx]
        ax.set_facecolor('#0B0F19')
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color='#38BDF8', edgecolor='#1F2937')
        ax.set_title(f"Distribution: {col}", color='#FFFFFF', fontweight='bold')
        ax.set_xlabel(col, color='#94A3B8')
        ax.set_ylabel("Density", color='#94A3B8')

    for j in range(len(numeric_cols), len(axes_list)):
        fig.delaxes(axes_list[j])

    plt.tight_layout()
    plt.savefig('chart.png', dpi=200, facecolor='#0B0F19', bbox_inches='tight')
    plt.close()
    print(f"Successfully rendered distributions for {len(numeric_cols)} numeric features: {numeric_cols}")
else:
    print("No numeric features found in the current dataset.")
```"""

        elif "outlier" in prompt_lower or "iqr" in prompt_lower:
            return """**THOUGHTS:**
I will scan all numeric columns using the Interquartile Range (IQR) method (1.5 * IQR) to detect outliers and print their exact counts and boundaries.

**ACTION:**
```python
numeric_cols = df.select_dtypes(include='number').columns

outlier_summary = {}
print("=== OUTLIER DETECTION REPORT (IQR METHOD) ===")
for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    count = len(outliers)
    pct = (count / len(df)) * 100
    outlier_summary[col] = count
    print(f"Feature: {col:<18} | Bounds: [{lower_bound:>8.2f}, {upper_bound:>8.2f}] | Outliers: {count:>3} ({pct:.1f}%)")

total_outliers = sum(outlier_summary.values())
print(f"\nTotal Outlier Data Points Identified: {total_outliers}")
```"""

        elif "model" in prompt_lower or "randomforest" in prompt_lower or "train" in prompt_lower:
            return """**THOUGHTS:**
I will prepare the dataset, identify target and features, perform safe train-test splitting, train a RandomForest baseline model, and evaluate accuracy and metrics.

**ACTION:**
```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score

# Identify target column: Prefer 'Churn', 'Survived', or last numeric column
target_candidates = ['Churn', 'Survived', 'Repeat_Customer', 'target']
target = None
for c in target_candidates:
    if c in df.columns:
        target = c
        break
if target is None:
    target = df.columns[-1]

print(f"Selected Target Variable: '{target}'")

# Split features & target
X = df.drop(columns=[target])
y = df[target]

# Keep only numeric or encode basic categorical
X_numeric = pd.get_dummies(X, drop_first=True)
# Fill missing
X_numeric = X_numeric.fillna(X_numeric.median(numeric_only=True))

is_classification = y.nunique() <= 10 and (y.dtype == 'object' or pd.api.types.is_integer_dtype(y))

X_train, X_test, y_train, y_test = safe_train_test_split(X_numeric, y, test_size=0.2, random_state=42)

if is_classification:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"RandomForestClassifier Accuracy: {acc*100:.2f}%")
else:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    print(f"RandomForestRegressor R2 Score: {r2:.4f}")

# Top 5 Feature Importances
importances = pd.Series(model.feature_importances_, index=X_numeric.columns).sort_values(ascending=False).head(5)
print("\nTop 5 Predictive Features:")
for feat, imp in importances.items():
    print(f" - {feat}: {imp:.4f}")
```"""

        elif "top" in prompt_lower or "feature" in prompt_lower or "correlation" in prompt_lower:
            return """**THOUGHTS:**
I will analyze the numerical features to find the top 3 most influential features based on variance and pairwise correlations.

**ACTION:**
```python
numeric_df = df.select_dtypes(include='number')
if numeric_df.shape[1] >= 2:
    corr = numeric_df.corr().abs()
    unstacked = corr.unstack().sort_values(ascending=False).drop_duplicates()
    top_pairs = unstacked[unstacked < 1.0].head(3)
    print("=== TOP 3 HIGHLY CORRELATED FEATURE PAIRS ===")
    for (f1, f2), val in top_pairs.items():
        print(f" 1. {f1} <---> {f2}: Correlation = {val:.4f}")
else:
    print("Top variance features:")
    print(numeric_df.var().sort_values(ascending=False).head(3))
```"""

        else:
            return f"""**THOUGHTS:**
I have examined the data structure and objectives. I will print the core descriptive statistics and column health metrics for `df`.

**ACTION:**
```python
print("Shape:", df.shape)
print("\nSummary Statistics:")
print(df.describe().T[['mean', 'std', 'min', '50%', 'max']])
```"""

    def chat(self, message: str) -> str:
        """Sends a message to the LLM and returns its response with automatic fallback."""
        for i, msg in enumerate(self.conversation_history):
            if isinstance(msg, HumanMessage) and "Current Profile:" in msg.content:
                pruned_content = re.sub(r'Current Profile:.*?What is your next step\?', '[PROFILE HIDDEN FOR BREVITY]', msg.content, flags=re.DOTALL)
                self.conversation_history[i] = HumanMessage(content=pruned_content)

        self.conversation_history.append(HumanMessage(content=message))
        
        if len(self.conversation_history) > 11:
            self.conversation_history = [self.conversation_history[0]] + self.conversation_history[-10:]
            
        try:
            response = self.llm.invoke(self.conversation_history)
        except Exception:
            try:
                response = self._fallback_invoke(self.conversation_history)
            except Exception:
                # If network/API is completely blocked, use resilient offline engine
                fallback_content = self._generate_offline_response(message)
                self.conversation_history.append(AIMessage(content=fallback_content))
                return fallback_content

        content = response.content
        if isinstance(content, list):
            content = "\n".join([block.get("text", "") if isinstance(block, dict) else str(block) for block in content])
            
        self.conversation_history.append(AIMessage(content=content))
        return content

    def extract_code(self, llm_response: str) -> str:
        """Extracts python code blocks from the LLM response."""
        match = re.search(r'```(?:python)?\s*\n(.*?)\n```', llm_response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
