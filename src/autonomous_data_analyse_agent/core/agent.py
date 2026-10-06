import os
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

class AutonomousDataAgent:
    def __init__(self, provider="google", model_name=None):
        self.provider = provider.lower()
        if model_name is None:
            if self.provider == "google":
                model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
            elif self.provider == "openai":
                model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            elif self.provider == "groq":
                model_name = os.getenv("GROQ_MODEL", "llama3-8b-8192")
            elif self.provider == "ollama":
                model_name = os.getenv("OLLAMA_MODEL", "llama3.1")
        self.llm = self._initialize_llm(self.provider, model_name)
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
            return ChatGoogleGenerativeAI(model=model_name, temperature=0.1)
        elif provider == "openai":
            return ChatOpenAI(model=model_name, temperature=0.1)
        elif provider == "groq":
            return ChatGroq(model=model_name, temperature=0.1)
        elif provider == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(model=model_name, temperature=0.1)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def chat(self, message: str) -> str:
        """Sends a message to the LLM and returns its response."""
        
        # 🐛 CRITICAL BUG FIX: Context Window Explosion
        # The frontend passes the full JSON profile in every message.
        # If we keep all of them in history, the context window will blow up exponentially.
        # We must prune the JSON profile from older messages before appending the new one.
        for i, msg in enumerate(self.conversation_history):
            if isinstance(msg, HumanMessage) and "Current Profile:" in msg.content:
                # Strip out the massive JSON payload from old history, keep the user objective
                import re
                pruned_content = re.sub(r'Current Profile:.*?What is your next step\?', '[PROFILE HIDDEN FOR BREVITY]', msg.content, flags=re.DOTALL)
                self.conversation_history[i] = HumanMessage(content=pruned_content)

        self.conversation_history.append(HumanMessage(content=message))
        
        # Ensure we don't exceed general bounds (keep last 10 messages + system prompt)
        if len(self.conversation_history) > 11:
            self.conversation_history = [self.conversation_history[0]] + self.conversation_history[-10:]
            
        response = self.llm.invoke(self.conversation_history)
        
        # Handle cases where response.content is a list of blocks
        content = response.content
        if isinstance(content, list):
            # Extract text from block structures
            content = "\n".join([block.get("text", "") if isinstance(block, dict) else str(block) for block in content])
            
        self.conversation_history.append(AIMessage(content=content))
        return content

    def extract_code(self, llm_response: str) -> str:
        """Extracts python code blocks from the LLM response."""
        match = re.search(r'```python\n(.*?)\n```', llm_response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
