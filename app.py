import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import glob
import time

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
import requests
from langchain_experimental.tools import PythonAstREPLTool
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

@st.cache_resource(ttl=300)
def check_ollama_health(base_url):
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
        return True
    except Exception:
        return False

def validate_result(question, answer, intermediate_steps):
    warnings = []
    lower_ans = str(answer).lower().strip()
    
    if lower_ans in ["0", "0.0", "empty", "none", "[]", "no data"] or lower_ans.startswith("0\n"):
        warnings.append("The result is zero or empty. Check if the filter criteria was too strict or misspelled.")
        
    if any(kw in question.lower() for kw in ["chart", "plot", "graph"]):
        saved_chart = False
        for action, obs in intermediate_steps:
            if "plt.savefig" in str(getattr(action, "tool_input", "")):
                saved_chart = True
                break
        if not saved_chart and "plt.savefig" not in lower_ans:
            warnings.append("You asked for a chart, but the agent did not appear to save one correctly.")
            
    if "failed to execute" in lower_ans:
        warnings.append("The agent encountered an error it could not recover from.")
        
    return warnings

def get_llm(api_key):
    if not api_key or not api_key.strip():
        ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.1")
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=ollama_model, temperature=0, base_url=base_url)
    return ChatGroq(model="openai/gpt-oss-120b", groq_api_key=api_key, temperature=0)

st.set_page_config(page_title="DataWhisperer", layout="wide", initial_sidebar_state="expanded")

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<div class="custom-header">
    <div>
        <div class="header-title">◉ DataWhisperer</div>
        <div class="header-subtitle">Talk to your data</div>
    </div>
    <div class="header-status">
        <div class="status-dot"></div>
        Ready
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<br><br><br>", unsafe_allow_html=True)
api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    value=os.environ.get("GROQ_API_KEY", ""),
    help="Get a free key at console.groq.com/keys",
)
st.sidebar.markdown("---")

uploaded = st.file_uploader("Drop your CSV here", type=["csv"])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded:
    df = pd.read_csv(uploaded)
    st.dataframe(df.head(10))

    # need this so the model stops guessing column names/values and gets them wrong
    def build_schema_summary(frame):
        lines = []
        for col in frame.columns:
            dtype = str(frame[col].dtype)
            nulls = int(frame[col].isnull().sum())
            nunique = frame[col].nunique()
            if pd.api.types.is_numeric_dtype(frame[col]):
                col_min = frame[col].min()
                col_max = frame[col].max()
                lines.append(f"- {col} ({dtype}, {nulls} nulls): numeric, range [{col_min}, {col_max}]")
            elif nunique <= 20:
                vals = frame[col].dropna().unique().tolist()
                lines.append(f"- {col} ({dtype}, {nulls} nulls): exact values = {vals}")
            else:
                sample = frame[col].dropna().unique()[:5].tolist()
                lines.append(f"- {col} ({dtype}, {nulls} nulls, {nunique} unique): sample values = {sample}")
        return "\n".join(lines)

    schema_summary = build_schema_summary(df)

    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    
    st.markdown(f"""
    <div class="glass-panel" style="padding: 1rem 1.5rem; margin-bottom: 2rem;">
        <div style="font-weight: 600; font-size: 1.1rem; margin-bottom: 0.2rem;">{uploaded.name}</div>
        <div style="color: rgba(255,255,255,0.6); font-size: 0.9rem;">{df.shape[0]:,} rows &nbsp;·&nbsp; {df.shape[1]} columns &nbsp;·&nbsp; Ready</div>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing", f"{missing_pct:.1f}%")

    st.sidebar.markdown("### DATASET")
    st.sidebar.caption(f"**{uploaded.name}**\n\n{df.shape[0]:,} rows\n\n{df.shape[1]} columns")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### DATA QUALITY")
    st.sidebar.caption(f"Missing: {missing_pct:.1f}%\n\nDuplicates: {df.duplicated().sum():,}")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### COLUMNS")
    st.sidebar.caption(" · ".join(df.columns.tolist()[:10]) + ("..." if len(df.columns) > 10 else ""))
    st.sidebar.markdown("---")
    st.sidebar.markdown("### SESSION")
    if st.sidebar.button("Clear chat"):
        st.session_state.chat_history = []
        st.rerun()

    llm = get_llm(api_key)
    is_ollama = isinstance(llm, ChatOllama)
    
    if is_ollama:
        ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.1")
        st.sidebar.caption(f"LLM: Ollama • {ollama_model}")
    else:
        st.sidebar.caption("LLM: Groq • llama-3.3-70b-versatile")

    # locking the repl down to just these 3 objects so it cant touch disk/files/internet
    safe_locals = {"df": df, "pd": pd, "plt": plt}
    repl_tool = PythonAstREPLTool(locals=safe_locals)
    repl_tool.name = "python_repl"
    repl_tool.description = (
        "A restricted Python execution environment with access ONLY to a pandas dataframe named df, "
        "pandas as pd, and matplotlib.pyplot as plt. Use it to run pandas code that "
        "answers questions about the data. To make a chart, build it with plt and "
        "save it using a unique filename (e.g. plt.savefig('chart_1.png')). Verify it "
        "is saved and not empty using os.path.exists and os.path.getsize instead of plt.show()."
    )
    tools = [repl_tool]

    prompt = PromptTemplate.from_template("""
You are a careful data analyst agent. A pandas dataframe called df is already loaded (do not reload or redefine it).
Answer the user's question by writing and running pandas/matplotlib code with the python_repl tool.
Always include the exact code you ran in your final answer, for transparency.

Dataset schema (trust this, do not guess or assume column names, dtypes, or spellings --
categorical values are shown EXACTLY as they appear in the data, including capitalization and
spacing, e.g. 'runout' vs 'run out' are NOT the same string, treat them exactly as listed):
{schema}

Rules you must follow:
1. Wrong aggregation scope: NEVER blindly count or sum a column without explicitly checking if it makes sense (e.g., if analyzing total runs, check if it's batter runs or total runs; if balls faced, distinguish legal deliveries from wides/no-balls). Use the schema to determine the correct columns.
2. Column hallucination: Before using df["column"], verify it exists exactly as spelled in the schema. NEVER silently invent or auto-correct a column name. If it doesn't exist, ask for clarification or explicitly state the assumption.
3. Filter before aggregation: ALWAYS apply filters/exclusions (e.g. filtered_df = df[condition]) BEFORE grouping or aggregating. DO NOT subtract values post-aggregation unless mathematically necessary.
4. NaN handling: Explicitly consider missing values (NaNs) in your code. Distinguish between .count() (ignores NaNs) and .size() (includes NaNs) when counting rows. Never let NaNs silently pass through.
5. Dtype mismatches: Inspect .dtypes before filtering or comparing. If a numeric comparison is needed on a text column, use pd.to_numeric(..., errors="coerce") instead of direct string-number comparison.
6. Explicit sorting: For ranking questions (top, bottom, highest, lowest, most, least), explicitly sort the result using .sort_values(ascending=True/False). Never rely on default ordering.
7. Empty-result validation: If a filter returns 0 rows, check actual values using .unique() or case-insensitive matching (.str.lower()) to see if it's a casing/formatting issue. DO NOT immediately answer 0 without investigating.
8. Execution timeout / iterations: If execution results in an error Observation, you have exactly 2 repair attempts. Identify the error, fix the code, and retry. If it still fails, your Final Answer MUST state "Failed to execute query due to error" (do NOT fabricate an answer).
9. Chart generation: When generating a chart, save it explicitly with a unique filename (e.g. plt.savefig('chart_1.png')). Do not use plt.show(). Verify it exists and is not empty (os.path.exists() and os.path.getsize() > 0) before your Final Answer.
10. Preserve original dataframe: NEVER overwrite or drop from the original df. Always assign filtered data to a new variable (e.g., working_df = df.copy()).
11. Cricket overs: For encoded overs (e.g. 5.1, 5.2), do NOT invent mathematical formulas. Use int(val) for the over number and validate actual dataset values before applying.
12. Ties: For "highest", "most", "maximum", return ALL tied rows. Do not arbitrarily pick one unless explicitly requested.
13. Conversation context (Follow-ups): When asked a follow-up (e.g. "show top 3", "what about team X?"), adapt the previous filtering/aggregation logic instead of starting from scratch. Maintain prior exclusions.
14. Concrete Final Answer: You must always give a concrete final answer based on actual tool Observations.

Conversation history:
{chat_history}

Tools: {tools}
Tool names: {tool_names}

Use this exact format:
Question: the input question
Thought: your reasoning
Action: one of [{tool_names}]
Action Input: the code to run
Observation: result of the code
... (Thought/Action/Action Input/Observation can repeat)
Thought: I now know the final answer
Final Answer: the answer to the user, including the code you ran

Question: {input}
{agent_scratchpad}
""")

    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True,
        handle_parsing_errors=True,
        max_iterations=15,
        max_execution_time=120,
        early_stopping_method="generate",
    )

    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn['q'])
        with st.chat_message("assistant"):
            if turn.get("provider"):
                st.caption(f"DataWhisperer · {turn['provider']}")
            else:
                st.caption("DataWhisperer")
            st.write(turn["a"])
            if turn.get("chart"):
                st.image(turn["chart"])

    suggested = None
    st.write("")
    st.caption("Suggested questions")
    sc1, sc2, sc3, sc4 = st.columns(4)
    if sc1.button("Top 5 products"): suggested = "Top 5 products"
    if sc2.button("Find anomalies"): suggested = "Find anomalies"
    if sc3.button("Show trends"): suggested = "Show trends"
    if sc4.button("Missing values"): suggested = "Missing values"
    
    question = st.chat_input("Ask anything about your data...") or suggested
    if question:
        for old_png in glob.glob("*.png"):
            os.remove(old_png)
        plt.close("all")

        history_text = "\n".join(
            f"Q: {t['q']}\nA: {t['a']}" for t in st.session_state.chat_history[-5:]
        ) or "None yet."

        with st.spinner("Thinking..."):
            result = None
            error_to_show = None
            provider_used = "Ollama" if is_ollama else "Groq"
            
            try:
                result = executor.invoke({"input": question, "chat_history": history_text, "schema": schema_summary})
            except Exception as e:
                if not is_ollama and ("rate_limit_exceeded" in str(e) or "429" in str(e)):
                    st.info("Groq rate limit hit — switched to local Ollama (llama3.1) for this response.")
                    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
                    if not check_ollama_health(base_url):
                        error_to_show = f"Ollama is not reachable at {base_url}. Start the Ollama app and make sure the configured model is available before using local fallback."
                    else:
                        try:
                            fallback_llm = ChatOllama(
                                model=os.environ.get("OLLAMA_MODEL", "llama3.1"), 
                                temperature=0, 
                                base_url=base_url
                            )
                            fallback_agent = create_react_agent(fallback_llm, tools, prompt)
                            fallback_executor = AgentExecutor(
                                agent=fallback_agent, tools=tools, verbose=True,
                                handle_parsing_errors=True,
                                max_iterations=15,
                                max_execution_time=120,
                                early_stopping_method="generate",
                            )
                            result = fallback_executor.invoke({"input": question, "chat_history": history_text, "schema": schema_summary})
                            provider_used = "Ollama (fallback)"
                        except Exception as fallback_e:
                            error_to_show = f"Error during Ollama fallback: {fallback_e}"
                else:
                    error_to_show = e

            if error_to_show is not None:
                if isinstance(error_to_show, Exception):
                    st.error(f"Error: {error_to_show}")
                    with st.expander("▸ Technical details"):
                        st.exception(error_to_show)
                else:
                    st.error(error_to_show)
            elif result is not None:
                answer = result["output"]
                intermediate = result.get("intermediate_steps", [])
                
                warnings = validate_result(question, answer, intermediate)
                
                st.markdown("### Answer")
                for w in warnings:
                    st.warning(f"⚠️ **Validation Warning:** {w}")
                st.write(answer)

                chart_path = "chart.png" if os.path.exists("chart.png") else None

                if chart_path is None:
                    pngs = glob.glob("*.png")
                    if pngs:
                        chart_path = max(pngs, key=os.path.getctime)

                # spent way too long figuring out charts werent showing - turned out sometimes
                # the agent builds the fig but never calls savefig at all. this grabs it anyway
                if chart_path is None and plt.get_fignums():
                    chart_path = "chart.png"
                    plt.savefig(chart_path, bbox_inches="tight")

                if chart_path:
                    st.image(chart_path)

                with st.expander("▸ Analysis details"):
                    for i, (action, observation) in enumerate(result.get("intermediate_steps", [])):
                        st.markdown(f"**Step {i+1} — Action:** `{action.tool}`")
                        st.code(action.tool_input, language="python")
                        st.markdown("**Observation:**")
                        st.code(str(observation))

                st.session_state.chat_history.append({
                    "q": question, 
                    "a": answer, 
                    "chart": chart_path,
                    "provider": provider_used
                })
else:
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">DataWhisperer</h1>
        <div class="hero-subtitle">
            Upload a CSV and ask questions in plain English. Get answers, tables, and beautiful visualizations.
        </div>
    </div>
    """, unsafe_allow_html=True)
