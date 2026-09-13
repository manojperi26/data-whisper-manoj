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

def validate_result(question, answer, chart_saved):
    warnings = []
    lower_ans = str(answer).lower().strip()
    
    if lower_ans in ["0", "0.0", "empty", "none", "[]", "no data"] or lower_ans.startswith("0\n"):
        warnings.append("The result is zero or empty. Check if the filter criteria was too strict or misspelled.")
        
    if any(kw in question.lower() for kw in ["chart", "plot", "graph"]) and not chart_saved:
        warnings.append("You asked for a chart, but the agent did not appear to save one correctly.")
            
    if "failed to execute" in lower_ans:
        warnings.append("The agent encountered an error it could not recover from.")
        
    return warnings

def get_llm(api_key):
    if not api_key or not api_key.strip():
        ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.1")
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=ollama_model, temperature=0, base_url=base_url)
    return ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=api_key, temperature=0)

st.set_page_config(page_title="DataWhisperer", layout="wide", initial_sidebar_state="expanded")

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = None
if "provider_error" not in st.session_state:
    st.session_state.provider_error = False

st.markdown("""
<div class="custom-header">
    <div>
        <div class="header-title"><span class="brand-mark">◒</span> DataWhisperer</div>
        <div class="header-subtitle">Talk to your data</div>
    </div>
    <div class="header-status">
        <div class="status-dot"></div>
        Ready
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<div class='sidebar-brand'><span class='brand-mark'>◒</span><span>DataWhisperer</span></div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-label'>CONNECTION</div>", unsafe_allow_html=True)
api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    value=os.environ.get("GROQ_API_KEY", ""),
    help="Get a free key at console.groq.com/keys",
)
if st.session_state.llm_provider is None:
    st.markdown("""
    <div class="welcome-shell">
        <div class="eyebrow">YOUR PRIVATE DATA ANALYST</div>
        <h1>Meet your data<br><span>in a new light.</span></h1>
        <p>Choose how DataWhisper should reason over your CSV. Your choice stays active for this session.</p>
    </div>
    """, unsafe_allow_html=True)
    provider_choice = st.radio("Choose an LLM provider", ["Groq API", "Local Ollama"], index=None, horizontal=True, key="provider_choice")
    if provider_choice:
        st.session_state.llm_provider = provider_choice
        st.rerun()
    st.stop()

st.sidebar.markdown("<div class='sidebar-label'>MODEL</div>", unsafe_allow_html=True)
provider_choice = st.sidebar.radio("Provider", ["Groq API", "Local Ollama"], index=0 if st.session_state.llm_provider == "Groq API" else 1, label_visibility="collapsed")
if provider_choice != st.session_state.llm_provider:
    st.session_state.llm_provider = provider_choice
    st.session_state.provider_error = False
    st.rerun()
st.sidebar.markdown("<div class='sidebar-rule'></div>", unsafe_allow_html=True)

st.markdown("<div class='section-kicker'>WORKSPACE</div>", unsafe_allow_html=True)
uploaded = st.file_uploader("Drop your CSV here", type=["csv"])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded:
    df = pd.read_csv(uploaded)
    with st.expander("Preview dataset", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)

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
    
    duplicate_count = int(df.duplicated().sum())
    st.markdown("<div class='section-kicker'>DATASET OVERVIEW</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for column, label, value, detail in [(c1, "Rows", f"{df.shape[0]:,}", "records loaded"), (c2, "Columns", f"{df.shape[1]}", "fields detected"), (c3, "Missing data", f"{missing_pct:.1f}%", "of all values"), (c4, "Duplicates", f"{duplicate_count:,}", "repeated rows")]:
        column.markdown(f"<div class='stat-card'><div class='stat-label'>{label}</div><div class='stat-value'>{value}</div><div class='stat-detail'>{detail}</div></div>", unsafe_allow_html=True)

    st.sidebar.markdown(f"<div class='side-dataset'><div class='side-file'>{uploaded.name}</div><div>{df.shape[0]:,} rows · {df.shape[1]} columns</div></div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='sidebar-label'>DATA HEALTH</div>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<div class='side-health'><span class='health-dot'></span>{missing_pct:.1f}% missing<br><span class='health-dot'></span>{duplicate_count:,} duplicates</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='sidebar-label'>COLUMNS</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='side-columns'>" + "".join(f"<span>{col}</span>" for col in df.columns.tolist()[:12]) + ("<span>...</span>" if len(df.columns) > 12 else "") + "</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='sidebar-label'>SESSION</div>", unsafe_allow_html=True)
    if st.sidebar.button("Clear chat"):
        st.session_state.chat_history = []
        st.rerun()

    llm = get_llm(api_key if st.session_state.llm_provider == "Groq API" else "")
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
                st.caption(f"DataWhisper · {turn['provider']}")
            else:
                st.caption("DataWhisper")
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
    if sc4.button("Find missing values"): suggested = "Find missing values"
    
    question = st.chat_input("Ask anything about your data...") or suggested
    if question:
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
                if not is_ollama:
                    st.session_state.provider_error = True
                error_to_show = e

            if error_to_show is not None:
                if isinstance(error_to_show, Exception):
                    st.error(f"Error: {error_to_show}")
                    with st.expander("▸ Technical details"):
                        st.exception(error_to_show)
                else:
                    st.error(error_to_show)
                if st.session_state.provider_error:
                    st.warning("Groq API is unavailable. Would you like to switch to Local Ollama?")
                    switch_col, stay_col = st.columns(2)
                    if switch_col.button("Switch to Ollama", type="primary"):
                        st.session_state.llm_provider = "Local Ollama"
                        st.session_state.provider_error = False
                        st.rerun()
                    if stay_col.button("Stay with Groq"):
                        st.session_state.provider_error = False
                        st.rerun()
            elif result is not None:
                answer = result["output"]
                intermediate = result.get("intermediate_steps", [])

                chart_path = "chart.png" if os.path.exists("chart.png") else None

                if chart_path is None:
                    pngs = glob.glob("*.png")
                    if pngs:
                        chart_path = max(pngs, key=os.path.getctime)

                if chart_path is None and plt.get_fignums():
                    chart_path = "chart.png"
                    plt.savefig(chart_path, bbox_inches="tight")

                if chart_path:
                    unique_path = f"chart_{len(st.session_state.chat_history)}.png"
                    if chart_path != unique_path:
                        try:
                            os.replace(chart_path, unique_path)
                        except OSError:
                            pass
                        else:
                            chart_path = unique_path

                warnings = validate_result(question, answer, chart_saved=bool(chart_path))

                st.markdown("<div class='answer-heading'><span class='assistant-avatar'>◒</span><span>DataWhisper</span><span class='answer-label'>ANSWER</span></div>", unsafe_allow_html=True)
                for w in warnings:
                    st.warning(f"⚠️ **Validation Warning:** {w}")
                st.write(answer)

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
        <div class="eyebrow">CSV INTELLIGENCE, WITHOUT THE FRICTION</div>
        <h1 class="hero-title">Ask better questions<br><span>of your data.</span></h1>
        <div class="hero-subtitle">
            Upload a CSV and let DataWhisper turn rows into clear answers, useful trends, and beautiful visualizations.
        </div>
    </div>
    """, unsafe_allow_html=True)
