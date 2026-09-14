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
from langchain_core.callbacks import BaseCallbackHandler

THINKING_STEPS = [
    "Understanding your question",
    "Generating Pandas code",
    "Executing analysis",
    "Validating result",
    "Creating visualization",
]


class ThinkingCallback(BaseCallbackHandler):
    """Drives the live checklist shown while the agent works, using real
    LangChain lifecycle hooks instead of a fake timer."""

    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.stage = 0
        self.render()

    def render(self):
        lines = []
        for i, label in enumerate(THINKING_STEPS):
            if i < self.stage:
                mark = "✓"
            elif i == self.stage:
                mark = "●"
            else:
                mark = "○"
            lines.append(f"{mark} {label}")
        self.placeholder.markdown(
            "**DataWhisperer is analyzing...**\n\n" + "\n\n".join(lines)
        )

    def advance_to(self, stage):
        if stage > self.stage:
            self.stage = stage
            self.render()

    def on_llm_start(self, *args, **kwargs):
        self.advance_to(1)  # generating pandas code

    def on_agent_action(self, action, **kwargs):
        self.advance_to(2)  # executing analysis

    def on_tool_end(self, output, **kwargs):
        self.advance_to(3)  # validating result

    def on_agent_finish(self, finish, **kwargs):
        self.advance_to(3)

GROQ_MODEL = "qwen/qwen3.6-27b"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


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


def classify_groq_error(e):
    """Turn a raw exception into a (type, human message) pair so the user
    gets an accurate, specific error instead of one generic 'limit reached' message."""
    s = str(e).lower()
    if "401" in s or "invalid_api_key" in s or "invalid api key" in s:
        return "invalid_key", "Your Groq API key was rejected. It may be incorrect, revoked, or expired."
    if "429" in s or "rate_limit_exceeded" in s or "rate limit" in s:
        return "rate_limit", "Your Groq API key has hit its rate limit for now."
    if "insufficient_quota" in s or "quota" in s:
        return "quota", "Your Groq account has run out of quota."
    if "model_not_found" in s or "does not exist" in s:
        return "model", f"The model '{GROQ_MODEL}' isn't available on your Groq account."
    if "connection" in s or "timeout" in s or "network" in s:
        return "network", "Couldn't reach Groq's servers. This looks like a network issue."
    return "unknown", str(e)


def build_agent(llm, tools, prompt):
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent, tools=tools, verbose=True,
        handle_parsing_errors=True,
        max_iterations=15,
        max_execution_time=120,
        early_stopping_method="generate",
    )


st.set_page_config(page_title="DataWhisperer", layout="wide", initial_sidebar_state="expanded")

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "provider" not in st.session_state:
    st.session_state.provider = None          # None | "groq" | "ollama"
if "groq_key" not in st.session_state:
    st.session_state.groq_key = ""
if "groq_error" not in st.session_state:
    st.session_state.groq_error = None         # dict: {"type": ..., "msg": ...} or None

# ---------------------------------------------------------------------------
# Top header (always shown)
# ---------------------------------------------------------------------------
if st.session_state.provider == "groq":
    provider_label = f"● Groq &middot; {GROQ_MODEL}"
elif st.session_state.provider == "ollama":
    provider_label = f"● Ollama &middot; {OLLAMA_MODEL}"
else:
    provider_label = "No engine selected"

st.markdown(f"""
<div class="custom-header">
    <div>
        <div class="header-title">◉ DataWhisperer</div>
        <div class="header-subtitle">Talk to your data</div>
    </div>
    <div class="header-status">
        <div class="status-dot"></div>
        {provider_label}
    </div>
</div>
""", unsafe_allow_html=True)

if st.session_state.provider is not None:
    if st.sidebar.button("Change AI engine"):
        st.session_state.provider = None
        st.session_state.groq_error = None
        st.rerun()
    st.sidebar.markdown("---")

# ---------------------------------------------------------------------------
# STEP 1 — Explicit engine choice (never auto-selected, never auto-switched)
# ---------------------------------------------------------------------------
if st.session_state.provider is None:
    st.markdown("""
    <div style="text-align:center; margin: 2rem 0 2.5rem 0;">
        <h2 style="margin-bottom: 0.3rem;">Choose your AI engine</h2>
        <p style="color: rgba(232,232,236,0.6);">
            DataWhisperer uses an AI model to understand your questions and generate Pandas analysis.<br>
            This choice will not change automatically — you're always in control.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="glass-panel" style="padding: 1.5rem;">
            <h3>⚡ Groq</h3>
            <p style="color: rgba(232,232,236,0.6); font-size: 0.9rem;">
                Fast cloud inference. Requires an API key.<br><br>
                ✓ Very fast &nbsp;·&nbsp; ✓ Powerful models<br>
                ⚠ Subject to rate limits / quota
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Select Groq", use_container_width=True):
            st.session_state.provider = "groq"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="glass-panel" style="padding: 1.5rem;">
            <h3>◉ Ollama</h3>
            <p style="color: rgba(232,232,236,0.6); font-size: 0.9rem;">
                Runs locally on your machine.<br><br>
                ✓ Private &nbsp;·&nbsp; ✓ No API key needed<br>
                ⚠ Requires Ollama running locally
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Select Ollama", use_container_width=True):
            st.session_state.provider = "ollama"
            st.rerun()

    st.stop()

# ---------------------------------------------------------------------------
# STEP 2A — Groq key entry / re-entry (shown whenever key missing or rejected)
# ---------------------------------------------------------------------------
if st.session_state.provider == "groq" and (not st.session_state.groq_key or (st.session_state.groq_error and st.session_state.groq_error["type"] == "invalid_key")):
    st.markdown("### Connect Groq")
    if st.session_state.groq_error and st.session_state.groq_error["type"] == "invalid_key":
        st.error(st.session_state.groq_error["msg"])
    new_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Get a free key at console.groq.com/keys",
    )
    if st.button("Test & Continue"):
        if new_key.strip():
            st.session_state.groq_key = new_key.strip()
            st.session_state.groq_error = None
            st.rerun()
        else:
            st.warning("Enter a key first.")
    st.stop()

# ---------------------------------------------------------------------------
# STEP 2B — Ollama health check (shown whenever it's not reachable)
# ---------------------------------------------------------------------------
if st.session_state.provider == "ollama":
    if not check_ollama_health(OLLAMA_BASE_URL):
        st.markdown("### Use Ollama locally")
        st.error(f"⚠ Ollama isn't reachable at {OLLAMA_BASE_URL}. Start it and make sure '{OLLAMA_MODEL}' is pulled.")
        if st.button("Retry connection"):
            check_ollama_health.clear()
            st.rerun()
        st.stop()

# ---------------------------------------------------------------------------
# STEP 3 — Groq error resolution screen (only reachable mid-conversation,
# after an actual failed call). Never auto-switches providers.
# ---------------------------------------------------------------------------
if st.session_state.groq_error is not None and st.session_state.groq_error["type"] != "invalid_key":
    err = st.session_state.groq_error
    title = {
        "rate_limit": "⚠ Groq limit reached",
        "quota": "⚠ Groq quota exhausted",
        "network": "⚠ Network error reaching Groq",
        "model": "⚠ Groq model unavailable",
        "unknown": "⚠ Groq error",
    }.get(err["type"], "⚠ Groq error")

    st.markdown(f"### {title}")
    st.warning(err["msg"])
    st.caption("Your dataset and previous analyses are safe.")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🔑 Enter a new Groq key"):
            st.session_state.groq_key = ""
            st.session_state.groq_error = {"type": "invalid_key", "msg": "Enter a new key to continue with Groq."}
            st.rerun()
    with c2:
        if st.button("◉ Switch to Ollama"):
            st.session_state.provider = "ollama"
            st.session_state.groq_error = None
            st.rerun()
    with c3:
        if err["type"] in ("network", "unknown") and st.button("↻ Retry"):
            st.session_state.groq_error = None
            st.rerun()
    st.stop()

# ---------------------------------------------------------------------------
# From here on, the provider is confirmed ready. Build the LLM.
# ---------------------------------------------------------------------------
if st.session_state.provider == "groq":
    llm = ChatGroq(model=GROQ_MODEL, groq_api_key=st.session_state.groq_key, temperature=0)
    is_ollama = False
else:
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0, base_url=OLLAMA_BASE_URL)
    is_ollama = True

uploaded = st.file_uploader("Drop your CSV here", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

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
    <div class="glass-panel" style="padding: 1rem 1.5rem; margin-bottom: 1.5rem;">
        <div style="font-weight: 600; font-size: 1.1rem; margin-bottom: 0.2rem;">{uploaded.name}</div>
        <div style="color: rgba(255,255,255,0.6); font-size: 0.9rem;">{df.shape[0]:,} rows &nbsp;·&nbsp; {df.shape[1]} columns &nbsp;·&nbsp; Ready</div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("### WORKSPACE")
    if "page" not in st.session_state:
        st.session_state.page = "Ask Data"
    st.session_state.page = st.sidebar.radio(
        "nav", ["Ask Data", "Overview", "Visualizations", "Data Explorer"],
        label_visibility="collapsed",
        index=["Ask Data", "Overview", "Visualizations", "Data Explorer"].index(st.session_state.page),
    )
    st.sidebar.markdown("---")

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

    if st.session_state.page == "Overview":
        st.markdown("## Dataset Overview")
        oc1, oc2, oc3, oc4 = st.columns(4)
        oc1.metric("Rows", f"{df.shape[0]:,}")
        oc2.metric("Columns", df.shape[1])
        oc3.metric("Missing", f"{missing_pct:.1f}%")
        oc4.metric("Duplicates", f"{df.duplicated().sum():,}")

        st.markdown("#### Dataset Health")
        dup_count = df.duplicated().sum()
        st.markdown(
            "\u2713 Dataset loaded  \n\u2713 Types detected  \n" +
            ("\u2713 No duplicate rows" if dup_count == 0 else f"\u26a0 {dup_count} duplicate rows found") +
            "  \n\u2713 Missing values analyzed"
        )

        st.markdown("#### Column Overview")
        numeric_cols = set(df.select_dtypes(include="number").columns)
        text_cols = set(df.select_dtypes(include="object").columns)
        for col in df.columns:
            kind = "Numeric" if col in numeric_cols else ("Text" if col in text_cols else "Other")
            st.caption(f"**{col}** \u2014 {kind}")

        st.markdown("#### Preview")
        st.dataframe(df.head(20))

    elif st.session_state.page == "Visualizations":
        st.markdown("## Visualizations")
        charts = [t for t in st.session_state.chat_history if t.get("chart")]
        if not charts:
            st.caption("No charts generated yet \u2014 ask a question that requests a chart from the Ask Data page.")
        else:
            vcols = st.columns(2)
            for i, turn in enumerate(charts):
                with vcols[i % 2]:
                    st.markdown(f"**{turn['q']}**")
                    if turn["chart"] and os.path.exists(turn["chart"]):
                        st.image(turn["chart"])
                        with open(turn["chart"], "rb") as cf:
                            st.download_button("Download PNG", cf, file_name=os.path.basename(turn["chart"]), key=f"dl_{i}")
                    else:
                        st.caption("(chart file no longer available)")

    elif st.session_state.page == "Data Explorer":
        st.markdown("## Data Explorer")
        search = st.text_input("Search across all columns", "")
        display_df = df
        if search.strip():
            mask = df.astype(str).apply(lambda row: row.str.contains(search, case=False, na=False)).any(axis=1)
            display_df = df[mask]
        st.caption(f"Showing {len(display_df):,} of {len(df):,} rows")
        st.dataframe(display_df, height=500)

    else:  # Ask Data
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

        executor = build_agent(llm, tools, prompt)

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
            plt.close("all")

            history_text = "\n".join(
                f"Q: {t['q']}\nA: {t['a']}" for t in st.session_state.chat_history[-5:]
            ) or "None yet."

            thinking_box = st.empty()
            callback = ThinkingCallback(thinking_box)

            result = None
            provider_used = "Ollama" if is_ollama else "Groq"

            try:
                result = executor.invoke(
                    {"input": question, "chat_history": history_text, "schema": schema_summary},
                    config={"callbacks": [callback]},
                )
            except Exception as e:
                thinking_box.empty()
                if is_ollama:
                    # Ollama failures are shown inline -- there's no "other provider" to fall back to
                    # since the user explicitly chose Ollama.
                    st.error(f"Error running local Ollama model: {e}")
                    with st.expander("▸ Technical details"):
                        st.exception(e)
                    result = None
                else:
                    err_type, err_msg = classify_groq_error(e)
                    st.session_state.groq_error = {"type": err_type, "msg": err_msg}
                    st.rerun()

            if result is not None:
                answer = result["output"]
                intermediate = result.get("intermediate_steps", [])

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
                    unique_path = f"chart_{len(st.session_state.chat_history)}.png"
                    if chart_path != unique_path:
                        try:
                            os.replace(chart_path, unique_path)
                        except OSError:
                            pass
                        else:
                            chart_path = unique_path

                callback.advance_to(4)  # creating visualization (resolved by now either way)
                time.sleep(0.3)  # let the final checkmark actually be visible before it disappears
                thinking_box.empty()

                warnings = validate_result(question, answer, chart_saved=bool(chart_path))

                st.markdown("### Answer")
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
        <h1 class="hero-title">DataWhisperer</h1>
        <div class="hero-subtitle">
            Upload a CSV and ask questions in plain English. Get answers, tables, and beautiful visualizations.
        </div>
    </div>
    """, unsafe_allow_html=True)
