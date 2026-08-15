import pandas as pd
import pytest
import os
import glob
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_experimental.tools import PythonAstREPLTool
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
import matplotlib.pyplot as plt

PROMPT_TEXT = """You are a careful data analyst agent. A pandas dataframe called df is already loaded (do not reload or redefine it).
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
{agent_scratchpad}"""

def build_schema_summary(frame):
    lines = []
    for col in frame.columns:
        dtype = str(frame[col].dtype)
        nulls = int(frame[col].isnull().sum())
        nunique = frame[col].nunique()
        if pd.api.types.is_numeric_dtype(frame[col]):
            lines.append(f"- {col} ({dtype}, {nulls} nulls): numeric, range [{frame[col].min()}, {frame[col].max()}]")
        elif nunique <= 20:
            lines.append(f"- {col} ({dtype}, {nulls} nulls): exact values = {frame[col].dropna().unique().tolist()}")
        else:
            lines.append(f"- {col} ({dtype}, {nulls} nulls, {nunique} unique): sample values = {frame[col].dropna().unique()[:5].tolist()}")
    return "\n".join(lines)

def create_test_executor(df):
    api_key = os.environ.get("GROQ_API_KEY", "")
    if api_key:
        llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=api_key, temperature=0)
    else:
        llm = ChatOllama(model=os.environ.get("OLLAMA_MODEL", "llama3.1"), temperature=0, base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))

    safe_locals = {"df": df, "pd": pd, "plt": plt}
    repl_tool = PythonAstREPLTool(locals=safe_locals)
    repl_tool.name = "python_repl"
    repl_tool.description = (
        "A sandboxed Python shell with access ONLY to a pandas dataframe named df, "
        "pandas as pd, and matplotlib.pyplot as plt. Use it to run pandas code that "
        "answers questions about the data. To make a chart, build it with plt and "
        "save it using a unique filename (e.g. plt.savefig('chart_1.png')). Verify it "
        "is saved and not empty using os.path.exists and os.path.getsize instead of plt.show()."
    )
    
    prompt = PromptTemplate.from_template(PROMPT_TEXT)
    agent = create_react_agent(llm, [repl_tool], prompt)
    return AgentExecutor(
        agent=agent, tools=[repl_tool], verbose=True,
        handle_parsing_errors=True,
        max_iterations=15,
        max_execution_time=120,
        early_stopping_method="generate",
    )

def extract_tool_inputs(result):
    return [action.tool_input for action, _ in result.get("intermediate_steps", [])]

# Tests

def test_1_column_hallucination():
    df = pd.DataFrame({"real_col": [1, 2, 3]})
    executor = create_test_executor(df)
    schema = build_schema_summary(df)
    res = executor.invoke({"input": "What is the average of fake_col?", "chat_history": "", "schema": schema})
    assert "fake_col" not in df.columns
    ans = str(res["output"]).lower()
    assert "not exist" in ans or "schema" in ans or "clarification" in ans or "error" in ans

def test_2_numeric_string():
    df = pd.DataFrame({"val": ["20", "10", "30"]})
    executor = create_test_executor(df)
    schema = build_schema_summary(df)
    res = executor.invoke({"input": "What is the max val?", "chat_history": "", "schema": schema})
    inputs = "".join(extract_tool_inputs(res))
    assert "pd.to_numeric" in inputs or "astype" in inputs
    assert "30" in str(res["output"])

def test_3_case_mismatch():
    df = pd.DataFrame({"Country": ["India", "USA"]})
    executor = create_test_executor(df)
    schema = build_schema_summary(df)
    res = executor.invoke({"input": "How many for india?", "chat_history": "", "schema": schema})
    assert "1" in str(res["output"])

def test_4_filter_before_aggregation():
    df = pd.DataFrame({"Team": ["A", "A", "B"], "Score": [10, 20, 30], "Type": ["valid", "invalid", "valid"]})
    executor = create_test_executor(df)
    schema = build_schema_summary(df)
    res = executor.invoke({"input": "Total score for team A excluding invalid types?", "chat_history": "", "schema": schema})
    assert "10" in str(res["output"])

def test_5_nan_handling():
    import numpy as np
    df = pd.DataFrame({"A": [1, 2, np.nan, 4]})
    executor = create_test_executor(df)
    schema = build_schema_summary(df)
    res = executor.invoke({"input": "How many nulls are in column A?", "chat_history": "", "schema": schema})
    assert "1" in str(res["output"])

def test_6_explicit_sorting():
    df = pd.DataFrame({"Name": ["X", "Y", "Z"], "Val": [10, 30, 20]})
    executor = create_test_executor(df)
    schema = build_schema_summary(df)
    res = executor.invoke({"input": "Top 2 by Val", "chat_history": "", "schema": schema})
    inputs = "".join(extract_tool_inputs(res))
    assert "sort_values" in inputs

def test_7_tie_handling():
    df = pd.DataFrame({"Name": ["A", "B", "C"], "Score": [100, 100, 90]})
    executor = create_test_executor(df)
    schema = build_schema_summary(df)
    res = executor.invoke({"input": "Who has the highest score?", "chat_history": "", "schema": schema})
    ans = str(res["output"])
    assert "A" in ans and "B" in ans

def test_8_preserve_original_df():
    df = pd.DataFrame({"A": [1, 2, 3]})
    original_shape = df.shape
    executor = create_test_executor(df)
    schema = build_schema_summary(df)
    executor.invoke({"input": "Drop rows where A is 2 and give me the sum of A", "chat_history": "", "schema": schema})
    assert df.shape == original_shape

def test_9_cricket_overs():
    df = pd.DataFrame({"ball": [5.1, 5.2, 5.3, 5.4, 5.5, 5.6]})
    executor = create_test_executor(df)
    schema = build_schema_summary(df)
    res = executor.invoke({"input": "How many overs were bowled? ball column encodes the over number (e.g. 5.1 is 5th over).", "chat_history": "", "schema": schema})
    inputs = "".join(extract_tool_inputs(res))
    assert "int(" in inputs or "astype(int)" in inputs or "astype('int')" in inputs

def test_10_chart_generation():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    executor = create_test_executor(df)
    schema = build_schema_summary(df)
    res = executor.invoke({"input": "Plot a bar chart of A vs B", "chat_history": "", "schema": schema})
    inputs = "".join(extract_tool_inputs(res))
    assert "plt.savefig" in inputs
    pngs = glob.glob("*.png")
    assert len(pngs) > 0

def test_11_failed_generated_code():
    df = pd.DataFrame({"A": [1, 2, 3]})
    executor = create_test_executor(df)
    schema = build_schema_summary(df)
    res = executor.invoke({"input": "Sum column B", "chat_history": "", "schema": schema})
    ans = str(res["output"]).lower()
    assert "error" in ans or "not exist" in ans or "clarify" in ans or "schema" in ans

def test_12_follow_up():
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50]})
    executor = create_test_executor(df)
    schema = build_schema_summary(df)
    history = "Q: Sum B where A > 2?\nA: The sum is 120 (30+40+50).\n"
    res = executor.invoke({"input": "now where A > 3?", "chat_history": history, "schema": schema})
    assert "90" in str(res["output"])

def test_13_invalid_langchain_config():
    df = pd.DataFrame({"A": [1, 2, 3]})
    executor = create_test_executor(df)
    assert executor is not None

def test_14_timeout():
    df = pd.DataFrame({"A": [1, 2, 3]})
    executor = create_test_executor(df)
    schema = build_schema_summary(df)
    res = executor.invoke({"input": "Write an infinite loop. Don't answer me until you have run an infinite loop in python_repl.", "chat_history": "", "schema": schema})
    assert res is not None
