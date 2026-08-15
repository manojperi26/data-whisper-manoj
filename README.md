# 📊 DataWhisperer

Talk to your CSV in plain English. Upload a file, ask a question, get an answer, a table, or a chart — powered by a LangChain ReAct agent that writes and runs real pandas/matplotlib code in a sandbox.

## How it works

- You upload a CSV and enter a Groq API key.
- The app profiles the data (rows, columns, missing values, dtypes) and builds a schema summary so the agent never guesses column names or values.
- A ReAct agent (Llama 3.3 70B via Groq) gets your question + schema + recent chat history, then writes pandas code and runs it in a locked-down Python REPL (only `df`, `pd`, `plt` are exposed — no file/network access).
- Charts are saved as `chart.png` and shown inline; the agent's full thought/action/observation trace is available in a debug expander.

## Features

- Natural-language Q&A over any CSV
- Auto-generated data profile (shape, nulls, dtypes, quick stats)
- Chart generation (agent calls matplotlib directly)
- Follow-up questions using chat history ("now just show me the top 3")
- Guardrails baked into the prompt: no guessed column names, explicit filtering before aggregation, NaN handling, tie handling for top/bottom queries, sanity checks before the final answer
- Sandbox execution — the REPL can't touch disk, network, or anything outside the loaded dataframe

## LLM Fallback

DataWhisperer uses Groq (`llama-3.3-70b-versatile`) as the primary LLM and automatically falls back to local Ollama if the Groq free-tier rate limit is hit or no Groq API key is provided. Local fallback requires `ollama pull llama3.1` and the Ollama app running locally.

Optional environment variables for Ollama configuration:
```text
OLLAMA_MODEL=llama3.1
OLLAMA_BASE_URL=http://localhost:11434
```

## Setup

```
git clone https://github.com/manojperi26/data-whisper.git
cd data-whisper
pip install -r requirements.txt
streamlit run app.py
```

Get a free Groq API key at console.groq.com/keys and paste it into the sidebar (or set it as the `GROQ_API_KEY` environment variable).

## Tech stack

`streamlit` · `langchain` + `langchain-classic` (ReAct agent) · `langchain-experimental` (sandboxed Python REPL tool) · `langchain-groq` (Llama 3.3 70B) · `pandas` · `matplotlib`

## Notes

- Max 25 agent iterations / 120s execution time per question
- If you hit a Groq rate limit, wait a bit or swap in a fresh key
