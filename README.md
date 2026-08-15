<div align="center">
  
# 📊 DataWhisperer

**Talk to your data in plain English.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://data-whisper-manoj.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

[**Live Demo**](https://data-whisper-manoj.streamlit.app/) • [**Features**](#features) • [**Installation**](#installation) • [**Architecture**](#architecture)

</div>

---

DataWhisperer is an intelligent data analysis application that turns any CSV file into an interactive conversation. Upload your data, ask questions in natural language, and instantly receive accurate answers, data tables, and beautiful visualizations.

## ✨ Features

- **🗣️ Natural Language Interface:** No SQL or Pandas knowledge required. Just ask questions.
- **📈 Automatic Visualizations:** The agent automatically generates, saves, and displays matplotlib charts.
- **🧠 ReAct Agent Logic:** Employs LangChain's ReAct agent to write, execute, and verify Pandas code in a sandboxed Python REPL.
- **🛡️ Data Privacy & Sandboxing:** The AI model never sees your raw data. It only sees your column schema and operates locally on your dataframe.
- **⚡ Dual-LLM Architecture:** Uses high-speed **Groq (Llama-3.3-70B)** by default, with a seamless automatic fallback to a local **Ollama** instance if API limits are reached.

## 🚀 Quickstart (Local)

### 1. Clone the repository
```bash
git clone https://github.com/manojperi26/data-whisper.git
cd data-whisper
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Environment Variables
Create a `.env` file in the root directory (you can copy `.env.example`) and add your Groq API key:
```ini
GROQ_API_KEY="your_groq_api_key_here"
```

### 4. Run the application
```bash
streamlit run app.py
```
Visit `http://localhost:8501` in your browser.

## ☁️ Deployment (Streamlit Cloud)

DataWhisperer is optimized for deployment on Streamlit Community Cloud:
1. Push this repository to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Deploy the app by selecting your repository and pointing the main file to `app.py`.
4. In the **Advanced Settings** > **Secrets** menu, add your API key: `GROQ_API_KEY = "..."`.

## 🏗️ Architecture

DataWhisperer is built with a highly resilient prompt architecture designed specifically for data analysis:
- **Schema Awareness:** The LLM is provided a dynamically generated schema summary of your CSV to prevent hallucinated columns or data types.
- **Strict Execution Rules:** The agent operates under a strict 14-point rule system to handle NaNs, prevent arbitrary aggregations, and ensure visualizations are explicitly saved to disk rather than trapped in standard output.
- **Auto-Correction:** If the generated Pandas code fails, the agent receives the exact Python stack trace and uses its remaining iteration budget to auto-correct and retry the execution.

## 📄 License
This project is licensed under the [MIT License](LICENSE).
