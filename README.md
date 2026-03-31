# 🔍 AI Data Analyst — Chat with Your Data

A powerful local web application that lets you upload any CSV file and ask questions in plain English. Powered by **Llama 3 on Groq** (free) — no OpenAI API key needed.

## 🌟 Features

### Core
- Upload any CSV dataset (tested with 500k+ rows)
- Ask questions in plain English — no SQL knowledge needed
- Auto-generates optimized SQL queries using an LLM
- Returns results with interactive charts and AI explanations
- Query caching for repeated questions
- Download any result as CSV
- 100% free to run

### Data Engineering
- Powered by **DuckDB** — industry-standard analytical query engine
- Handles large datasets (500k+ rows) efficiently
- Auto-detects and converts date/datetime columns
- Supports advanced SQL — window functions, DATE_TRUNC, MEDIAN, STDDEV

### Data Science
- **Statistical Summary** — mean, median, std dev, min, max, outlier detection
- **Data Quality Report** — missing values, duplicate detection, column profiling
- **Correlation Matrix** — interactive heatmap of numeric column relationships
- **Distribution Plots** — histograms for all numeric columns
- **Trend Detection** — automatically detects if metrics are trending up, down, or stable

### AI Engineering
- **Conversation Memory** — remembers last 10 questions for context-aware follow-ups
- **Multi-CSV Support** — upload multiple datasets and query across all of them
- **Model Switcher** — switch between Groq models (Llama 3.3 70B, Llama 3.1 8B, Mixtral 8x7B)

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Database | DuckDB |
| Data Processing | Pandas, NumPy |
| LLM | Llama 3 via Groq API (free) |
| Visualization | Plotly |
| Language | Python 3.10+ |

## 📁 Project Structure
```
ai_data_analyst/
├── app.py            # Main Streamlit UI
├── llm.py            # LLM interaction — SQL generation, explanations, model switcher
├── db.py             # DuckDB layer — CSV ingestion, schema, query execution
├── analysis.py       # Data Science — stats, quality, correlations, trends
├── memory.py         # Conversation memory — context-aware follow-up questions
├── utils.py          # Helpers — chart detection, caching, DataFrame utilities
├── requirements.txt  # Dependencies
├── .env.example      # Environment variable template
└── README.md
```

## 🚀 Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/oveemanolkar/ai-data-analyst.git
cd ai-data-analyst
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your free Groq API key
- Sign up free at [console.groq.com](https://console.groq.com)
- Create an API key
- Copy `.env.example` to `.env` and add your key:
```
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run the app
```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

## 💡 Example Questions

- *"What are the top 10 countries by total revenue?"*
- *"Show monthly revenue trend"*
- *"Now filter that by European countries only"* (follow-up with memory)
- *"Which products have the highest average unit price?"*
- *"How many unique customers per country?"*

## 📊 Example Dataset

Tested with the [E-Commerce Data](https://www.kaggle.com/datasets/carrie1/ecommerce-data) dataset from Kaggle — 541,909 rows of real UK retail transactions.

## 🤖 Available Models

| Model | Best For |
|---|---|
| Llama 3.3 70B | Best SQL quality, complex questions |
| Llama 3.1 8B | Fastest responses, simple queries |
| Mixtral 8x7B | Balanced speed and quality |

## 🔒 Security

- Only SELECT queries are allowed — no INSERT/UPDATE/DELETE/DROP
- API key stored locally in `.env` — never committed to GitHub
- Data processed entirely in memory — nothing persisted after session ends