"""
app.py - AI Data Analyst: Chat with Your Data
Main Streamlit application entry point
"""

import os
import streamlit as st
import pandas as pd

from db import load_csv_to_duckdb, get_schema, run_query, sanitize_table_name, get_table_stats
from analysis import (
    get_statistical_summary, get_data_quality_report,
    get_correlation_matrix, get_distribution_plots, detect_trends
)
from llm import generate_sql, explain_results
from utils import (
    detect_chart_type, build_chart, df_preview, is_valid_dataframe,
    get_cached_result, cache_result, clear_cache
)

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.stApp { background: linear-gradient(135deg, #0f0f1a 0%, #111827 50%, #0f0f1a 100%); }
section[data-testid="stSidebar"] { background: #111827; border-right: 1px solid #1f2937; }
.result-card { background: #1a1f2e; border: 1px solid #2d3748; border-radius: 12px; padding: 1.25rem 1.5rem; margin-bottom: 1rem; }
.sql-block { background: #0d1117; border: 1px solid #30363d; border-left: 3px solid #6366f1; border-radius: 8px; padding: 1rem 1.25rem; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #c9d1d9; white-space: pre-wrap; word-break: break-all; }
.explanation-box { background: linear-gradient(135deg, #1a1f2e, #1e2333); border: 1px solid #3b4a6b; border-left: 3px solid #22d3ee; border-radius: 8px; padding: 1rem 1.25rem; color: #94a3b8; font-size: 0.95rem; line-height: 1.6; }
.badge { display: inline-block; padding: 0.2rem 0.65rem; border-radius: 9999px; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; }
.badge-cached { background: #1a3a2a; color: #4ade80; border: 1px solid #166534; }
.badge-live { background: #1e1a3a; color: #a78bfa; border: 1px solid #4c1d95; }
.metric-tile { background: #1a1f2e; border: 1px solid #2d3748; border-radius: 10px; padding: 0.9rem 1.1rem; text-align: center; }
.metric-tile .val { font-size: 1.6rem; font-weight: 700; color: #e2e8f0; }
.metric-tile .lbl { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; }
h1, h2, h3 { color: #f1f5f9 !important; }
</style>
""", unsafe_allow_html=True)


def init_state():
    defaults = {
        "conn": None,
        "df": None,
        "table_name": None,
        "schema": None,
        "history": [],
        "api_key": os.getenv("GROQ_API_KEY", ""),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

with st.sidebar:
    st.markdown("## 🔍 AI Data Analyst")
    st.markdown("---")
    st.markdown("### Groq API Key")
    st.markdown(
        "<small style='color:#4b5563;'>Free at <a href='https://console.groq.com' target='_blank' style='color:#6366f1;'>console.groq.com</a></small>",
        unsafe_allow_html=True
    )
    api_key_input = st.text_input(
        "Enter your key",
        type="password",
        value=st.session_state.api_key,
        placeholder="gsk_...",
        help="Your key is stored only in this browser session.",
    )
    if api_key_input:
        st.session_state.api_key = api_key_input

    st.markdown("---")
    st.markdown("### Upload Dataset")
    uploaded_file = st.file_uploader(
        "CSV file", type=["csv"],
        help="Upload any CSV file. It will be loaded into an in-memory SQLite database."
    )

    st.markdown("---")
    st.markdown("### Chart Options")
    chart_override = st.selectbox(
        "Chart type (auto = smart detection)",
        options=["auto", "bar", "line", "scatter", "none"],
        index=0,
    )

    st.markdown("---")
    if st.button("Clear query cache"):
        clear_cache()
        st.success("Cache cleared!")
    if st.session_state.history and st.button("Clear history"):
        st.session_state.history = []
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<small style='color:#4b5563;'>Built with Streamlit · Llama 3 (Groq) · SQLite</small>",
        unsafe_allow_html=True
    )

if uploaded_file:
    file_key = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.get("_file_key") != file_key:
        with st.spinner("Loading dataset into database..."):
            try:
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='latin-1')
                table_name = sanitize_table_name(uploaded_file.name.replace(".csv", ""))
                conn = load_csv_to_duckdb(df, table_name)
                schema = get_schema(conn, table_name)
                st.session_state.df = df
                st.session_state.conn = conn
                st.session_state.table_name = table_name
                st.session_state.schema = schema
                st.session_state.history = []
                st.session_state._file_key = file_key
                clear_cache()
                st.toast(f"Loaded {uploaded_file.name} — {len(df):,} rows x {len(df.columns)} columns")
            except Exception as e:
                st.error(f"Failed to load file: {e}")

st.markdown("# 🔍 AI Data Analyst")
st.markdown("<p style='color:#64748b; margin-top:-0.5rem;'>Upload a CSV, ask questions in plain English, get SQL + insights instantly. Powered by Llama 3 on Groq — 100% free.</p>", unsafe_allow_html=True)

if st.session_state.df is None:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='result-card'><h3>📤 Upload</h3><p style='color:#64748b;'>Drop any CSV in the sidebar to get started.</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='result-card'><h3>💬 Ask</h3><p style='color:#64748b;'>Type a question in plain English — no SQL needed.</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='result-card'><h3>📊 Explore</h3><p style='color:#64748b;'>Get SQL queries, result tables, charts, and AI explanations.</p></div>", unsafe_allow_html=True)
    st.info("Upload a CSV file from the sidebar to begin.")
    st.stop()

df = st.session_state.df

with st.expander("📋 Dataset Overview", expanded=True):
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"<div class='metric-tile'><div class='val'>{len(df):,}</div><div class='lbl'>Rows</div></div>", unsafe_allow_html=True)
    with m2:
        st.markdown(f"<div class='metric-tile'><div class='val'>{len(df.columns)}</div><div class='lbl'>Columns</div></div>", unsafe_allow_html=True)
    with m3:
        numeric_count = len(df.select_dtypes(include='number').columns)
        st.markdown(f"<div class='metric-tile'><div class='val'>{numeric_count}</div><div class='lbl'>Numeric</div></div>", unsafe_allow_html=True)
    with m4:
        null_pct = round(df.isnull().sum().sum() / df.size * 100, 1)
        st.markdown(f"<div class='metric-tile'><div class='val'>{null_pct}%</div><div class='lbl'>Null values</div></div>", unsafe_allow_html=True)
    st.markdown("**Column names:**")
    st.code(", ".join(df.columns.tolist()), language=None)
    st.markdown("**Sample rows:**")
    st.dataframe(df_preview(df, 5), use_container_width=True)

# ---------------------------------------------------------------------------
# Data Science Analysis Section
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 🔬 Data Science Analysis")

ds_tab1, ds_tab2, ds_tab3, ds_tab4 = st.tabs([
    "📊 Statistics", "🧹 Data Quality", "🔗 Correlations", "📈 Distributions"
])

with ds_tab1:
    st.markdown("**Statistical Summary**")
    summary_df = get_statistical_summary(df)
    if not summary_df.empty:
        st.dataframe(summary_df, use_container_width=True)
        trends = detect_trends(df)
        if trends:
            st.markdown("**Trend Detection**")
            st.dataframe(pd.DataFrame(trends), use_container_width=True)
        else:
            st.info("No datetime columns detected for trend analysis.")
    else:
        st.info("No numeric columns found for statistical analysis.")

with ds_tab2:
    st.markdown("**Data Quality Report**")
    quality = get_data_quality_report(df)
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.markdown(f"<div class='metric-tile'><div class='val'>{quality['total_rows']:,}</div><div class='lbl'>Total Rows</div></div>", unsafe_allow_html=True)
    with q2:
        st.markdown(f"<div class='metric-tile'><div class='val'>{quality['total_columns']}</div><div class='lbl'>Columns</div></div>", unsafe_allow_html=True)
    with q3:
        st.markdown(f"<div class='metric-tile'><div class='val'>{quality['duplicate_rows']:,}</div><div class='lbl'>Duplicates</div></div>", unsafe_allow_html=True)
    with q4:
        st.markdown(f"<div class='metric-tile'><div class='val'>{quality['duplicate_pct']}%</div><div class='lbl'>Duplicate %</div></div>", unsafe_allow_html=True)

    if not quality['missing_df'].empty:
        st.markdown("**Missing Values by Column**")
        st.dataframe(quality['missing_df'], use_container_width=True)
    else:
        st.success("No missing values found!")

with ds_tab3:
    corr_fig = get_correlation_matrix(df)
    if corr_fig:
        st.plotly_chart(corr_fig, use_container_width=True)
    else:
        st.info("Need at least 2 numeric columns for correlation analysis.")

with ds_tab4:
    dist_fig = get_distribution_plots(df)
    if dist_fig:
        st.plotly_chart(dist_fig, use_container_width=True)
    else:
        st.info("No numeric columns found for distribution plots.")

st.markdown("---")
st.markdown("### 💬 Ask a Question")

example_questions = [
    "Show total sales by region",
    "What are the top 5 products by revenue?",
    "Show monthly sales trend",
    "Which category has the highest average order value?",
    "Count records per year",
]

with st.form("query_form", clear_on_submit=False):
    question = st.text_input(
        "Your question",
        placeholder="e.g. Which region has the highest sales?",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("🔍 Analyze", use_container_width=False)

st.markdown("<small style='color:#4b5563;'>Try an example:</small>", unsafe_allow_html=True)
cols = st.columns(len(example_questions))
for i, eq in enumerate(example_questions):
    with cols[i]:
        if st.button(eq, key=f"ex_{i}", use_container_width=True):
            question = eq
            submitted = True

if submitted and question:
    if not st.session_state.api_key:
        st.error("Please enter your Groq API key in the sidebar. Get one free at https://console.groq.com")
        st.stop()

    schema = st.session_state.schema
    table_name = st.session_state.table_name
    conn = st.session_state.conn

    cached = get_cached_result(question, schema)

    if cached:
        sql, result_df = cached
        was_cached = True
    else:
        was_cached = False

        with st.spinner("Generating SQL query..."):
            sql, err = generate_sql(question, schema, table_name)

        if err:
            st.error(f"SQL Generation Failed: {err}")
            st.stop()

        with st.spinner("Running query..."):
            result_df, run_err = run_query(conn, sql)

        if run_err:
            st.error(f"Query Execution Failed: {run_err}")
            st.markdown("**Generated SQL was:**")
            st.code(sql, language="sql")
            st.stop()

        if is_valid_dataframe(result_df):
            cache_result(question, schema, sql, result_df)

    with st.spinner("Generating explanation..."):
        explanation, _ = explain_results(question, sql, result_df)

    if chart_override == "auto":
        chart_type = detect_chart_type(result_df) if is_valid_dataframe(result_df) else "none"
    else:
        chart_type = chart_override

    fig = build_chart(result_df, chart_type, title=question) if is_valid_dataframe(result_df) else None

    st.session_state.history.insert(0, {
        "question": question,
        "sql": sql,
        "result_df": result_df,
        "explanation": explanation,
        "cached": was_cached,
        "chart_type": chart_type,
        "fig": fig,
    })

if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📊 Results")

    for idx, entry in enumerate(st.session_state.history):
        badge = (
            "<span class='badge badge-cached'>cached</span>"
            if entry["cached"]
            else "<span class='badge badge-live'>live</span>"
        )

        with st.expander(f"{entry['question']} {badge}", expanded=(idx == 0)):
            tab1, tab2, tab3 = st.tabs(["📊 Chart & Results", "🔧 SQL Query", "💡 Explanation"])

            with tab1:
                if entry["fig"]:
                    st.plotly_chart(entry["fig"], use_container_width=True)
                elif entry["chart_type"] == "none":
                    st.info("No chart generated — result does not lend itself to visualization.")

                result_df = entry["result_df"]
                if is_valid_dataframe(result_df):
                    st.markdown(f"**{len(result_df):,} row(s) returned**")
                    st.dataframe(result_df, use_container_width=True)
                    csv_bytes = result_df.to_csv(index=False).encode()
                    st.download_button(
                        "Download results as CSV",
                        data=csv_bytes,
                        file_name=f"result_{idx}.csv",
                        mime="text/csv",
                        key=f"dl_{idx}"
                    )
                else:
                    st.warning("The query returned no results.")

            with tab2:
                st.code(entry["sql"], language="sql")

            with tab3:
                if entry["explanation"]:
                    st.markdown(f"<div class='explanation-box'>{entry['explanation']}</div>", unsafe_allow_html=True)
                else:
                    st.info("No explanation available.")