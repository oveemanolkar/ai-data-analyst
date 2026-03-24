"""
llm.py - LLM interaction module
Uses Groq's free API with Llama 3 for SQL generation and result explanation.
"""

from dotenv import load_dotenv
load_dotenv()

import os
import re
import streamlit as st
from groq import Groq


def get_client():
    api_key = st.session_state.get("api_key", "") or os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("No Groq API key found. Please enter your API key in the sidebar.")
    return Groq(api_key=api_key)


def build_sql_prompt(question: str, schema: str, table_name: str) -> str:
    return f"""You are an expert SQL analyst. Your job is to convert a natural language question into a valid SQLite SQL query.

DATABASE SCHEMA:
{schema}

RULES - follow these strictly:
1. Output ONLY a single valid SQLite SELECT statement. No explanations, no markdown, no code blocks.
2. Use only the table name: {table_name}
3. Use only column names that exist in the schema above.
4. Do NOT use features unsupported by SQLite (e.g., no FULL OUTER JOIN, no RIGHT JOIN).
5. If the question involves time/dates, use SQLite date functions (strftime, date, etc.).
6. Keep the query simple and correct. Do not guess column names.
7. If you cannot answer with the given schema, output exactly: CANNOT_ANSWER

USER QUESTION:
{question}

SQL QUERY:"""


def generate_sql(question: str, schema: str, table_name: str):
    try:
        client = get_client()
        prompt = build_sql_prompt(question, schema, table_name)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```(?:sql)?", "", raw, flags=re.IGNORECASE).replace("```", "").strip()
        if raw.upper() == "CANNOT_ANSWER":
            return None, "The model could not answer this question with the available data. Try rephrasing."
        if not raw.upper().startswith("SELECT"):
            return None, f"Model returned unexpected output: {raw[:200]}"
        return raw, None
    except ValueError as e:
        return None, str(e)
    except Exception as e:
        return None, f"LLM error: {str(e)}"


def build_explanation_prompt(question: str, sql: str, result_summary: str) -> str:
    return f"""You are a data analyst assistant. A user asked a question and we ran a SQL query to answer it.

USER QUESTION: {question}

SQL QUERY USED:
{sql}

RESULT SUMMARY:
{result_summary}

Write a concise 2-3 sentence explanation of what the results show. Be specific and reference actual numbers or values from the results. Write in plain English for a non-technical audience."""


def explain_results(question: str, sql: str, result_df):
    try:
        client = get_client()
        if result_df is None or result_df.empty:
            result_summary = "The query returned no results."
        else:
            result_summary = result_df.head(20).to_string(index=False)
            if len(result_df) > 20:
                result_summary += f"\n... ({len(result_df)} total rows)"
        prompt = build_explanation_prompt(question, sql, result_summary)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip(), None
    except ValueError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Could not generate explanation: {str(e)}"