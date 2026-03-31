"""
llm.py - LLM interaction module
Uses Groq's free API with multiple Llama models.
Supports conversation memory for context-aware follow-up questions.
"""

import os
import re
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Available Groq models for the model switcher
AVAILABLE_MODELS = {
    "Llama 3.3 70B (Best Quality)": "llama-3.3-70b-versatile",
    "Llama 3.1 8B (Fastest)": "llama-3.1-8b-instant",
    "Mixtral 8x7B (Balanced)": "mixtral-8x7b-32768",
}


def get_client() -> Groq:
    """Initialize and return a Groq client using the API key from session state."""
    api_key = st.session_state.get("api_key", "") or os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("No Groq API key found. Please enter your API key in the sidebar.")
    return Groq(api_key=api_key)


def get_selected_model() -> str:
    """Get the currently selected model from session state."""
    model_name = st.session_state.get("selected_model", "Llama 3.3 70B (Best Quality)")
    return AVAILABLE_MODELS.get(model_name, "llama-3.3-70b-versatile")


def build_sql_prompt(question: str, schema: str, table_name: str, context: str = "") -> str:
    """
    Build a robust prompt for SQL generation.
    Includes conversation history context for follow-up questions.
    """
    context_section = f"\n{context}\n" if context else ""

    # Handle multiple tables
    if isinstance(table_name, list):
        table_ref = f"Available tables: {', '.join(table_name)}"
    else:
        table_ref = f"Use only the table name: {table_name}"

    return f"""You are an expert SQL analyst. Your job is to convert a natural language question into a valid DuckDB SQL query.

DATABASE SCHEMA:
{schema}

{context_section}
RULES - follow these strictly:
1. Output ONLY a single valid DuckDB SELECT statement. No explanations, no markdown, no code blocks.
2. {table_ref}
3. Use only column names that exist in the schema above.
4. DuckDB supports advanced SQL - you can use window functions, QUALIFY, DATE_TRUNC, STRFTIME, MEDIAN, STDDEV, and other analytical functions.
5. If the question involves time/dates, use DuckDB date functions like DATE_TRUNC('month', column) or STRFTIME(column, '%Y-%m').
6. If this is a follow-up question, refer to the previous conversation history above for context.
7. Keep the query simple and correct. Do not guess column names.
8. If you cannot answer with the given schema, output exactly: CANNOT_ANSWER

USER QUESTION:
{question}

SQL QUERY:"""


def generate_sql(question: str, schema: str, table_name, context: str = ""):
    """
    Use the LLM to generate a SQL query from a natural language question.
    Supports conversation context for follow-up questions.

    Args:
        question: User's natural language question
        schema: Database schema string
        table_name: Name of the SQLite table (string or list)
        context: Optional conversation history context

    Returns:
        Tuple of (sql_query or None, error_message or None)
    """
    try:
        client = get_client()
        model = get_selected_model()
        prompt = build_sql_prompt(question, schema, table_name, context)

        response = client.chat.completions.create(
            model=model,
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
    """Build a prompt for generating a plain-English explanation of query results."""
    return f"""You are a data analyst assistant. A user asked a question and we ran a SQL query to answer it.

USER QUESTION: {question}

SQL QUERY USED:
{sql}

RESULT SUMMARY:
{result_summary}

Write a concise 2-3 sentence explanation of what the results show. Be specific and reference actual numbers or values from the results. Write in plain English for a non-technical audience."""


def explain_results(question: str, sql: str, result_df):
    """
    Generate a plain-English explanation of query results.

    Args:
        question: Original user question
        sql: SQL query that was executed
        result_df: Pandas DataFrame with query results

    Returns:
        Tuple of (explanation string or None, error message or None)
    """
    try:
        client = get_client()
        model = get_selected_model()

        if result_df is None or result_df.empty:
            result_summary = "The query returned no results."
        else:
            result_summary = result_df.head(20).to_string(index=False)
            if len(result_df) > 20:
                result_summary += f"\n... ({len(result_df)} total rows)"

        prompt = build_explanation_prompt(question, sql, result_summary)

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )

        return response.choices[0].message.content.strip(), None

    except ValueError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Could not generate explanation: {str(e)}"