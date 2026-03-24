"""
utils.py - Utility / helper functions
Chart generation, type detection, query caching, and misc helpers
"""

import hashlib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


_query_cache = {}


def cache_key(question: str, schema: str) -> str:
    raw = f"{question.strip().lower()}|{schema}"
    return hashlib.md5(raw.encode()).hexdigest()


def get_cached_result(question: str, schema: str):
    key = cache_key(question, schema)
    return _query_cache.get(key)


def cache_result(question: str, schema: str, sql: str, df: pd.DataFrame) -> None:
    key = cache_key(question, schema)
    _query_cache[key] = (sql, df)


def clear_cache() -> None:
    _query_cache.clear()


def _looks_like_datetime(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_object_dtype(series):
        sample = series.dropna().head(5).tolist()
        date_keywords = ["jan", "feb", "mar", "apr", "may", "jun",
                         "jul", "aug", "sep", "oct", "nov", "dec",
                         "2020", "2021", "2022", "2023", "2024", "2025", "2026",
                         "-01-", "-02-", "/01/", "/02/"]
        for val in sample:
            val_str = str(val).lower()
            if any(kw in val_str for kw in date_keywords):
                return True
    return False


def detect_chart_type(df: pd.DataFrame) -> str:
    if df is None or df.empty or len(df.columns) < 2:
        return "none"
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]
    if not numeric_cols:
        return "none"
    for col in non_numeric_cols:
        if _looks_like_datetime(df[col]):
            return "line"
    if non_numeric_cols:
        first_cat = df[non_numeric_cols[0]]
        if first_cat.nunique() <= 30:
            return "bar"
    if len(numeric_cols) >= 2:
        return "scatter"
    return "none"


def build_chart(df: pd.DataFrame, chart_type: str, title: str = ""):
    if df is None or df.empty or chart_type == "none":
        return None
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]
    try:
        if chart_type == "bar" and non_numeric_cols and numeric_cols:
            x_col = non_numeric_cols[0]
            y_col = numeric_cols[0]
            fig = px.bar(
                df, x=x_col, y=y_col, title=title or f"{y_col} by {x_col}",
                color_discrete_sequence=["#6366f1"],
                template="plotly_dark"
            )
            fig.update_layout(xaxis_tickangle=-35)
            return fig
        elif chart_type == "line":
            x_col = non_numeric_cols[0] if non_numeric_cols else df.columns[0]
            y_col = numeric_cols[0] if numeric_cols else df.columns[1]
            fig = px.line(
                df, x=x_col, y=y_col, title=title or f"{y_col} over {x_col}",
                markers=True,
                color_discrete_sequence=["#22d3ee"],
                template="plotly_dark"
            )
            return fig
        elif chart_type == "scatter" and len(numeric_cols) >= 2:
            fig = px.scatter(
                df, x=numeric_cols[0], y=numeric_cols[1],
                title=title or f"{numeric_cols[1]} vs {numeric_cols[0]}",
                color_discrete_sequence=["#f472b6"],
                template="plotly_dark"
            )
            return fig
    except Exception:
        return None
    return None


def df_preview(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return df.head(n)


def is_valid_dataframe(df) -> bool:
    return isinstance(df, pd.DataFrame) and not df.empty