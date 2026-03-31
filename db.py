"""
db.py - Database handling module
Uses DuckDB for fast analytical queries on large datasets.
DuckDB is significantly faster than SQLite for analytical workloads
and supports larger datasets efficiently.
"""

import duckdb
import pandas as pd
import re


def sanitize_table_name(name: str) -> str:
    """Convert a filename into a valid DuckDB table name."""
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    if name and name[0].isdigit():
        name = 't_' + name
    return name.lower() or 'data'


def load_csv_to_duckdb(df: pd.DataFrame, table_name: str):
    """
    Load a Pandas DataFrame into an in-memory DuckDB database.
    Automatically detects and converts date/datetime columns.

    Args:
        df: The DataFrame to load
        table_name: Name of the table to create

    Returns:
        DuckDB connection object
    """
    # Auto-detect and convert datetime columns
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(100)
            try:
                converted = pd.to_datetime(sample, infer_datetime_format=True)
                if converted.notna().sum() > 80:
                    df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors='coerce')
            except Exception:
                pass

    conn = duckdb.connect(database=':memory:')
    conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
    return conn

def get_schema(conn, table_name: str) -> str:
    """
    Retrieve schema string for a given DuckDB table.

    Args:
        conn: Active DuckDB connection
        table_name: Name of the table

    Returns:
        Schema string with column names and types
    """
    result = conn.execute(f"DESCRIBE {table_name}").fetchall()
    if not result:
        return ""
    col_defs = ", ".join([f"{row[0]} ({row[1]})" for row in result])
    return f"Table: {table_name}\nColumns: {col_defs}"


def run_query(conn, sql: str):
    """
    Execute a SQL query safely on the DuckDB connection.

    Args:
        conn: Active DuckDB connection
        sql: SQL query string

    Returns:
        Tuple of (DataFrame with results, error message or None)
    """
    try:
        # Basic safety check — only allow SELECT statements
        stripped = sql.strip().upper()
        if not stripped.startswith("SELECT"):
            return None, "Only SELECT queries are allowed for safety reasons."

        result = conn.execute(sql).fetchdf()
        return result, None

    except Exception as e:
        return None, f"Query error: {str(e)}"


def get_table_stats(conn, table_name: str):
    """
    Get basic statistics about the table — row count and column count.

    Args:
        conn: Active DuckDB connection
        table_name: Name of the table

    Returns:
        Dict with row_count and col_count
    """
    try:
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        col_count = len(conn.execute(f"DESCRIBE {table_name}").fetchall())
        return {"row_count": row_count, "col_count": col_count}
    except Exception:
        return {"row_count": 0, "col_count": 0}