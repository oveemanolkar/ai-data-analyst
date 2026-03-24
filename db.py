"""
db.py - Database handling module
Manages CSV ingestion and SQLite operations
"""

import sqlite3
import pandas as pd
import re


def sanitize_table_name(name: str) -> str:
    """Convert a filename into a valid SQLite table name."""
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    if name and name[0].isdigit():
        name = 't_' + name
    return name.lower() or 'data'


def load_csv_to_sqlite(df: pd.DataFrame, table_name: str, db_path: str = ":memory:") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.commit()
    return conn


def get_schema(conn: sqlite3.Connection, table_name: str) -> str:
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    if not columns:
        return ""
    col_defs = ", ".join([f"{col[1]} ({col[2]})" for col in columns])
    return f"Table: {table_name}\nColumns: {col_defs}"


def run_query(conn: sqlite3.Connection, sql: str):
    try:
        stripped = sql.strip().upper()
        if not stripped.startswith("SELECT"):
            return None, "Only SELECT queries are allowed for safety reasons."
        df = pd.read_sql_query(sql, conn)
        return df, None
    except pd.errors.DatabaseError as e:
        return None, f"Database error: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error while running query: {str(e)}"