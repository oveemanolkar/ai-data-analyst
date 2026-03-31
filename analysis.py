"""
analysis.py - Data Science analysis module
Provides automatic statistical summaries, correlation analysis,
data quality reports, and trend detection.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_statistical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a detailed statistical summary for all numeric columns.
    Includes mean, median, std, min, max, and outlier count.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with statistical summary
    """
    numeric_df = df.select_dtypes(include='number')
    if numeric_df.empty:
        return pd.DataFrame()

    summary = []
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        outliers = ((series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))).sum()

        summary.append({
            "Column": col,
            "Mean": round(series.mean(), 2),
            "Median": round(series.median(), 2),
            "Std Dev": round(series.std(), 2),
            "Min": round(series.min(), 2),
            "Max": round(series.max(), 2),
            "Outliers": int(outliers),
            "Outlier %": round(outliers / len(series) * 100, 1)
        })

    return pd.DataFrame(summary)


def get_data_quality_report(df: pd.DataFrame) -> dict:
    """
    Generate a data quality report including missing values,
    duplicates, and column data types.

    Args:
        df: Input DataFrame

    Returns:
        Dict with quality metrics
    """
    total_rows = len(df)
    duplicate_rows = df.duplicated().sum()

    missing = []
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            missing.append({
                "Column": col,
                "Missing Count": int(null_count),
                "Missing %": round(null_count / total_rows * 100, 1)
            })

    return {
        "total_rows": total_rows,
        "total_columns": len(df.columns),
        "duplicate_rows": int(duplicate_rows),
        "duplicate_pct": round(duplicate_rows / total_rows * 100, 1),
        "missing_df": pd.DataFrame(missing) if missing else pd.DataFrame(),
        "columns_with_missing": len(missing)
    }


def get_correlation_matrix(df: pd.DataFrame):
    """
    Compute and visualize the correlation matrix for numeric columns.

    Args:
        df: Input DataFrame

    Returns:
        Plotly heatmap figure or None
    """
    numeric_df = df.select_dtypes(include='number')
    if numeric_df.shape[1] < 2:
        return None

    corr = numeric_df.corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Correlation Matrix",
        template="plotly_dark"
    )
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


def get_distribution_plots(df: pd.DataFrame):
    """
    Generate distribution histograms for all numeric columns.

    Args:
        df: Input DataFrame

    Returns:
        Plotly figure with subplots or None
    """
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if not numeric_cols:
        return None

    # Limit to first 6 columns to avoid overcrowding
    numeric_cols = numeric_cols[:6]
    cols = min(3, len(numeric_cols))
    rows = (len(numeric_cols) + cols - 1) // cols

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=numeric_cols)

    for i, col in enumerate(numeric_cols):
        row = i // cols + 1
        col_pos = i % cols + 1
        series = df[col].dropna()

        fig.add_trace(
            go.Histogram(
                x=series,
                name=col,
                marker_color="#6366f1",
                showlegend=False
            ),
            row=row, col=col_pos
        )

    fig.update_layout(
        height=300 * rows,
        template="plotly_dark",
        title="Distribution of Numeric Columns",
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig


def detect_trends(df: pd.DataFrame) -> list:
    """
    Detect trends in numeric columns over time.
    Looks for datetime columns and computes if numeric columns
    are trending up, down, or are stable.

    Args:
        df: Input DataFrame

    Returns:
        List of dicts with trend info per column
    """
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    if not datetime_cols or not numeric_cols:
        return []

    date_col = datetime_cols[0]
    trends = []

    for col in numeric_cols[:4]:  # Limit to 4 columns
        try:
            temp = df[[date_col, col]].dropna()
            temp = temp.sort_values(date_col)

            # Split into first and second half and compare means
            mid = len(temp) // 2
            first_half_mean = temp[col].iloc[:mid].mean()
            second_half_mean = temp[col].iloc[mid:].mean()

            if first_half_mean == 0:
                continue

            change_pct = ((second_half_mean - first_half_mean) / abs(first_half_mean)) * 100

            if change_pct > 5:
                direction = "📈 Trending Up"
            elif change_pct < -5:
                direction = "📉 Trending Down"
            else:
                direction = "➡️ Stable"

            trends.append({
                "Column": col,
                "Trend": direction,
                "Change": f"{change_pct:+.1f}%"
            })
        except Exception:
            continue

    return trends