"""Data analysis tools exposed to the ReAct agent.

Each tool operates on the sales dataset (a pandas :class:`~pandas.DataFrame`)
and returns a structured, JSON serialisable result. Tools are declared with
LangChain's ``@tool`` decorator and use Pydantic models for their input
schemas. The decorator turns each function into a LangGraph compatible tool
whose name, description and argument schema are advertised to the language
model. Because the model selects tools purely from these descriptions and
schemas, the wording here is deliberately explicit about what each tool does
and when to use it.

The dataset is loaded once and cached so repeated tool calls within a single
agent run do not re-read the CSV from disk.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.config import get_config

EXPECTED_COLUMNS = {
    "date",
    "product_category",
    "region",
    "units_sold",
    "revenue",
}


@lru_cache(maxsize=1)
def load_dataframe() -> pd.DataFrame:
    """Load and cache the sales dataset from the configured CSV path.

    The result is cached with :func:`functools.lru_cache` so the file is read
    from disk only once per process.

    Returns:
        A :class:`pandas.DataFrame` with parsed ``date`` values.

    Raises:
        FileNotFoundError: If the configured dataset path does not exist.
        ValueError: If the dataset is missing any expected column.
    """

    config = get_config()
    frame = pd.read_csv(config.data_path)

    missing = EXPECTED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            f"Dataset {config.data_path!r} is missing columns: {sorted(missing)}"
        )

    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _resolve_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Return the provided frame, or the cached dataset when ``None``."""

    return load_dataframe() if frame is None else frame


class FilterByRegionInput(BaseModel):
    """Input schema for :func:`filter_by_region`."""

    region: str = Field(
        description=(
            "Region name to filter on, for example 'North', 'South', 'East' "
            "or 'West'. Matching is case insensitive."
        )
    )


class AggregateByCategoryInput(BaseModel):
    """Input schema for :func:`aggregate_by_category`."""

    metric: str = Field(
        default="revenue",
        description=(
            "Numeric column to aggregate per product category. Use 'revenue' "
            "for sales value or 'units_sold' for quantity."
        ),
    )
    operation: str = Field(
        default="sum",
        description=(
            "Aggregation to apply within each category. One of 'sum', 'mean', "
            "'min', 'max' or 'count'."
        ),
    )


class SummaryStatisticsInput(BaseModel):
    """Input schema for :func:`compute_summary_statistics`."""

    column: str = Field(
        default="revenue",
        description=(
            "Numeric column to summarise. Use 'revenue' or 'units_sold'."
        ),
    )


class TopPerformersInput(BaseModel):
    """Input schema for :func:`find_top_performers`."""

    metric: str = Field(
        default="revenue",
        description=(
            "Numeric column used to rank rows. Use 'revenue' or 'units_sold'."
        ),
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of top rows to return (between 1 and 50).",
    )


def _filter_by_region(region: str, frame: pd.DataFrame | None = None) -> dict[str, Any]:
    """Core logic for :func:`filter_by_region` (also used directly in tests)."""

    data = _resolve_frame(frame)
    matched = data[data["region"].str.lower() == region.strip().lower()]
    return {
        "region": region,
        "row_count": int(len(matched)),
        "total_units_sold": int(matched["units_sold"].sum()),
        "total_revenue": round(float(matched["revenue"].sum()), 2),
    }


def _aggregate_by_category(
    metric: str = "revenue",
    operation: str = "sum",
    frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Core logic for :func:`aggregate_by_category` (also used in tests)."""

    data = _resolve_frame(frame)
    if metric not in {"revenue", "units_sold"}:
        raise ValueError("metric must be 'revenue' or 'units_sold'")
    if operation not in {"sum", "mean", "min", "max", "count"}:
        raise ValueError(
            "operation must be one of 'sum', 'mean', 'min', 'max', 'count'"
        )

    grouped = data.groupby("product_category")[metric].agg(operation)
    result = {
        category: round(float(value), 2)
        for category, value in grouped.sort_values(ascending=False).items()
    }
    return {"metric": metric, "operation": operation, "by_category": result}


def _compute_summary_statistics(
    column: str = "revenue", frame: pd.DataFrame | None = None
) -> dict[str, Any]:
    """Core logic for :func:`compute_summary_statistics` (also used in tests)."""

    data = _resolve_frame(frame)
    if column not in {"revenue", "units_sold"}:
        raise ValueError("column must be 'revenue' or 'units_sold'")

    series = data[column]
    return {
        "column": column,
        "count": int(series.count()),
        "sum": round(float(series.sum()), 2),
        "mean": round(float(series.mean()), 2),
        "median": round(float(series.median()), 2),
        "min": round(float(series.min()), 2),
        "max": round(float(series.max()), 2),
        "std": round(float(series.std()), 2),
    }


def _find_top_performers(
    metric: str = "revenue",
    limit: int = 5,
    frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Core logic for :func:`find_top_performers` (also used in tests)."""

    data = _resolve_frame(frame)
    if metric not in {"revenue", "units_sold"}:
        raise ValueError("metric must be 'revenue' or 'units_sold'")

    top = data.nlargest(limit, metric)
    records = [
        {
            "date": row["date"].strftime("%Y-%m-%d"),
            "product_category": row["product_category"],
            "region": row["region"],
            "units_sold": int(row["units_sold"]),
            "revenue": round(float(row["revenue"]), 2),
        }
        for _, row in top.iterrows()
    ]
    return {"metric": metric, "limit": limit, "top_performers": records}


@tool(args_schema=FilterByRegionInput)
def filter_by_region(region: str) -> dict[str, Any]:
    """Filter the sales data to a single region and summarise it.

    Use this when a question is scoped to one geographic region (North, South,
    East or West). Returns the number of matching rows, total units sold and
    total revenue for that region.
    """

    return _filter_by_region(region)


@tool(args_schema=AggregateByCategoryInput)
def aggregate_by_category(
    metric: str = "revenue", operation: str = "sum"
) -> dict[str, Any]:
    """Aggregate a numeric metric per product category.

    Use this to compare product categories, for example total revenue per
    category or average units sold per category. Returns a mapping of category
    to the aggregated value, sorted from highest to lowest.
    """

    return _aggregate_by_category(metric=metric, operation=operation)


@tool(args_schema=SummaryStatisticsInput)
def compute_summary_statistics(column: str = "revenue") -> dict[str, Any]:
    """Compute descriptive statistics for a numeric column.

    Use this for questions about the overall distribution of a metric across
    the whole dataset. Returns count, sum, mean, median, min, max and standard
    deviation for the chosen column ('revenue' or 'units_sold').
    """

    return _compute_summary_statistics(column=column)


@tool(args_schema=TopPerformersInput)
def find_top_performers(
    metric: str = "revenue", limit: int = 5
) -> dict[str, Any]:
    """Return the individual sales records with the highest metric values.

    Use this to identify the best performing rows, for example the top five
    sales by revenue. Returns a ranked list of records with their date,
    category, region, units sold and revenue.
    """

    return _find_top_performers(metric=metric, limit=limit)


# Convenience collection passed to the agent when the graph is built.
ALL_TOOLS = [
    filter_by_region,
    aggregate_by_category,
    compute_summary_statistics,
    find_top_performers,
]
