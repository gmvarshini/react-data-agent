"""Unit tests for the data analysis tools.

The tests run against a small, hand-built fixture dataframe with values chosen
so the expected results can be computed by hand. This isolates the tool logic
from the full sample dataset and the language model.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.tools import (
    _aggregate_by_category,
    _compute_summary_statistics,
    _filter_by_region,
    _find_top_performers,
)


@pytest.fixture()
def sample_frame() -> pd.DataFrame:
    """A tiny, fully known sales dataframe used across the tool tests."""

    frame = pd.DataFrame(
        {
            "date": [
                "2025-01-01",
                "2025-01-02",
                "2025-01-03",
                "2025-01-04",
                "2025-01-05",
                "2025-01-06",
            ],
            "product_category": [
                "Electronics",
                "Electronics",
                "Furniture",
                "Furniture",
                "Clothing",
                "Clothing",
            ],
            "region": ["North", "South", "North", "South", "North", "West"],
            "units_sold": [10, 20, 5, 15, 4, 6],
            "revenue": [1000.0, 2000.0, 500.0, 1500.0, 200.0, 300.0],
        }
    )
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def test_filter_by_region_matches_expected(sample_frame: pd.DataFrame) -> None:
    """North has three rows totalling 19 units and 1700.0 revenue."""

    result = _filter_by_region("North", frame=sample_frame)

    assert result["region"] == "North"
    assert result["row_count"] == 3
    assert result["total_units_sold"] == 19
    assert result["total_revenue"] == 1700.0


def test_filter_by_region_is_case_insensitive(sample_frame: pd.DataFrame) -> None:
    """Region matching ignores case and surrounding whitespace."""

    result = _filter_by_region("  south ", frame=sample_frame)

    assert result["row_count"] == 2
    assert result["total_units_sold"] == 35
    assert result["total_revenue"] == 3500.0


def test_filter_by_region_no_match(sample_frame: pd.DataFrame) -> None:
    """An unknown region yields zeroed totals rather than an error."""

    result = _filter_by_region("East", frame=sample_frame)

    assert result["row_count"] == 0
    assert result["total_units_sold"] == 0
    assert result["total_revenue"] == 0.0


def test_aggregate_by_category_sum_revenue(sample_frame: pd.DataFrame) -> None:
    """Revenue summed per category, sorted highest first."""

    result = _aggregate_by_category(
        metric="revenue", operation="sum", frame=sample_frame
    )

    assert result["by_category"] == {
        "Electronics": 3000.0,
        "Furniture": 2000.0,
        "Clothing": 500.0,
    }
    # Sorted descending: Electronics must come before Clothing.
    keys = list(result["by_category"].keys())
    assert keys == ["Electronics", "Furniture", "Clothing"]


def test_aggregate_by_category_mean_units(sample_frame: pd.DataFrame) -> None:
    """Mean units sold per category."""

    result = _aggregate_by_category(
        metric="units_sold", operation="mean", frame=sample_frame
    )

    assert result["by_category"]["Electronics"] == 15.0
    assert result["by_category"]["Furniture"] == 10.0
    assert result["by_category"]["Clothing"] == 5.0


def test_aggregate_by_category_rejects_bad_metric(
    sample_frame: pd.DataFrame,
) -> None:
    """An unsupported metric raises ValueError."""

    with pytest.raises(ValueError):
        _aggregate_by_category(metric="profit", frame=sample_frame)


def test_compute_summary_statistics_revenue(sample_frame: pd.DataFrame) -> None:
    """Descriptive statistics for the revenue column."""

    result = _compute_summary_statistics(column="revenue", frame=sample_frame)

    assert result["count"] == 6
    assert result["sum"] == 5500.0
    # Mean of the six revenue values is 916.67.
    assert result["mean"] == pytest.approx(916.67, abs=0.01)
    # Median of [200, 300, 500, 1000, 1500, 2000] is 750.0.
    assert result["median"] == 750.0
    assert result["min"] == 200.0
    assert result["max"] == 2000.0


def test_compute_summary_statistics_rejects_bad_column(
    sample_frame: pd.DataFrame,
) -> None:
    """An unsupported column raises ValueError."""

    with pytest.raises(ValueError):
        _compute_summary_statistics(column="date", frame=sample_frame)


def test_find_top_performers_by_revenue(sample_frame: pd.DataFrame) -> None:
    """The two highest revenue rows are returned in descending order."""

    result = _find_top_performers(metric="revenue", limit=2, frame=sample_frame)

    records = result["top_performers"]
    assert len(records) == 2
    assert records[0]["revenue"] == 2000.0
    assert records[0]["region"] == "South"
    assert records[1]["revenue"] == 1500.0


def test_find_top_performers_by_units(sample_frame: pd.DataFrame) -> None:
    """Ranking by units_sold returns the largest quantity first."""

    result = _find_top_performers(
        metric="units_sold", limit=1, frame=sample_frame
    )

    top = result["top_performers"][0]
    assert top["units_sold"] == 20
    assert top["product_category"] == "Electronics"
