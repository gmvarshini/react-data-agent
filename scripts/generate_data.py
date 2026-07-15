"""Generate a deterministic sample sales dataset.

Run with:  uv run python scripts/generate_data.py

The output is written to ``data/sales_data.csv`` and contains at least 100
rows spanning several product categories and regions. A fixed random seed
keeps the data reproducible so tests and documentation stay stable.
"""

from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

CATEGORIES = ["Electronics", "Furniture", "Clothing", "Groceries", "Toys"]
REGIONS = ["North", "South", "East", "West"]

# Approximate per-unit price range for each category, used to derive revenue.
PRICE_RANGE = {
    "Electronics": (120, 900),
    "Furniture": (80, 600),
    "Clothing": (15, 120),
    "Groceries": (2, 40),
    "Toys": (8, 90),
}

ROWS = 180
START_DATE = date(2025, 1, 1)


def main() -> None:
    """Generate the dataset and write it to ``data/sales_data.csv``."""

    rng = random.Random(42)
    out_path = Path(__file__).resolve().parent.parent / "data" / "sales_data.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["date", "product_category", "region", "units_sold", "revenue"]
        )
        for i in range(ROWS):
            sale_date = START_DATE + timedelta(days=rng.randint(0, 364))
            category = rng.choice(CATEGORIES)
            region = rng.choice(REGIONS)
            units = rng.randint(1, 50)
            low, high = PRICE_RANGE[category]
            unit_price = rng.uniform(low, high)
            revenue = round(units * unit_price, 2)
            writer.writerow(
                [sale_date.isoformat(), category, region, units, revenue]
            )

    print(f"Wrote {ROWS} rows to {out_path}")


if __name__ == "__main__":
    main()
