"""View a monthly churn-rate table (repo rows x month columns) as a live
matplotlib heatmap.

Usage: python analysis/plot_churn.py [CSV_PATH]
Defaults to analysis/churn-monthly.csv next to this script.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main(argv: list[str]) -> int:
    default = Path(__file__).resolve().parent / "churn-monthly.csv"
    path = Path(argv[0]) if argv else default

    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        months = header[1:]
        repos: list[str] = []
        grid: list[list[float]] = []
        for row in reader:
            repos.append(row[0])
            grid.append([float(cell) if cell else math.nan for cell in row[1:]])

    data = np.ma.masked_invalid(np.array(grid))
    figure, axes = plt.subplots(figsize=(14, 1.0 * len(repos) + 2.5))
    colormap = plt.get_cmap("viridis").copy()
    colormap.set_bad("#d0d0d0")  # months with no added lines
    image = axes.imshow(data, aspect="auto", cmap=colormap, vmin=0.0)

    axes.set_yticks(range(len(repos)), repos)
    year_starts = [i for i, month in enumerate(months) if month.endswith("-01")]
    axes.set_xticks(year_starts, [months[i][:4] for i in year_starts])
    axes.set_xlabel("month the lines were added")
    axes.set_title("30-day code churn rate by repository, monthly cohorts")
    figure.colorbar(image, ax=axes, label="churn rate (lines churned / lines added)")

    figure.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
