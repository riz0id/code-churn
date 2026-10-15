"""Render survival curves and churn heatmaps, to image files or a live window.

This is the only module that imports matplotlib, mirroring how gitwalk is the
only module importing pygit2. The CLI imports it lazily so plain runs never
pay the matplotlib import cost, and calls configure() once before rendering.
"""

from __future__ import annotations

import math

import matplotlib
import matplotlib.pyplot as plt

from code_churn.decay import DAY, DecayFit, SurvivalData, kaplan_meier

# Heatmap axes stay readable by truncating to the most-involved authors.
MAX_HEATMAP_AUTHORS = 20


def configure(live: bool) -> None:
    """Pick the backend once per run. File-only runs force Agg (headless-safe,
    matches the old unconditional behavior); live runs keep matplotlib's
    auto-selected interactive backend. savefig works on any backend, so a run
    mixing file and live output stays live."""
    if not live:
        matplotlib.use("Agg")


def is_interactive() -> bool:
    return "agg" not in matplotlib.get_backend().lower()


def render_decay_plot(data: SurvivalData, fit: DecayFit | None, path: str) -> None:
    points = kaplan_meier(data)
    max_age_days = (
        max(max(data.event_lifetimes, default=0), max(data.censored_ages, default=0))
        / DAY
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    observed = len(data.event_lifetimes) + len(data.censored_ages)
    xs = [t / DAY for t, _ in points] + [max_age_days]
    ys = [s for _, s in points] + [points[-1][1]]
    ax.step(
        xs,
        ys,
        where="post",
        label=f"observed survival (Kaplan–Meier, n={observed})",
    )

    if fit is not None:
        half_life_days = fit.half_life_seconds / DAY
        lam_per_day = fit.lam * DAY
        grid = [max_age_days * i / 200 for i in range(201)]
        ax.plot(
            grid,
            [math.exp(-lam_per_day * t) for t in grid],
            linestyle="-",
            label=(f"fitted exp(−λt), " f"half-life ≈ {half_life_days:.1f} days"),
        )
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
        if half_life_days <= max_age_days:
            ax.axvline(half_life_days, color="grey", linestyle="--", linewidth=0.8)

    ax.set_xlabel("line age (days)")
    ax.set_ylabel("fraction of lines surviving")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    ax.set_title("code churn decay")
    ax.legend(loc="best")
    if fit is not None and fit.caveats:
        fig.text(0.01, 0.01, "; ".join(fit.caveats), fontsize=7, color="grey")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def render_churn_heatmap(matrix: dict[tuple[str, str], int], path: str | None) -> None:
    """origin_author (rows) x churn_author (cols) churn counts.

    path=None shows a live interactive window instead of saving. Both axes use
    one author ordering (most-involved first), so the diagonal is self-churn.
    """
    involvement: dict[str, int] = {}
    for (origin, churner), count in matrix.items():
        involvement[origin] = involvement.get(origin, 0) + count
        involvement[churner] = involvement.get(churner, 0) + count
    ranked = sorted(involvement, key=lambda name: (-involvement[name], name))
    authors = ranked[:MAX_HEATMAP_AUTHORS]
    index = {name: i for i, name in enumerate(authors)}

    n = len(authors)
    grid = [[0] * n for _ in range(n)]
    for (origin, churner), count in matrix.items():
        row = index.get(origin)
        col = index.get(churner)
        if row is not None and col is not None:
            grid[row][col] = count

    fig, ax = plt.subplots(figsize=(max(6, 0.5 * n + 3), max(5, 0.5 * n + 2)))
    image = ax.imshow(grid, cmap="viridis")
    fig.colorbar(image, ax=ax, label="churn events")

    short_names = [name.split(" <", 1)[0] for name in authors]
    ax.set_xticks(range(n), labels=short_names, rotation=45, ha="right")
    ax.set_yticks(range(n), labels=short_names)
    ax.set_xlabel("churn author (who changed it)")
    ax.set_ylabel("origin author (whose code)")
    title = "cross-author churn"
    if len(ranked) > n:
        title += f" (top {n} of {len(ranked)} authors)"
    ax.set_title(title)

    if n <= 12:
        peak = max((count for row in grid for count in row), default=0)
        for i in range(n):
            for j in range(n):
                if grid[i][j]:
                    color = "white" if grid[i][j] < 0.6 * peak else "black"
                    ax.text(
                        j, i, str(grid[i][j]), ha="center", va="center", color=color
                    )

    fig.tight_layout()
    if path is None:
        plt.show()
    else:
        fig.savefig(path)
        plt.close(fig)
