"""Aggregate raw engine output into the report structure."""

from __future__ import annotations

from code_churn.decay import DecayFit, StratifiedDecay
from code_churn.model import (
    AnalysisConfig,
    AnalysisResult,
    AuthorStats,
    ChurnEvent,
    GroupStats,
    RepoReport,
)


def churn_matrix(events: list[ChurnEvent]) -> dict[tuple[str, str], int]:
    """(origin_author, churn_author) -> count of A's lines churned by B."""
    matrix: dict[tuple[str, str], int] = {}
    for event in events:
        key = (event.origin_author, event.churn_author)
        matrix[key] = matrix.get(key, 0) + 1
    return matrix


def build_report(
    config: AnalysisConfig,
    result: AnalysisResult,
    generated_at: int,
    decay: DecayFit | None = None,
    stratified: StratifiedDecay | None = None,
) -> RepoReport:
    totals = GroupStats()
    files: dict[str, GroupStats] = {}
    authors: dict[str, AuthorStats] = {}

    for (path, author), count in result.added.items():
        totals.added += count
        files.setdefault(path, GroupStats()).added += count
        authors.setdefault(author, AuthorStats()).received.added += count

    for event in result.events:
        totals.churned += 1
        files.setdefault(event.origin_path, GroupStats()).churned += 1
        received = authors.setdefault(event.origin_author, AuthorStats())
        received.received.churned += 1
        authors.setdefault(event.churn_author, AuthorStats()).caused += 1

    return RepoReport(
        config=config,
        generated_at=generated_at,
        totals=totals,
        files=files,
        authors=authors,
        events=list(result.events),
        decay=decay,
        stratified_decay=stratified,
    )
