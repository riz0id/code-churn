"""Aggregate raw engine output into the report structure."""

from __future__ import annotations

from code_churn.model import (
    AnalysisConfig,
    AnalysisResult,
    AuthorStats,
    GroupStats,
    RepoReport,
)


def build_report(
    config: AnalysisConfig, result: AnalysisResult, generated_at: int
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
    )
