"""Core data types shared by the engine, reports, and formatters.

This module must not import pygit2: everything downstream of the engine
(aggregation, formatting, tests for those) works on these plain types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ChurnKind = Literal["modified", "deleted"]


@dataclass(frozen=True, slots=True)
class LineOrigin:
    """Provenance of one line of a tracked file.

    ``in_range`` is False for lines whose origin commit falls outside the
    analysis range [since, until]; such lines keep correct positions in the
    origin map but never produce churn events or count as added lines.
    """

    commit_id: str
    path: str  # path of the file when the line was added (stats key)
    time: int  # origin commit's commit time, epoch seconds
    author: str  # "Name <email>", mailmap-resolved
    in_range: bool


@dataclass(frozen=True, slots=True)
class ChurnEvent:
    path: str  # path of the file at the time of the churning commit
    origin_path: str  # path of the file when the churned line was added
    kind: ChurnKind
    origin_commit: str
    origin_author: str
    origin_time: int
    churn_commit: str
    churn_author: str
    churn_time: int


@dataclass(slots=True)
class AnalysisResult:
    """Raw engine output, aggregated into a RepoReport by report.py."""

    events: list[ChurnEvent] = field(default_factory=list)
    # (add-time path, author) -> fresh lines added by in-range commits
    added: dict[tuple[str, str], int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    repo_path: str
    rev: str
    window_seconds: int
    since: int | None  # epoch seconds, inclusive
    until: int | None  # epoch seconds, inclusive


@dataclass(slots=True)
class GroupStats:
    added: int = 0
    churned: int = 0

    @property
    def churn_rate(self) -> float | None:
        if self.added == 0:
            return None
        return self.churned / self.added


@dataclass(slots=True)
class AuthorStats:
    """Both attribution directions, reported separately."""

    received: GroupStats = field(default_factory=GroupStats)
    caused: int = 0


@dataclass(slots=True)
class RepoReport:
    config: AnalysisConfig
    generated_at: int
    totals: GroupStats
    files: dict[str, GroupStats]  # keyed by path at the time lines were added
    authors: dict[str, AuthorStats]
    events: list[ChurnEvent]

    @property
    def total_events(self) -> int:
        return len(self.events)

    def caused_share(self, author: str) -> float | None:
        if not self.events:
            return None
        stats = self.authors.get(author)
        if stats is None:
            return None
        return stats.caused / len(self.events)
