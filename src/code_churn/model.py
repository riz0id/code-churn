"""Core data types shared by the engine, reports, and formatters.

This module must not import pygit2: everything downstream of the engine
(aggregation, formatting, tests for those) works on these plain types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from code_churn.decay import DecayFit, StratifiedDecay, SurvivalData

ChurnKind = Literal["modified", "deleted"]
GroupBy = Literal["author", "filetype"]


def author_name(author: str) -> str:
    """Name part of 'Name <email>'. A name containing ' <' truncates at the
    first occurrence; display (plot) and --exclude-author share this rule."""
    return author.split(" <", 1)[0]


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
    # Lifetimes for decay fitting; only populated when the config asks for it.
    survival: SurvivalData | None = None
    # Same lifetimes keyed by group label; only populated when the config
    # sets survival_group_by (alongside collect_survival).
    survival_by_group: dict[str, SurvivalData] | None = None
    # (birth_time, death_time | None) per in-range, non-excluded line; only
    # populated when the config sets collect_lifespans. None death = alive
    # at the end of the walk.
    lifespans: list[tuple[int, int | None]] | None = None
    # Time of the newest walked commit; set alongside lifespans (censoring
    # ages at any cutoff need it, and it cannot be derived from lifespans
    # alone when the newest commits touch no tracked lines).
    observation_end: int | None = None


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    repo_path: str
    rev: str
    window_seconds: int
    since: int | None  # epoch seconds, inclusive
    until: int | None  # epoch seconds, inclusive
    # Gitignore-style globs (pathlib full-match: `**` recurses, `*` does not
    # cross `/`) tested against a line's add-time path. Matching lines count
    # neither as added nor as churned.
    exclude: tuple[str, ...] = ()
    # Contents (text) of an external mailmap in standard git mailmap format,
    # applied by gitwalk after the repo's own mailmap: external entries match
    # the repo-resolved identity and take effect last (see README). The CLI
    # reads the file so gitwalk stays free of filesystem I/O.
    mailmap: str | None = None
    # The CLI path the mailmap text came from; echoed in output only.
    mailmap_path: str | None = None
    # Author names (the part of "Name <email>" before " <", post-mailmap),
    # matched exactly and case-sensitively. Excluded in both roles: their
    # lines count neither as added nor churned, and churn they perform is
    # dropped entirely (the victim line's death goes unrecorded — neither
    # event nor censored observation, so its exposure leaves --fit-decay).
    exclude_authors: tuple[str, ...] = ()
    # Collect survival data (lifetimes of churned and surviving in-range
    # lines) for decay fitting. Ignores the window: the full history is
    # walked so churn past `until + window` is observed, not miscounted as
    # survival.
    collect_survival: bool = False
    # Additionally label survival data per group for stratified decay fits.
    # Only takes effect when collect_survival is set.
    survival_group_by: GroupBy | None = None
    # Collect per-line (birth, death | None) pairs so the decay fit can be
    # recomputed at any historical cutoff (--churn-timeline). Like
    # collect_survival, this ignores the window and walks the full history.
    collect_lifespans: bool = False


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
    # Fitted decay function; None when fitting was not requested (see
    # config.collect_survival) or produced no fit (no churn events).
    decay: DecayFit | None = None
    # Per-group fits plus the shared-rate likelihood-ratio test; None when
    # no grouping was requested (see config.survival_group_by).
    stratified_decay: StratifiedDecay | None = None

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
