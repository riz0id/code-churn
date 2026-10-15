"""The churn pass: one linear walk driving the origin map.

For every first-parent commit we splice its diff into the OriginMap. Deleted
lines whose origin commit is in the analysis range and within the window are
churn; genuinely fresh lines from in-range commits feed the denominator.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from code_churn.decay import SurvivalData
from code_churn.gitwalk import open_repo, walk_first_parent
from code_churn.model import (
    AnalysisConfig,
    AnalysisResult,
    ChurnEvent,
    GroupBy,
    LineOrigin,
)
from code_churn.origins import OriginMap


def path_is_excluded(path: str, patterns: tuple[str, ...]) -> bool:
    """Gitignore-style match: `**` recurses, `*` stays within one segment."""
    return any(PurePosixPath(path).full_match(pattern) for pattern in patterns)


def survival_group_label(group_by: GroupBy, author: str, origin_path: str) -> str:
    """Stratum label for a line, from its add-time author and path."""
    if group_by == "author":
        return author
    suffix = PurePosixPath(origin_path).suffix.lower()
    return suffix if suffix else "(no extension)"


def analyze(config: AnalysisConfig) -> AnalysisResult:
    repo = open_repo(config.repo_path)
    origin_map = OriginMap()
    result = AnalysisResult()

    # Exclusion is decided by a line's add-time path. Excluded files stay
    # fully tracked in the origin map (cross-file moves must keep matching
    # across the exclusion boundary); they are only skipped at attribution
    # time, here. Distinct paths are few relative to lines, so cache verdicts.
    excluded_cache: dict[str, bool] = {}

    def excluded(path: str) -> bool:
        verdict = excluded_cache.get(path)
        if verdict is None:
            verdict = path_is_excluded(path, config.exclude)
            excluded_cache[path] = verdict
        return verdict

    # Commits after until + window can neither churn in-window lines nor add
    # in-range lines; first-parent history is linear, so stopping is safe.
    # (Caveat: wildly non-monotonic commit times past this point are ignored.)
    # Survival collection must see the whole history: churn landing past the
    # window is a death with a long lifetime, not a surviving line.
    stop_after = (
        config.until + config.window_seconds
        if config.until is not None and not config.collect_survival
        else None
    )

    event_lifetimes: list[int] = []
    group_by = config.survival_group_by
    events_by: dict[str, list[int]] = {}
    censored_by: dict[str, list[int]] = {}
    observation_end = 0  # time of the newest walked commit

    for commit in walk_first_parent(repo, config.rev):
        if stop_after is not None and commit.time > stop_after:
            break
        observation_end = max(observation_end, commit.time)

        in_range = (config.since is None or commit.time >= config.since) and (
            config.until is None or commit.time <= config.until
        )
        fresh = LineOrigin(
            commit_id=commit.commit_id,
            path="",  # stamped per file by apply_commit
            time=commit.time,
            author=commit.author,
            in_range=in_range,
        )
        outcome = origin_map.apply_commit(list(commit.patches), fresh)

        if in_range:
            for path, count in outcome.fresh_added.items():
                if excluded(path):
                    continue
                key = (path, commit.author)
                result.added[key] = result.added.get(key, 0) + count

        for cand in outcome.candidates:
            if not cand.origin.in_range:
                continue
            if excluded(cand.origin.path):
                continue
            if config.collect_survival:
                # Clamp guards non-monotonic commit clocks.
                lifetime = max(0, commit.time - cand.origin.time)
                event_lifetimes.append(lifetime)
                if group_by is not None:
                    label = survival_group_label(
                        group_by, cand.origin.author, cand.origin.path
                    )
                    events_by.setdefault(label, []).append(lifetime)
            # Boundary is inclusive: churn at exactly `window` after the add
            # still counts.
            if commit.time - cand.origin.time <= config.window_seconds:
                result.events.append(
                    ChurnEvent(
                        path=cand.path,
                        origin_path=cand.origin.path,
                        kind=cand.kind,
                        origin_commit=cand.origin.commit_id,
                        origin_author=cand.origin.author,
                        origin_time=cand.origin.time,
                        churn_commit=commit.commit_id,
                        churn_author=commit.author,
                        churn_time=commit.time,
                    )
                )

    if config.collect_survival:
        # Lines still alive at the end of the walk are right-censored
        # observations: they survived at least their current age.
        censored_ages: list[int] = []
        for line in origin_map.all_lines():
            if not line.in_range or excluded(line.path):
                continue
            age = max(0, observation_end - line.time)
            censored_ages.append(age)
            if group_by is not None:
                label = survival_group_label(group_by, line.author, line.path)
                censored_by.setdefault(label, []).append(age)
        result.survival = SurvivalData(
            event_lifetimes=event_lifetimes, censored_ages=censored_ages
        )
        if group_by is not None:
            result.survival_by_group = {
                label: SurvivalData(
                    event_lifetimes=events_by.get(label, []),
                    censored_ages=censored_by.get(label, []),
                )
                for label in sorted(events_by.keys() | censored_by.keys())
            }

    return result
