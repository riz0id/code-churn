"""The churn pass: one linear walk driving the origin map.

For every first-parent commit we splice its diff into the OriginMap. Deleted
lines whose origin commit is in the analysis range and within the window are
churn; genuinely fresh lines from in-range commits feed the denominator.
"""

from __future__ import annotations

from code_churn.gitwalk import open_repo, walk_first_parent
from code_churn.model import AnalysisConfig, AnalysisResult, ChurnEvent, LineOrigin
from code_churn.origins import OriginMap


def analyze(config: AnalysisConfig) -> AnalysisResult:
    repo = open_repo(config.repo_path)
    origin_map = OriginMap()
    result = AnalysisResult()

    # Commits after until + window can neither churn in-window lines nor add
    # in-range lines; first-parent history is linear, so stopping is safe.
    # (Caveat: wildly non-monotonic commit times past this point are ignored.)
    stop_after = (
        config.until + config.window_seconds if config.until is not None else None
    )

    for commit in walk_first_parent(repo, config.rev):
        if stop_after is not None and commit.time > stop_after:
            break

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
                key = (path, commit.author)
                result.added[key] = result.added.get(key, 0) + count

        for cand in outcome.candidates:
            if not cand.origin.in_range:
                continue
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

    return result
