"""Monthly churn-rate table across repositories, 2016-01 .. 2026-09.

One first-parent walk per repo using the same OriginMap machinery as the
code-churn engine (identical move/whitespace/first-parent semantics, no
excludes). Lines are bucketed by the month they were ADDED; a churn event
within the 30-day window increments its ORIGIN month. Cell value is that
cohort's churn rate (churned / added), empty when nothing was added.

Usage: python analysis/monthly_churn.py OUT.csv REPO_PATH [REPO_PATH ...]

Also writes OUT_added.csv and OUT_churned.csv with the raw counts. The
output is written after every repo, so partial results survive interruption.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from code_churn.gitwalk import open_repo, walk_first_parent  # noqa: E402
from code_churn.model import LineOrigin  # noqa: E402
from code_churn.origins import OriginMap  # noqa: E402

WINDOW_SECONDS = 30 * 86_400
FIRST_MONTH = (2016, 1)
LAST_MONTH = (2026, 9)  # inclusive
SINCE = int(datetime(2016, 1, 1, tzinfo=timezone.utc).timestamp())
UNTIL = int(datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp())  # exclusive


def month_key(epoch: int) -> str:
    moment = datetime.fromtimestamp(epoch, tz=timezone.utc)
    return f"{moment.year:04d}-{moment.month:02d}"


def month_columns() -> list[str]:
    columns = []
    year, month = FIRST_MONTH
    while (year, month) <= LAST_MONTH:
        columns.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return columns


def analyze_monthly(repo_path: str) -> tuple[Counter[str], Counter[str]]:
    repo = open_repo(repo_path)
    origin_map = OriginMap()
    added: Counter[str] = Counter()
    churned: Counter[str] = Counter()
    walked = 0
    for commit in walk_first_parent(repo, "HEAD"):
        walked += 1
        if walked % 5000 == 0:
            print(f"    {walked} commits...", file=sys.stderr, flush=True)
        in_range = SINCE <= commit.time < UNTIL
        fresh = LineOrigin(
            commit_id=commit.commit_id,
            path="",  # stamped per file by apply_commit
            time=commit.time,
            author=commit.author,
            in_range=in_range,
        )
        outcome = origin_map.apply_commit(list(commit.patches), fresh)
        if in_range:
            added[month_key(commit.time)] += sum(outcome.fresh_added.values())
        for cand in outcome.candidates:
            if (
                cand.origin.in_range
                and commit.time - cand.origin.time <= WINDOW_SECONDS
            ):
                churned[month_key(cand.origin.time)] += 1
    print(f"    done: {walked} first-parent commits", file=sys.stderr, flush=True)
    return added, churned


def write_tables(
    out: Path,
    columns: list[str],
    rows: list[tuple[str, Counter[str], Counter[str]]],
) -> None:
    def dump(
        path: Path, cell: Callable[[Counter[str], Counter[str], str], str]
    ) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["repo", *columns])
            for name, added, churned in rows:
                writer.writerow(
                    [name, *(cell(added, churned, month) for month in columns)]
                )

    def rate(added: Counter[str], churned: Counter[str], month: str) -> str:
        count = added.get(month, 0)
        return f"{churned.get(month, 0) / count:.4f}" if count else ""

    dump(out, rate)
    dump(out.with_name(out.stem + "_added.csv"), lambda a, c, m: str(a.get(m, 0)))
    dump(out.with_name(out.stem + "_churned.csv"), lambda a, c, m: str(c.get(m, 0)))


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    out = Path(argv[0])
    columns = month_columns()
    rows: list[tuple[str, Counter[str], Counter[str]]] = []
    for repo_path in argv[1:]:
        name = Path(repo_path).name
        print(f"analyzing {name}...", file=sys.stderr, flush=True)
        added, churned = analyze_monthly(repo_path)
        rows.append((name, added, churned))
        write_tables(out, columns, rows)  # incremental: survive interruption
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
