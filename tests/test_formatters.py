from __future__ import annotations

import csv
import io
import json

from code_churn.formatters import render_csv, render_json, render_table
from code_churn.model import (
    AnalysisConfig,
    AuthorStats,
    ChurnEvent,
    GroupStats,
    RepoReport,
)

ALICE = "alice <alice@example.com>"
BOB = "bob <bob@example.com>"


def sample_report() -> RepoReport:
    event = ChurnEvent(
        path="a.py",
        origin_path="a.py",
        kind="modified",
        origin_commit="c1",
        origin_author=ALICE,
        origin_time=1_700_000_000,
        churn_commit="c2",
        churn_author=BOB,
        churn_time=1_700_100_000,
    )
    return RepoReport(
        config=AnalysisConfig(
            repo_path=".",
            rev="HEAD",
            window_seconds=30 * 86_400,
            since=None,
            until=None,
        ),
        generated_at=1_700_200_000,
        totals=GroupStats(added=10, churned=1),
        files={
            "a.py": GroupStats(added=6, churned=1),
            "b.py": GroupStats(added=4, churned=0),
        },
        authors={
            ALICE: AuthorStats(received=GroupStats(added=10, churned=1), caused=0),
            BOB: AuthorStats(received=GroupStats(), caused=1),
        },
        events=[event],
    )


def test_table_contains_all_sections_and_rates() -> None:
    out = render_table(sample_report(), ["summary", "file", "author"])
    assert "churn rate:    10.0%" in out
    assert "per file:" in out
    assert "a.py" in out and "b.py" in out
    assert "per author:" in out
    assert ALICE in out and BOB in out
    # bob caused the only churn event: 100% caused share.
    bob_row = next(line for line in out.splitlines() if line.startswith("bob"))
    assert "100.0%" in bob_row


def test_table_single_section() -> None:
    out = render_table(sample_report(), ["file"])
    assert "per file:" in out
    assert "per author:" not in out
    assert "churn rate:    " not in out  # summary omitted


def test_json_round_trips_and_has_events() -> None:
    payload = json.loads(render_json(sample_report()))
    assert payload["summary"] == {"added": 10, "churned": 1, "churn_rate": 0.1}
    assert payload["files"][0]["path"] == "a.py"  # sorted: churned first
    authors = {entry["author"]: entry for entry in payload["authors"]}
    assert authors[ALICE]["churn_rate"] == 0.1
    assert authors[ALICE]["caused"] == 0
    assert authors[BOB]["caused"] == 1
    assert authors[BOB]["caused_share"] == 1.0
    assert authors[BOB]["churn_rate"] is None  # added nothing
    assert payload["events"][0]["origin_commit"] == "c1"
    assert payload["events"][0]["kind"] == "modified"


def test_csv_file_section() -> None:
    rows = list(csv.reader(io.StringIO(render_csv(sample_report(), "file"))))
    assert rows[0] == ["path", "added", "churned", "churn_rate"]
    assert rows[1][0] == "a.py"
    assert rows[2] == ["b.py", "4", "0", "0.0"]


def test_csv_author_and_summary_sections() -> None:
    author_rows = list(csv.reader(io.StringIO(render_csv(sample_report(), "author"))))
    assert author_rows[0][0] == "author"
    assert author_rows[1][0] == ALICE
    summary_rows = list(csv.reader(io.StringIO(render_csv(sample_report(), "summary"))))
    assert summary_rows[1] == ["10", "1", "0.1"]


def test_empty_report_renders() -> None:
    report = sample_report()
    report.files = {}
    report.authors = {}
    report.events = []
    report.totals = GroupStats()
    out = render_table(report, ["summary", "file", "author"])
    assert "churn rate:    -" in out  # no division by zero
    json.loads(render_json(report))
