from __future__ import annotations

import csv
import io
import json

import pytest

from code_churn.decay import DecayFit, GroupFit, LRTResult, StratifiedDecay
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
    # Author-identity defaults are present (and null/empty) in the config.
    assert payload["config"]["mailmap"] is None
    assert payload["config"]["exclude_authors"] == []


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


def stratified_report(lrt: LRTResult | None) -> RepoReport:
    report = sample_report()
    alice_fit = DecayFit(
        lam=1 / (20 * 86_400),
        half_life_seconds=20 * 86_400 * 0.6931,
        events=1,
        censored=1,
        exposure_seconds=20 * 86_400,
        caveats=("short history: oldest observation is 10.0 days",),
    )
    report.stratified_decay = StratifiedDecay(
        group_by="author",
        groups={
            ALICE: GroupFit(
                events=1,
                censored=1,
                exposure_seconds=20 * 86_400,
                fit=alice_fit,
            ),
            BOB: GroupFit(events=0, censored=1, exposure_seconds=86_400, fit=None),
        },
        lrt=lrt,
    )
    return report


def test_table_stratified_section_and_lrt_line() -> None:
    lrt = LRTResult(statistic=6.21, df=1, p_value=0.0127)
    out = render_table(stratified_report(lrt), ["summary"])
    assert "decay by author:" in out
    alice_row = next(line for line in out.splitlines() if line.startswith("alice"))
    assert "13.9 days" in alice_row
    bob_row = next(line for line in out.splitlines() if line.startswith("bob"))
    assert "-" in bob_row  # no fit for the zero-death group
    assert "caveat (alice <alice@example.com>): short history" in out
    assert (
        "likelihood-ratio test (shared λ vs per-group λ): "
        "χ²=6.21, df=1, p=0.0127" in out
    )


def test_table_stratified_tiny_p_and_missing_lrt() -> None:
    tiny = render_table(
        stratified_report(LRTResult(statistic=30.0, df=2, p_value=0.0002)),
        ["summary"],
    )
    assert "p<0.001" in tiny
    missing = render_table(stratified_report(None), ["summary"])
    assert "likelihood-ratio test: - (needs at least two groups with data)" in missing


def test_json_stratified_payload() -> None:
    lrt = LRTResult(statistic=6.21, df=1, p_value=0.0127)
    payload = json.loads(render_json(stratified_report(lrt)))
    stratified = payload["stratified_decay"]
    assert stratified["group_by"] == "author"
    # Sorted by (-events, label): alice's fitted group first.
    groups = stratified["groups"]
    assert [g["group"] for g in groups] == [ALICE, BOB]
    assert groups[0]["events"] == 1
    assert groups[0]["half_life_days"] == pytest.approx(20 * 0.6931)
    assert groups[0]["caveats"] == ["short history: oldest observation is 10.0 days"]
    assert groups[1]["half_life_days"] is None
    assert groups[1]["lambda_per_day"] is None
    assert groups[1]["exposure_days"] == 1.0
    assert stratified["lrt"] == {"statistic": 6.21, "df": 1, "p_value": 0.0127}


def test_json_stratified_null_lrt_and_absent_key() -> None:
    payload = json.loads(render_json(stratified_report(None)))
    assert payload["stratified_decay"]["lrt"] is None
    assert "stratified_decay" not in json.loads(render_json(sample_report()))


def test_empty_report_renders() -> None:
    report = sample_report()
    report.files = {}
    report.authors = {}
    report.events = []
    report.totals = GroupStats()
    out = render_table(report, ["summary", "file", "author"])
    assert "churn rate:    -" in out  # no division by zero
    json.loads(render_json(report))
