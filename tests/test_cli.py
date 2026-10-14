from __future__ import annotations

import json
from pathlib import Path

import pytest

from code_churn.cli import main, parse_date, parse_duration
from tests.conftest import RepoBuilder, day


@pytest.fixture
def churned_repo(repo: RepoBuilder) -> RepoBuilder:
    repo.commit({"a.py": "alpha = 1\nbeta = 2\n"}, when=day(0), author="alice")
    repo.commit({"a.py": "alpha = 1\nbeta = 99\n"}, when=day(10), author="bob")
    return repo


def test_parse_duration() -> None:
    assert parse_duration("30d") == 30 * 86_400
    assert parse_duration("2w") == 14 * 86_400
    assert parse_duration("12h") == 12 * 3_600
    with pytest.raises(Exception):
        parse_duration("30x")


def test_parse_date_utc() -> None:
    assert parse_date("1970-01-02") == 86_400


def test_table_output(
    churned_repo: RepoBuilder, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(churned_repo.path)]) == 0
    out = capsys.readouterr().out
    assert "lines added:   3" in out
    assert "lines churned: 1" in out
    assert "a.py" in out
    assert "alice <alice@example.com>" in out


def test_json_output(
    churned_repo: RepoBuilder, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(churned_repo.path), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["churned"] == 1
    assert len(payload["events"]) == 1
    assert payload["events"][0]["churn_author"] == "bob <bob@example.com>"


def test_csv_output_to_file(churned_repo: RepoBuilder, tmp_path: Path) -> None:
    target = tmp_path / "out.csv"
    code = main(
        [
            str(churned_repo.path),
            "--format",
            "csv",
            "--by",
            "author",
            "--output",
            str(target),
        ]
    )
    assert code == 0
    assert target.read_text().startswith("author,added,churned")


def test_window_flag_excludes_late_churn(
    churned_repo: RepoBuilder, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(churned_repo.path), "--window", "5d"]) == 0
    assert "lines churned: 0" in capsys.readouterr().out


def test_not_a_repo_fails_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = tmp_path / "nothing"
    empty.mkdir()
    assert main([str(empty)]) == 1
    assert "not a git repository" in capsys.readouterr().err


def test_bad_range_rejected(
    churned_repo: RepoBuilder, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [str(churned_repo.path), "--since", "2026-02-01", "--until", "2026-01-01"]
    )
    assert code == 2
    assert "--since is after --until" in capsys.readouterr().err
