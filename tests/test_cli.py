from __future__ import annotations

import json
import math
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


def test_exclude_flag_filters_and_echoes_config(
    repo: RepoBuilder, capsys: pytest.CaptureFixture[str]
) -> None:
    repo.commit(
        {"gen/out.c": "generated();\n", "a.py": "alpha = 1\n"},
        when=day(0),
        author="alice",
    )
    repo.commit(
        {"gen/out.c": "regenerated();\n", "a.py": "alpha = 2\n"},
        when=day(10),
        author="bob",
    )
    code = main(
        [
            str(repo.path),
            "--exclude",
            "gen/**",
            "--exclude",
            "**/*.ipynb",
            "--format",
            "json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["config"]["exclude"] == ["gen/**", "**/*.ipynb"]
    assert payload["summary"] == {"added": 2, "churned": 1, "churn_rate": 0.5}
    assert [f["path"] for f in payload["files"]] == ["a.py"]


def test_exclude_patterns_listed_in_table_summary(
    churned_repo: RepoBuilder, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(churned_repo.path), "--exclude", "vendor/**"]) == 0
    assert "exclude: vendor/**" in capsys.readouterr().out


def test_invalid_exclude_pattern_rejected(
    churned_repo: RepoBuilder, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(churned_repo.path), "--exclude", ""]) == 2
    assert "invalid --exclude pattern" in capsys.readouterr().err


def test_mailmap_flag_joins_authors_and_echoes_config(
    churned_repo: RepoBuilder, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mailmap = tmp_path / "team.mailmap"
    mailmap.write_text("alice <alice@example.com> <bob@example.com>\n")
    code = main([str(churned_repo.path), "--mailmap", str(mailmap), "--format", "json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["config"]["mailmap"] == str(mailmap)
    assert [a["author"] for a in payload["authors"]] == ["alice <alice@example.com>"]
    assert payload["events"][0]["churn_author"] == "alice <alice@example.com>"


def test_mailmap_unreadable_file_rejected(
    churned_repo: RepoBuilder, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main([str(churned_repo.path), "--mailmap", str(tmp_path / "missing.map")])
    assert code == 2
    assert "cannot read --mailmap" in capsys.readouterr().err


def test_exclude_author_filters_and_echoes_config(
    churned_repo: RepoBuilder, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main([str(churned_repo.path), "--exclude-author", "bob", "--format", "json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["config"]["exclude_authors"] == ["bob"]
    # bob's churn of alice's line and his replacement line are both dropped.
    assert payload["summary"] == {"added": 2, "churned": 0, "churn_rate": 0.0}
    assert [a["author"] for a in payload["authors"]] == ["alice <alice@example.com>"]


def test_exclude_author_listed_in_table_summary(
    churned_repo: RepoBuilder, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(churned_repo.path), "--exclude-author", "bob"]) == 0
    assert "exclude authors: bob" in capsys.readouterr().out


def test_invalid_exclude_author_rejected(
    churned_repo: RepoBuilder, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(churned_repo.path), "--exclude-author", ""]) == 2
    assert "invalid --exclude-author" in capsys.readouterr().err


def test_exclude_author_empties_heatmap(
    churned_repo: RepoBuilder, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "churn.png"
    code = main(
        [
            str(churned_repo.path),
            "--exclude-author",
            "bob",
            "--churn-heatmap",
            str(target),
        ]
    )
    assert code == 1
    assert "no churn events" in capsys.readouterr().err


def test_bad_range_rejected(
    churned_repo: RepoBuilder, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [str(churned_repo.path), "--since", "2026-02-01", "--until", "2026-01-01"]
    )
    assert code == 2
    assert "--since is after --until" in capsys.readouterr().err


def test_fit_decay_in_table_summary(
    churned_repo: RepoBuilder, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(churned_repo.path), "--fit-decay"]) == 0
    assert "fitted half-life:" in capsys.readouterr().out


def test_fit_decay_json_exact_half_life(
    churned_repo: RepoBuilder, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(churned_repo.path), "--fit-decay", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    decay = payload["decay"]
    # One death at 10 days; survivors: alpha (10 days old) and the
    # replacement (0 days old). lambda = 1/20 per day -> half-life 20*ln2.
    assert decay["events"] == 1
    assert decay["censored"] == 2
    assert decay["exposure_days"] == pytest.approx(20)
    assert decay["half_life_days"] == pytest.approx(20 * math.log(2))
    assert decay["lambda_per_day"] == pytest.approx(1 / 20)


def test_json_has_no_decay_key_unless_requested(
    churned_repo: RepoBuilder, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(churned_repo.path), "--format", "json"]) == 0
    assert "decay" not in json.loads(capsys.readouterr().out)


def test_decay_plot_writes_png_and_svg(
    churned_repo: RepoBuilder, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    png = tmp_path / "decay.png"
    assert main([str(churned_repo.path), "--decay-plot", str(png)]) == 0
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    # --decay-plot implies --fit-decay in the text output.
    assert "fitted half-life:" in capsys.readouterr().out

    svg = tmp_path / "decay.svg"
    assert main([str(churned_repo.path), "--decay-plot", str(svg)]) == 0
    assert "<svg" in svg.read_text()


def test_decay_plot_bad_extension_rejected(
    churned_repo: RepoBuilder, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main([str(churned_repo.path), "--decay-plot", str(tmp_path / "out.bmp")])
    assert code == 2
    assert "--decay-plot" in capsys.readouterr().err


@pytest.fixture
def two_author_repo(repo: RepoBuilder) -> RepoBuilder:
    repo.commit({"a.py": "alpha = 1\n"}, when=day(0), author="alice")
    repo.commit({"b.py": "bravo = 1\n"}, when=day(0), author="bob")
    repo.commit({"a.py": "alpha = 2\n"}, when=day(10), author="bob")
    return repo


def test_group_by_implies_fit_and_reports_lrt(
    two_author_repo: RepoBuilder, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(two_author_repo.path), "--group-by", "author"]) == 0
    out = capsys.readouterr().out
    assert "fitted half-life:" in out  # --group-by implies --fit-decay
    assert "decay by author:" in out
    assert "likelihood-ratio test (shared λ vs per-group λ):" in out


def test_group_by_json_exact(
    two_author_repo: RepoBuilder, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main([str(two_author_repo.path), "--group-by", "author", "--format", "json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    stratified = payload["stratified_decay"]
    assert stratified["group_by"] == "author"
    groups = {g["group"]: g for g in stratified["groups"]}
    # Alice: her only line died at day 10 -> lambda = 1/10 per day.
    alice = groups["alice <alice@example.com>"]
    assert alice["events"] == 1
    assert alice["exposure_days"] == pytest.approx(10)
    assert alice["lambda_per_day"] == pytest.approx(1 / 10)
    assert alice["half_life_days"] == pytest.approx(10 * math.log(2))
    # Bob: no deaths (b.py survives 10 days, the replacement 0 days).
    bob = groups["bob <bob@example.com>"]
    assert bob["events"] == 0
    assert bob["exposure_days"] == pytest.approx(10)
    assert bob["half_life_days"] is None
    # D=1, E=20d vs alice's d=1, E=10d: stat = 2*ln(2), df = 1, and for
    # df=1 the p-value is erfc(sqrt(stat/2)).
    lrt = stratified["lrt"]
    assert lrt["statistic"] == pytest.approx(2 * math.log(2))
    assert lrt["df"] == 1
    assert lrt["p_value"] == pytest.approx(math.erfc(math.sqrt(math.log(2))))


def test_churn_heatmap_writes_png(
    churned_repo: RepoBuilder, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    png = tmp_path / "heatmap.png"
    assert main([str(churned_repo.path), "--churn-heatmap", str(png)]) == 0
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_churn_heatmap_bad_target_rejected(
    churned_repo: RepoBuilder, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main([str(churned_repo.path), "--churn-heatmap", str(tmp_path / "out.bmp")])
    assert code == 2
    assert "--churn-heatmap" in capsys.readouterr().err


def test_churn_heatmap_no_events_fails_cleanly(
    repo: RepoBuilder, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo.commit({"a.py": "alpha = 1\n"}, when=day(0))
    code = main([str(repo.path), "--churn-heatmap", str(tmp_path / "out.png")])
    assert code == 1
    assert "no churn events" in capsys.readouterr().err


def test_churn_heatmap_live_calls_show(
    churned_repo: RepoBuilder, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Pin a non-interactive backend so no real window opens, and record the
    # show() call the live path must make.
    monkeypatch.setenv("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt

    calls: list[bool] = []
    monkeypatch.setattr(plt, "show", lambda: calls.append(True))
    assert main([str(churned_repo.path), "--churn-heatmap", "live"]) == 0
    assert calls == [True]


@pytest.fixture
def timeline_repo(repo: RepoBuilder) -> RepoBuilder:
    # Three calendar months of history (T0 = 2023-11-14) with churn in the
    # second and third, keeping the GIF tiny (3 frames).
    repo.commit({"a.py": "alpha = 1\nbeta = 2\n"}, when=day(0), author="alice")
    repo.commit({"a.py": "alpha = 9\nbeta = 2\n"}, when=day(35), author="bob")
    repo.commit({"a.py": "alpha = 9\nbeta = 99\n"}, when=day(65), author="bob")
    return repo


def test_churn_timeline_writes_gif(timeline_repo: RepoBuilder, tmp_path: Path) -> None:
    gif = tmp_path / "timeline.gif"
    assert main([str(timeline_repo.path), "--churn-timeline", str(gif)]) == 0
    assert gif.read_bytes()[:4] == b"GIF8"


def test_churn_timeline_bad_target_rejected(
    churned_repo: RepoBuilder, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # .png is valid for the other plot flags but not for an animation.
    for name in ("out.bmp", "out.png"):
        code = main([str(churned_repo.path), "--churn-timeline", str(tmp_path / name)])
        assert code == 2
        assert "--churn-timeline" in capsys.readouterr().err


def test_churn_timeline_no_data_fails_cleanly(
    churned_repo: RepoBuilder, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            str(churned_repo.path),
            "--since",
            "2030-01-01",
            "--churn-timeline",
            str(tmp_path / "out.gif"),
        ]
    )
    assert code == 1
    assert "nothing to animate" in capsys.readouterr().err


def test_churn_timeline_live_calls_show(
    timeline_repo: RepoBuilder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt

    calls: list[bool] = []
    monkeypatch.setattr(plt, "show", lambda: calls.append(True))
    assert main([str(timeline_repo.path), "--churn-timeline", "live"]) == 0
    assert calls == [True]
