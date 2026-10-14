"""End-to-end engine scenarios on real (tiny, deterministic) git repos."""

from __future__ import annotations

from collections.abc import Callable

from code_churn.engine import analyze
from code_churn.model import AnalysisConfig, AnalysisResult
from tests.conftest import DAY, RepoBuilder, day

WINDOW_DAYS = 30


def run(
    repo: RepoBuilder,
    *,
    window_days: int = WINDOW_DAYS,
    since: int | None = None,
    until: int | None = None,
) -> AnalysisResult:
    config = AnalysisConfig(
        repo_path=str(repo.path),
        rev="HEAD",
        window_seconds=window_days * DAY,
        since=since,
        until=until,
    )
    return analyze(config)


def test_basic_churn_within_window(repo: RepoBuilder) -> None:
    c1 = repo.commit({"a.py": "alpha = 1\nbeta = 2\n"}, when=day(0))
    c2 = repo.commit({"a.py": "alpha = 1\nbeta = 99\n"}, when=day(10))
    result = run(repo)
    assert len(result.events) == 1
    event = result.events[0]
    assert event.origin_commit == c1
    assert event.churn_commit == c2
    assert event.kind == "modified"
    assert event.path == "a.py"
    assert event.origin_path == "a.py"
    # 2 initial lines + 1 replacement line
    assert sum(result.added.values()) == 3


def test_churn_at_exact_window_boundary_counts(repo: RepoBuilder) -> None:
    repo.commit({"a.py": "alpha = 1\n"}, when=day(0))
    repo.commit({"a.py": "alpha = 2\n"}, when=day(0) + WINDOW_DAYS * DAY)
    assert len(run(repo).events) == 1


def test_churn_one_second_past_window_does_not_count(repo: RepoBuilder) -> None:
    repo.commit({"a.py": "alpha = 1\n"}, when=day(0))
    repo.commit({"a.py": "alpha = 2\n"}, when=day(0) + WINDOW_DAYS * DAY + 1)
    assert run(repo).events == []


def test_modification_outside_window_is_not_churn(repo: RepoBuilder) -> None:
    repo.commit({"a.py": "alpha = 1\n"}, when=day(0))
    repo.commit({"a.py": "alpha = 2\n"}, when=day(40))
    assert run(repo).events == []


def test_whitespace_only_change_is_not_churn_and_preserves_origin(
    repo: RepoBuilder,
) -> None:
    c1 = repo.commit({"a.py": "value = compute()\n"}, when=day(0))
    repo.commit({"a.py": "    value =   compute()\n"}, when=day(5))
    c3 = repo.commit({"a.py": "    value = compute_else()\n"}, when=day(10))
    result = run(repo)
    assert len(result.events) == 1
    # The real edit blames the original commit, not the whitespace one.
    assert result.events[0].origin_commit == c1
    assert result.events[0].churn_commit == c3
    # The whitespace-only commit added nothing.
    assert sum(result.added.values()) == 2  # c1's line + c3's replacement


def test_moved_line_is_not_churn_and_keeps_original_clock(
    repo: RepoBuilder,
) -> None:
    c1 = repo.commit({"a.py": "moved_function_call()\nother_line = 1\n"}, when=day(0))
    # Move the line to another file (delete + identical add, same commit).
    repo.commit(
        {"a.py": "other_line = 1\n", "b.py": "moved_function_call()\n"},
        when=day(5),
    )
    result = run(repo)
    assert result.events == []

    # A later edit of the moved line churns against the ORIGINAL commit...
    c3 = repo.commit({"b.py": "changed_function_call()\n"}, when=day(10))
    result = run(repo)
    assert len(result.events) == 1
    event = result.events[0]
    assert event.origin_commit == c1
    assert event.churn_commit == c3
    assert event.path == "b.py"  # where the churn happened
    assert event.origin_path == "a.py"  # where the line was born
    # ...and the move itself never inflated the denominator.
    assert sum(result.added.values()) == 3  # c1: 2 lines, c3: 1 line


def test_moved_line_window_runs_from_original_add(repo: RepoBuilder) -> None:
    repo.commit({"a.py": "moved_function_call()\n"}, when=day(0))
    repo.commit({"b.py": "moved_function_call()\n", "a.py": None}, when=day(5))
    # Past the window of the ORIGINAL add (day 0), even though the move was
    # day 5: no churn.
    repo.commit({"b.py": "changed()\n"}, when=day(31))
    assert run(repo).events == []


def test_trivial_line_is_not_a_move(repo: RepoBuilder) -> None:
    repo.commit({"a.py": "if x:\n    y()\n}\n"}, when=day(0))
    # Delete the `}` from a.py, add an unrelated `}` in b.py, same commit.
    repo.commit({"a.py": "if x:\n    y()\n", "b.py": "}\n"}, when=day(5))
    result = run(repo)
    assert len(result.events) == 1  # the deleted `}` is genuine churn
    assert result.events[0].kind == "deleted"


def test_rename_followed_only_edited_line_churns(repo: RepoBuilder) -> None:
    content = "def f() -> int:\n    return compute_value()\n\nCONST = 12345\n"
    c1 = repo.commit({"old.py": content}, when=day(0))
    renamed = content.replace("12345", "54321")
    c2 = repo.commit({"old.py": None, "new.py": renamed}, when=day(10))
    result = run(repo)
    assert len(result.events) == 1
    event = result.events[0]
    assert event.origin_commit == c1
    assert event.churn_commit == c2
    assert event.origin_path == "old.py"
    # Later edits in new.py still blame c1 for surviving lines.
    c3 = repo.commit({"new.py": renamed.replace("f()", "g()")}, when=day(20))
    result = run(repo)
    assert len(result.events) == 2
    assert result.events[1].origin_commit == c1
    assert result.events[1].churn_commit == c3


def test_whole_file_delete_churns_all_window_lines(repo: RepoBuilder) -> None:
    repo.commit({"a.py": "line_one = 1\nline_two = 2\n"}, when=day(0))
    repo.commit({"a.py": None}, when=day(10))
    result = run(repo)
    assert len(result.events) == 2
    assert all(e.kind == "deleted" for e in result.events)


def test_merge_first_parent_no_double_count(repo: RepoBuilder) -> None:
    base = repo.commit({"main.py": "base_line = 0\n"}, when=day(0))
    # Side branch adds side.py (not on HEAD).
    side = repo.commit(
        {"side.py": "side_feature()\n"},
        when=day(1),
        author="bob",
        parents=[base],
        update_head=False,
    )
    main2 = repo.commit({"main.py": "base_line = 1\n"}, when=day(2))
    merge = repo.commit(
        {"side.py": "side_feature()\n"},
        when=day(3),
        author="carol",
        parents=[main2, side],
        message="merge",
    )
    # Edit the branch-added line within the window of the MERGE.
    repo.commit({"side.py": "side_feature(arg)\n"}, when=day(5))
    result = run(repo)
    churn_of_side = [e for e in result.events if e.path == "side.py"]
    assert len(churn_of_side) == 1
    # First-parent attribution: the line belongs to the merge commit/author.
    assert churn_of_side[0].origin_commit == merge
    assert "carol" in churn_of_side[0].origin_author
    # side.py's line was added exactly once (at the merge).
    assert result.added[("side.py", "carol <carol@example.com>")] == 1


def test_line_churns_at_most_once_then_replacement_owns_it(
    repo: RepoBuilder,
) -> None:
    c1 = repo.commit({"a.py": "version = 1\n"}, when=day(0))
    c2 = repo.commit({"a.py": "version = 2\n"}, when=day(5))
    c3 = repo.commit({"a.py": "version = 3\n"}, when=day(10))
    result = run(repo)
    assert len(result.events) == 2
    first, second = result.events
    assert (first.origin_commit, first.churn_commit) == (c1, c2)
    # The second churn blames c2's replacement line, NOT c1 again.
    assert (second.origin_commit, second.churn_commit) == (c2, c3)


def test_self_churn_counts(repo: RepoBuilder) -> None:
    repo.commit({"a.py": "mine = 1\n"}, when=day(0), author="alice")
    repo.commit({"a.py": "mine = 2\n"}, when=day(5), author="alice")
    result = run(repo)
    assert len(result.events) == 1
    assert result.events[0].origin_author == result.events[0].churn_author


def test_since_until_range_filters_origins(repo: RepoBuilder) -> None:
    repo.commit({"a.py": "ancient = 1\n"}, when=day(0))
    repo.commit({"a.py": "ancient = 1\nnewer = 2\n"}, when=day(100))
    repo.commit({"a.py": "ancient = 2\nnewer = 3\n"}, when=day(110))
    result = run(repo, since=day(90), until=day(120))
    # Only the in-range line ("newer") can churn; "ancient" is out of range.
    assert len(result.events) == 1
    assert result.events[0].origin_time == day(100)
    # Denominator: only in-range additions (newer@100 + 2 replacements@110).
    assert sum(result.added.values()) == 3


def test_commits_after_until_still_churn_in_range_lines(
    repo: RepoBuilder,
) -> None:
    repo.commit({"a.py": "late_line = 1\n"}, when=day(10))
    # Churned after `until`, but within the window of the add: counts.
    repo.commit({"a.py": "late_line = 2\n"}, when=day(25))
    result = run(repo, until=day(15))
    assert len(result.events) == 1
    # The churning commit itself is out of range: its replacement line is
    # not part of the denominator.
    assert sum(result.added.values()) == 1


def test_binary_files_are_skipped(repo: RepoBuilder) -> None:
    repo.commit({"blob.bin": b"\x00\x01\x02\x03binary"}, when=day(0))
    repo.commit({"blob.bin": b"\x00\x05\x06\x07binary"}, when=day(5))
    result = run(repo)
    assert result.events == []
    assert result.added == {}


def test_shrink_to_fit_recent_lines_count_fully(repo: RepoBuilder) -> None:
    # Line added 5 days before `until`, churned 2 days later: full churn,
    # despite never having a complete 30-day window.
    repo.commit({"a.py": "recent = 1\n"}, when=day(95))
    repo.commit({"a.py": "recent = 2\n"}, when=day(97))
    result = run(repo, until=day(100))
    assert len(result.events) == 1


def test_multiple_repos_fixture(make_repo: Callable[[], RepoBuilder]) -> None:
    repo_a, repo_b = make_repo(), make_repo()
    repo_a.commit({"a.py": "only_in_a = 1\n"}, when=day(0))
    repo_b.commit({"b.py": "only_in_b = 1\nsecond = 2\n"}, when=day(0))
    assert sum(run(repo_a).added.values()) == 1
    assert sum(run(repo_b).added.values()) == 2
