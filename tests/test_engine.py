"""End-to-end engine scenarios on real (tiny, deterministic) git repos."""

from __future__ import annotations

from collections.abc import Callable

from code_churn.engine import analyze
from code_churn.model import AnalysisConfig, AnalysisResult, GroupBy
from tests.conftest import DAY, RepoBuilder, day

WINDOW_DAYS = 30


def run(
    repo: RepoBuilder,
    *,
    window_days: int = WINDOW_DAYS,
    since: int | None = None,
    until: int | None = None,
    exclude: tuple[str, ...] = (),
    mailmap: str | None = None,
    exclude_authors: tuple[str, ...] = (),
    collect_survival: bool = False,
    survival_group_by: GroupBy | None = None,
    collect_lifespans: bool = False,
) -> AnalysisResult:
    config = AnalysisConfig(
        repo_path=str(repo.path),
        rev="HEAD",
        window_seconds=window_days * DAY,
        since=since,
        until=until,
        exclude=exclude,
        mailmap=mailmap,
        exclude_authors=exclude_authors,
        collect_survival=collect_survival,
        survival_group_by=survival_group_by,
        collect_lifespans=collect_lifespans,
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


def test_exclude_removes_added_and_churned(repo: RepoBuilder) -> None:
    repo.commit(
        {"gen/out.c": "generated_code();\n", "src.py": "handwritten = 1\n"},
        when=day(0),
    )
    repo.commit(
        {"gen/out.c": "regenerated_code();\n", "src.py": "handwritten = 2\n"},
        when=day(10),
    )
    result = run(repo, exclude=("gen/**",))
    # Only src.py contributes: its churn event and its 2 added lines.
    assert len(result.events) == 1
    assert result.events[0].origin_path == "src.py"
    assert set(result.added) == {("src.py", "alice <alice@example.com>")}
    assert sum(result.added.values()) == 2


def test_exclude_glob_depth_semantics(repo: RepoBuilder) -> None:
    repo.commit(
        {"top.c": "top_level();\n", "gen/deep/x.c": "nested_code();\n"},
        when=day(0),
    )
    # `**/*.c` matches at any depth; bare `*.c` matches top level only.
    deep = run(repo, exclude=("**/*.c",))
    assert deep.added == {}
    shallow = run(repo, exclude=("*.c",))
    assert set(shallow.added) == {("gen/deep/x.c", "alice <alice@example.com>")}


def test_exclude_move_from_excluded_file_stays_excluded(repo: RepoBuilder) -> None:
    # A line born in an excluded file, moved into an included one, keeps its
    # excluded add-time path: never counted as added, its churn filtered.
    repo.commit({"gen/out.c": "carried_over_line();\n"}, when=day(0))
    repo.commit({"gen/out.c": None, "src.py": "carried_over_line();\n"}, when=day(5))
    repo.commit({"src.py": "edited_line();\n"}, when=day(10))
    result = run(repo, exclude=("gen/**",))
    assert result.events == []
    # The only counted add is day(10)'s replacement line, genuinely born in
    # src.py; the carried line itself never entered the denominator.
    assert result.added == {("src.py", "alice <alice@example.com>"): 1}


def test_exclude_move_into_excluded_file_still_counts(repo: RepoBuilder) -> None:
    # A line born in an included file was counted as added; churn after a
    # move into an excluded file still counts (origin path decides).
    repo.commit({"src.py": "carried_over_line();\n"}, when=day(0))
    repo.commit({"src.py": None, "gen/out.c": "carried_over_line();\n"}, when=day(5))
    repo.commit({"gen/out.c": "edited_line();\n"}, when=day(10))
    result = run(repo, exclude=("gen/**",))
    assert len(result.events) == 1
    assert result.events[0].origin_path == "src.py"
    assert result.events[0].path == "gen/out.c"
    assert set(result.added) == {("src.py", "alice <alice@example.com>")}


def test_survival_not_collected_by_default(repo: RepoBuilder) -> None:
    repo.commit({"a.py": "alpha = 1\n"}, when=day(0))
    assert run(repo).survival is None


def test_survival_records_churn_outside_window(repo: RepoBuilder) -> None:
    repo.commit({"a.py": "alpha = 1\n"}, when=day(0))
    repo.commit({"a.py": "alpha = 2\n"}, when=day(40))
    result = run(repo, collect_survival=True)
    # Outside the 30-day window: no report event, but a death for the fit.
    assert result.events == []
    survival = result.survival
    assert survival is not None
    assert survival.event_lifetimes == [40 * DAY]
    # The day(40) replacement line survives with age 0 at observation end.
    assert survival.censored_ages == [0]


def test_survival_censored_ages_measured_to_last_commit(repo: RepoBuilder) -> None:
    repo.commit({"a.py": "alpha = 1\nbeta = 2\n"}, when=day(0))
    repo.commit({"a.py": "alpha = 1\nbeta = 99\n"}, when=day(10))
    survival = run(repo, collect_survival=True).survival
    assert survival is not None
    assert survival.event_lifetimes == [10 * DAY]
    # Survivors: alpha (born day 0) and beta's replacement (born day 10).
    assert sorted(survival.censored_ages) == [0, 10 * DAY]


def test_survival_walk_continues_past_until_plus_window(repo: RepoBuilder) -> None:
    repo.commit({"a.py": "alpha = 1\n"}, when=day(0))
    # Past until + window: the windowed walk stops before this commit, but
    # survival collection must observe it as a death, not a survivor.
    repo.commit({"a.py": "alpha = 2\n"}, when=day(40))
    result = run(repo, until=day(1), collect_survival=True)
    assert result.events == []
    survival = result.survival
    assert survival is not None
    assert survival.event_lifetimes == [40 * DAY]
    # The day(40) replacement is out of range: not a censored observation.
    assert survival.censored_ages == []


def test_survival_respects_exclusions(repo: RepoBuilder) -> None:
    repo.commit(
        {"gen/out.c": "generated_code();\n", "src.py": "handwritten = 1\n"},
        when=day(0),
    )
    repo.commit({"gen/out.c": "regenerated_code();\n"}, when=day(10))
    survival = run(repo, exclude=("gen/**",), collect_survival=True).survival
    assert survival is not None
    # Only src.py's surviving line counts; gen/** appears in neither list.
    assert survival.event_lifetimes == []
    assert survival.censored_ages == [10 * DAY]


def test_survival_moved_line_keeps_original_clock(repo: RepoBuilder) -> None:
    repo.commit({"a.py": "moved_function_call()\n"}, when=day(0))
    repo.commit({"b.py": "moved_function_call()\n", "a.py": None}, when=day(5))
    repo.commit({"b.py": "changed_function_call()\n"}, when=day(10))
    survival = run(repo, collect_survival=True).survival
    assert survival is not None
    # Lifetime runs from the original add (day 0), not the move (day 5).
    assert survival.event_lifetimes == [10 * DAY]


def test_survival_by_group_not_collected_without_flag(repo: RepoBuilder) -> None:
    repo.commit({"a.py": "alpha = 1\n"}, when=day(0))
    assert run(repo).survival_by_group is None
    assert run(repo, collect_survival=True).survival_by_group is None


def test_survival_by_group_author(repo: RepoBuilder) -> None:
    repo.commit({"a.py": "alpha = 1\n"}, when=day(0), author="alice")
    repo.commit({"b.py": "bravo = 1\n"}, when=day(0), author="bob")
    repo.commit({"a.py": "alpha = 2\n"}, when=day(10), author="bob")
    result = run(repo, collect_survival=True, survival_group_by="author")

    # Pooled survival is unchanged by grouping.
    survival = result.survival
    assert survival is not None
    assert survival.event_lifetimes == [10 * DAY]
    assert sorted(survival.censored_ages) == [0, 10 * DAY]

    groups = result.survival_by_group
    assert groups is not None
    alice = groups["alice <alice@example.com>"]
    bob = groups["bob <bob@example.com>"]
    assert set(groups) == {"alice <alice@example.com>", "bob <bob@example.com>"}
    # Alice's line died at day 10; she has no surviving lines.
    assert alice.event_lifetimes == [10 * DAY]
    assert alice.censored_ages == []
    # Bob survives with b.py (10 days old) and the replacement line (0 days).
    assert bob.event_lifetimes == []
    assert sorted(bob.censored_ages) == [0, 10 * DAY]


def test_survival_by_group_filetype(repo: RepoBuilder) -> None:
    repo.commit(
        {"a.py": "alpha = 1\n", "b.c": "bravo();\n", "Makefile": "all:\n"},
        when=day(0),
    )
    repo.commit({"a.py": "alpha = 2\n"}, when=day(10))
    groups = run(
        repo, collect_survival=True, survival_group_by="filetype"
    ).survival_by_group
    assert groups is not None
    assert set(groups) == {".py", ".c", "(no extension)"}
    assert groups[".py"].event_lifetimes == [10 * DAY]
    assert groups[".py"].censored_ages == [0]  # the replacement line
    assert groups[".c"].censored_ages == [10 * DAY]
    assert groups["(no extension)"].censored_ages == [10 * DAY]


def test_survival_by_group_moved_line_keeps_origin_suffix(repo: RepoBuilder) -> None:
    # The line is born in a .py file, moved verbatim into a .c file, then
    # churned: the stratum follows the add-time path, so it dies in ".py".
    repo.commit({"a.py": "moved_function_call()\n"}, when=day(0))
    repo.commit({"b.c": "moved_function_call()\n", "a.py": None}, when=day(5))
    repo.commit({"b.c": "changed_function_call()\n"}, when=day(10))
    groups = run(
        repo, collect_survival=True, survival_group_by="filetype"
    ).survival_by_group
    assert groups is not None
    assert groups[".py"].event_lifetimes == [10 * DAY]
    assert groups[".c"].event_lifetimes == []


def test_lifespans_not_collected_by_default(repo: RepoBuilder) -> None:
    repo.commit({"a.py": "alpha = 1\n"}, when=day(0))
    result = run(repo)
    assert result.lifespans is None
    assert result.observation_end is None


def test_lifespans_record_birth_and_death(repo: RepoBuilder) -> None:
    repo.commit({"a.py": "alpha = 1\nbeta = 2\n"}, when=day(0))
    repo.commit({"a.py": "alpha = 1\nbeta = 99\n"}, when=day(10))
    result = run(repo, collect_lifespans=True)
    assert result.observation_end == day(10)
    assert result.lifespans is not None
    # beta died at day 10; alpha and beta's replacement are alive.
    assert sorted(result.lifespans, key=lambda p: (p[0], p[1] or 0)) == [
        (day(0), None),
        (day(0), day(10)),
        (day(10), None),
    ]


def test_lifespans_walk_continues_past_until_plus_window(repo: RepoBuilder) -> None:
    # collect_lifespans alone (without collect_survival) must disable the
    # until + window early-stop, or the day(40) death looks like survival.
    repo.commit({"a.py": "alpha = 1\n"}, when=day(0))
    repo.commit({"a.py": "alpha = 2\n"}, when=day(40))
    result = run(repo, until=day(1), collect_lifespans=True)
    assert result.lifespans is not None
    assert (day(0), day(40)) in result.lifespans


def test_lifespans_respect_exclusions(repo: RepoBuilder) -> None:
    repo.commit(
        {"gen/out.c": "generated_code();\n", "src.py": "handwritten = 1\n"},
        when=day(0),
    )
    repo.commit({"gen/out.c": "regenerated_code();\n"}, when=day(10))
    result = run(repo, exclude=("gen/**",), collect_lifespans=True)
    # Only src.py's surviving line counts; gen/** appears nowhere.
    assert result.lifespans == [(day(0), None)]


def test_lifespans_moved_line_keeps_original_birth(repo: RepoBuilder) -> None:
    repo.commit({"a.py": "moved_function_call()\n"}, when=day(0))
    repo.commit({"b.py": "moved_function_call()\n", "a.py": None}, when=day(5))
    repo.commit({"b.py": "changed_function_call()\n"}, when=day(10))
    result = run(repo, collect_lifespans=True)
    assert result.lifespans is not None
    # Birth is the original add (day 0), not the move (day 5).
    assert (day(0), day(10)) in result.lifespans


def test_mailmap_joins_added_and_churn_attribution(repo: RepoBuilder) -> None:
    repo.commit({"a.py": "alpha = 1\n"}, when=day(0), author="alice2")
    repo.commit({"a.py": "alpha = 2\n"}, when=day(10), author="bob")
    mailmap = "alice <alice@example.com> <alice2@example.com>\n"
    result = run(repo, mailmap=mailmap)
    assert len(result.events) == 1
    assert result.events[0].origin_author == "alice <alice@example.com>"
    assert set(result.added) == {
        ("a.py", "alice <alice@example.com>"),
        ("a.py", "bob <bob@example.com>"),
    }


def test_mailmap_joins_survival_groups(repo: RepoBuilder) -> None:
    repo.commit({"a.py": "alpha = 1\n"}, when=day(0), author="alice")
    repo.commit({"b.py": "bravo = 1\n"}, when=day(0), author="alice2")
    repo.commit({"b.py": "bravo = 2\n"}, when=day(10), author="bob")
    mailmap = "alice <alice@example.com> <alice2@example.com>\n"
    result = run(
        repo, mailmap=mailmap, collect_survival=True, survival_group_by="author"
    )
    assert result.survival_by_group is not None
    merged = result.survival_by_group["alice <alice@example.com>"]
    # alice2's b.py line (died at day 10) lands in alice's stratum; alice's
    # own a.py line is censored at 10 days.
    assert merged.event_lifetimes == [10 * DAY]
    assert merged.censored_ages == [10 * DAY]
    assert "alice2 <alice2@example.com>" not in result.survival_by_group


def test_mailmap_applies_after_repo_mailmap(repo: RepoBuilder) -> None:
    # The external mailmap resolves AFTER the repo's own .mailmap, so its
    # entries must be keyed on the repo-resolved identity — and chain.
    (repo.path / ".mailmap").write_text("Repo Alice <ralice@x> <alice@example.com>\n")
    repo.commit({"a.py": "alpha = 1\n"}, when=day(0), author="alice")
    result = run(repo, mailmap="Final Alice <falice@x> <ralice@x>\n")
    assert set(result.added) == {("a.py", "Final Alice <falice@x>")}


def test_exclude_author_removes_added_and_received_churn(repo: RepoBuilder) -> None:
    repo.commit({"a.py": "alpha = 1\n"}, when=day(0), author="alice")
    repo.commit({"b.py": "bravo = 1\n"}, when=day(0), author="bob")
    repo.commit({"a.py": "alpha = 2\n"}, when=day(10), author="bob")
    result = run(repo, exclude_authors=("alice",))
    # alice's add and the churn of her line are gone; bob's lines remain
    # (including his replacement line in a.py).
    assert result.events == []
    assert result.added == {
        ("b.py", "bob <bob@example.com>"): 1,
        ("a.py", "bob <bob@example.com>"): 1,
    }


def test_exclude_author_as_churner_drops_victim_death(repo: RepoBuilder) -> None:
    repo.commit({"a.py": "alpha = 1\n"}, when=day(0), author="alice")
    repo.commit({"a.py": "alpha = 2\n"}, when=day(10), author="renovate")
    result = run(
        repo,
        exclude_authors=("renovate",),
        collect_survival=True,
        collect_lifespans=True,
    )
    assert result.events == []
    assert result.survival is not None
    # The victim's death is recorded nowhere: not an event, not a survival
    # death — and the line left the origin map, so it is never censored
    # either. Its exposure drops out of --fit-decay entirely.
    assert result.survival.event_lifetimes == []
    assert result.survival.censored_ages == []
    assert result.lifespans == []
    assert result.added == {("a.py", "alice <alice@example.com>"): 1}


def test_exclude_author_vanishes_from_groups_and_lifespans(repo: RepoBuilder) -> None:
    repo.commit({"a.py": "alpha = 1\n"}, when=day(0), author="alice")
    repo.commit({"b.py": "bravo = 1\n"}, when=day(0), author="bot")
    repo.commit({"b.py": "bravo = 2\n"}, when=day(10), author="bot")
    result = run(
        repo,
        exclude_authors=("bot",),
        collect_survival=True,
        survival_group_by="author",
        collect_lifespans=True,
    )
    assert result.survival_by_group is not None
    assert set(result.survival_by_group) == {"alice <alice@example.com>"}
    assert result.lifespans == [(day(0), None)]


def test_exclude_author_is_case_sensitive(repo: RepoBuilder) -> None:
    repo.commit({"a.py": "alpha = 1\n"}, when=day(0), author="Alice")
    result = run(repo, exclude_authors=("alice",))
    assert set(result.added) == {("a.py", "Alice <Alice@example.com>")}


def test_exclude_author_matches_post_mailmap_name(repo: RepoBuilder) -> None:
    repo.commit({"a.py": "alpha = 1\n"}, when=day(0), author="alice2")
    mailmap = "alice <alice@example.com> <alice2@example.com>\n"
    result = run(repo, mailmap=mailmap, exclude_authors=("alice",))
    assert result.added == {}


def test_multiple_repos_fixture(make_repo: Callable[[], RepoBuilder]) -> None:
    repo_a, repo_b = make_repo(), make_repo()
    repo_a.commit({"a.py": "only_in_a = 1\n"}, when=day(0))
    repo_b.commit({"b.py": "only_in_b = 1\nsecond = 2\n"}, when=day(0))
    assert sum(run(repo_a).added.values()) == 1
    assert sum(run(repo_b).added.values()) == 2
