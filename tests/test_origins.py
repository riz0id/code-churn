"""Pure unit tests for OriginMap splice and move-matching logic (no git)."""

from __future__ import annotations

from code_churn.model import LineOrigin
from code_churn.origins import (
    FilePatch,
    OriginMap,
    PatchHunk,
    PatchLine,
)


def origin(commit: str, time: int = 0, in_range: bool = True) -> LineOrigin:
    return LineOrigin(
        commit_id=commit,
        path="",  # apply_commit stamps the real add-time path per file
        time=time,
        author=f"{commit}-author",
        in_range=in_range,
    )


def added(new_lineno: int, content: str) -> PatchLine:
    return PatchLine(origin="+", old_lineno=-1, new_lineno=new_lineno, content=content)


def deleted(old_lineno: int, content: str) -> PatchLine:
    return PatchLine(origin="-", old_lineno=old_lineno, new_lineno=-1, content=content)


def new_file(path: str, *contents: str) -> FilePatch:
    lines = tuple(added(i, text) for i, text in enumerate(contents, start=1))
    return FilePatch(old_path=None, new_path=path, hunks=(PatchHunk(lines=lines),))


def test_new_file_adds_fresh_lines() -> None:
    om = OriginMap()
    out = om.apply_commit([new_file("a.py", "one\n", "two\n")], origin("c1"))
    assert out.candidates == []
    assert out.fresh_added == {"a.py": 2}
    assert [o.commit_id for o in om.lines("a.py")] == ["c1", "c1"]


def test_modification_churns_and_replaces_origin() -> None:
    om = OriginMap()
    om.apply_commit([new_file("a.py", "alpha line\n", "beta line\n")], origin("c1"))
    patch = FilePatch(
        old_path="a.py",
        new_path="a.py",
        hunks=(
            PatchHunk(
                lines=(deleted(2, "beta line\n"), added(2, "gamma line\n")),
            ),
        ),
    )
    out = om.apply_commit([patch], origin("c2"))
    assert len(out.candidates) == 1
    cand = out.candidates[0]
    assert cand.path == "a.py"
    assert cand.kind == "modified"
    assert cand.origin.commit_id == "c1"
    # The replacement line belongs to c2; line 1 still belongs to c1.
    assert [o.commit_id for o in om.lines("a.py")] == ["c1", "c2"]
    assert out.fresh_added == {"a.py": 1}


def test_pure_deletion_has_deleted_kind_and_shifts_positions() -> None:
    om = OriginMap()
    om.apply_commit(
        [new_file("a.py", "first line\n", "second line\n", "third line\n")],
        origin("c1"),
    )
    patch = FilePatch(
        old_path="a.py",
        new_path="a.py",
        hunks=(PatchHunk(lines=(deleted(1, "first line\n"),)),),
    )
    out = om.apply_commit([patch], origin("c2"))
    assert [c.kind for c in out.candidates] == ["deleted"]
    assert len(om.lines("a.py")) == 2
    # Later edit of (new) line 1 must blame "second line"'s origin: c1.
    patch2 = FilePatch(
        old_path="a.py",
        new_path="a.py",
        hunks=(PatchHunk(lines=(deleted(1, "second line\n"), added(1, "changed\n"))),),
    )
    out2 = om.apply_commit([patch2], origin("c3"))
    assert out2.candidates[0].origin.commit_id == "c1"


def test_cross_file_move_carries_origin_and_is_not_churn() -> None:
    om = OriginMap()
    om.apply_commit([new_file("a.py", "moved_code = 1\n")], origin("c1", time=100))
    move = [
        FilePatch(
            old_path="a.py",
            new_path="a.py",
            hunks=(PatchHunk(lines=(deleted(1, "moved_code = 1\n"),)),),
        ),
        new_file("b.py", "moved_code = 1\n"),
    ]
    out = om.apply_commit(move, origin("c2", time=200))
    assert out.candidates == []  # a move, not churn
    assert out.fresh_added == {}  # not re-counted as added
    assert om.lines("a.py") == []
    carried = om.lines("b.py")[0]
    assert carried.commit_id == "c1"
    assert carried.time == 100


def test_whitespace_differences_still_match_as_move() -> None:
    om = OriginMap()
    om.apply_commit([new_file("a.py", "moved_code = 1\n")], origin("c1"))
    move = [
        FilePatch(
            old_path="a.py",
            new_path="a.py",
            hunks=(PatchHunk(lines=(deleted(1, "moved_code = 1\n"),)),),
        ),
        new_file("b.py", "    moved_code=1\n"),
    ]
    out = om.apply_commit(move, origin("c2"))
    assert out.candidates == []
    assert om.lines("b.py")[0].commit_id == "c1"


def test_trivial_lines_never_match_as_move() -> None:
    om = OriginMap()
    om.apply_commit([new_file("a.py", "}\n")], origin("c1"))
    patches = [
        FilePatch(
            old_path="a.py",
            new_path="a.py",
            hunks=(PatchHunk(lines=(deleted(1, "}\n"),)),),
        ),
        new_file("b.py", "}\n"),
    ]
    out = om.apply_commit(patches, origin("c2"))
    assert len(out.candidates) == 1  # the deleted `}` is real churn
    assert out.fresh_added == {"b.py": 1}  # the added `}` is genuinely new


def test_move_matching_consumes_each_line_once() -> None:
    om = OriginMap()
    om.apply_commit(
        [new_file("a.py", "same_content\n", "same_content\n")], origin("c1")
    )
    # Delete both copies, re-add only one: one move, one churn.
    patches = [
        FilePatch(
            old_path="a.py",
            new_path="a.py",
            hunks=(
                PatchHunk(
                    lines=(
                        deleted(1, "same_content\n"),
                        deleted(2, "same_content\n"),
                        added(1, "same_content\n"),
                    )
                ),
            ),
        ),
    ]
    out = om.apply_commit(patches, origin("c2"))
    assert len(out.candidates) == 1
    assert len(om.lines("a.py")) == 1
    assert om.lines("a.py")[0].commit_id == "c1"


def test_rename_rekeys_map() -> None:
    om = OriginMap()
    om.apply_commit([new_file("old.py", "content line\n")], origin("c1"))
    rename = FilePatch(old_path="old.py", new_path="new.py", hunks=())
    out = om.apply_commit([rename], origin("c2"))
    assert out.candidates == []
    assert om.lines("old.py") == []
    assert om.lines("new.py")[0].commit_id == "c1"


def test_file_deletion_churns_every_line_and_removes_key() -> None:
    om = OriginMap()
    om.apply_commit([new_file("a.py", "line one x\n", "line two y\n")], origin("c1"))
    patch = FilePatch(
        old_path="a.py",
        new_path=None,
        hunks=(
            PatchHunk(lines=(deleted(1, "line one x\n"), deleted(2, "line two y\n"))),
        ),
    )
    out = om.apply_commit([patch], origin("c2"))
    assert len(out.candidates) == 2
    assert all(c.kind == "deleted" for c in out.candidates)
    assert om.lines("a.py") == []


def test_multi_hunk_offsets() -> None:
    om = OriginMap()
    om.apply_commit(
        [new_file("a.py", *[f"line number {i}\n" for i in range(1, 7)])],
        origin("c1"),
    )
    # Delete line 2; insert one new line after old line 4 (new position 4).
    patch = FilePatch(
        old_path="a.py",
        new_path="a.py",
        hunks=(
            PatchHunk(lines=(deleted(2, "line number 2\n"),)),
            PatchHunk(lines=(added(4, "inserted line\n"),)),
        ),
    )
    out = om.apply_commit([patch], origin("c2"))
    assert len(out.candidates) == 1
    result = om.lines("a.py")
    assert [o.commit_id for o in result] == ["c1", "c1", "c1", "c2", "c1", "c1"]


def test_untracked_file_degrades_gracefully() -> None:
    # A file git once mis-detected as binary was never tracked; a later text
    # diff against it must not churn (no provenance) and must not blow up.
    om = OriginMap()
    patch = FilePatch(
        old_path="was_binary.py",
        new_path="was_binary.py",
        hunks=(
            PatchHunk(
                lines=(deleted(3, "mystery line\n"), added(3, "replacement x\n"))
            ),
        ),
    )
    out = om.apply_commit([patch], origin("c1"))
    assert len(out.candidates) == 1
    assert out.candidates[0].origin.in_range is False  # sentinel: never churn
    # Position 3 is the replacement (owned by c1); 1-2 are sentinel padding.
    lines = om.lines("was_binary.py")
    assert len(lines) == 3
    assert lines[2].commit_id == "c1"
    assert lines[0].in_range is False and lines[1].in_range is False


def test_binary_patch_untracks_file() -> None:
    om = OriginMap()
    om.apply_commit([new_file("a.py", "text line here\n")], origin("c1"))
    binary = FilePatch(old_path="a.py", new_path="a.py", hunks=(), binary=True)
    out = om.apply_commit([binary], origin("c2"))
    assert out.candidates == []
    assert om.lines("a.py") == []  # stale text provenance dropped
