"""Line-origin tracking: the positional map at the heart of the churn pass.

An :class:`OriginMap` holds, for every file in the tree as of the last
processed commit, one :class:`~code_churn.model.LineOrigin` per line.
Applying a commit's diff splices the map (so positions stay correct),
detects moved lines (which keep their original origin and never churn),
and reports which deleted lines are churn candidates.

This module is pure: it knows nothing about pygit2. The walker layer
converts pygit2 patches into :class:`FilePatch` values.

Positional correctness depends on the diff never suppressing a change that
alters line counts. ``IGNORE_WHITESPACE`` is safe (it can only equate
same-count line runs); ``IGNORE_BLANK_LINES`` is not (it drops blank-line
insertions from the patch while later hunks still account for them), so the
walker must never use it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from typing import Literal

from code_churn.model import ChurnKind, LineOrigin

# A deleted line only pairs with an added line as a "move" when its content,
# stripped of all whitespace, is at least this long. Stops a deleted `}` from
# being matched to an unrelated added `}`.
MOVE_MIN_SIGNIFICANT_CHARS = 4

# Origin for lines in regions the map has no provenance for — e.g. a file git
# once mis-detected as binary (its add was skipped) later diffed as text.
# in_range=False guarantees such lines never churn and are never counted.
UNTRACKED = LineOrigin(commit_id="", path="", time=0, author="", in_range=False)


def normalize(content: str) -> str:
    """Whitespace-free form of a line, used for move matching."""
    return "".join(content.split())


@dataclass(frozen=True, slots=True)
class PatchLine:
    origin: Literal["+", "-"]
    old_lineno: int  # 1-based position in the old file; -1 for '+' lines
    new_lineno: int  # 1-based position in the new file; -1 for '-' lines
    content: str


@dataclass(frozen=True, slots=True)
class PatchHunk:
    lines: tuple[PatchLine, ...]


@dataclass(frozen=True, slots=True)
class FilePatch:
    old_path: str | None  # None when the file was added in this commit
    new_path: str | None  # None when the file was deleted in this commit
    hunks: tuple[PatchHunk, ...]  # may be empty (pure rename)
    binary: bool = False  # binary delta: no hunks; the file becomes untracked


@dataclass(frozen=True, slots=True)
class ChurnCandidate:
    """A non-move deleted line; the engine decides if it is in-window churn."""

    path: str  # path at the churning commit (new path if the file survives)
    kind: ChurnKind
    origin: LineOrigin


@dataclass(slots=True)
class CommitOutcome:
    candidates: list[ChurnCandidate] = field(default_factory=list)
    fresh_added: dict[str, int] = field(default_factory=dict)  # path -> lines


@dataclass(slots=True)
class _Deletion:
    patch_index: int
    old_lineno: int
    origin: LineOrigin
    kind: ChurnKind
    consumed: bool = False  # True once matched to an added line (a move)


class OriginMap:
    """Per-file line origins for the tree as of the last applied commit."""

    def __init__(self) -> None:
        self._files: dict[str, list[LineOrigin]] = {}

    def lines(self, path: str) -> list[LineOrigin]:
        return self._files.get(path, [])

    def apply_commit(
        self, patches: list[FilePatch], fresh: LineOrigin
    ) -> CommitOutcome:
        """Splice one commit's patches into the map.

        ``fresh`` is the origin assigned to genuinely new lines, with its
        ``path`` re-stamped per file to the add-time path (move targets keep
        the full origin — including add-time path — of the deleted line they
        matched instead).
        """
        old_lists: list[list[LineOrigin]] = []
        deletions: list[list[_Deletion]] = []  # parallel to patches, in order
        by_content: dict[str, deque[_Deletion]] = {}

        # Pass 1: snapshot old line lists and collect every deleted line with
        # its origin; index significant deletions by normalized content.
        for idx, patch in enumerate(patches):
            old = self._files.get(patch.old_path, []) if patch.old_path else []
            old_lists.append(old)
            dels: list[_Deletion] = []
            for hunk in patch.hunks:
                kind: ChurnKind = (
                    "modified"
                    if any(line.origin == "+" for line in hunk.lines)
                    else "deleted"
                )
                for line in hunk.lines:
                    if line.origin != "-":
                        continue
                    # A deletion beyond the tracked length means the file has
                    # a region we never saw (e.g. it was added while git
                    # considered it binary): no provenance, never churn.
                    origin = (
                        old[line.old_lineno - 1]
                        if 1 <= line.old_lineno <= len(old)
                        else UNTRACKED
                    )
                    record = _Deletion(
                        patch_index=idx,
                        old_lineno=line.old_lineno,
                        origin=origin,
                        kind=kind,
                    )
                    dels.append(record)
                    norm = normalize(line.content)
                    if len(norm) >= MOVE_MIN_SIGNIFICANT_CHARS:
                        by_content.setdefault(norm, deque()).append(record)
            deletions.append(dels)

        # Pass 2: match added lines against deleted content (cross-file,
        # within this commit). A match is a move: the added line carries the
        # deleted line's origin, and the deleted line stops being churn.
        carried: list[dict[int, LineOrigin]] = []  # per patch: new_lineno -> origin
        for patch in patches:
            carry: dict[int, LineOrigin] = {}
            for hunk in patch.hunks:
                for line in hunk.lines:
                    if line.origin != "+":
                        continue
                    norm = normalize(line.content)
                    if len(norm) < MOVE_MIN_SIGNIFICANT_CHARS:
                        continue
                    queue = by_content.get(norm)
                    if not queue:
                        continue
                    record = queue.popleft()
                    record.consumed = True
                    carry[line.new_lineno] = record.origin
            carried.append(carry)

        # Pass 3: unconsumed deletions are churn candidates; then rebuild each
        # file's origin list and count genuinely fresh additions.
        outcome = CommitOutcome()
        new_lists: dict[str, list[LineOrigin]] = {}
        removed_paths: list[str] = []

        for idx, patch in enumerate(patches):
            churn_path = patch.new_path or patch.old_path
            assert churn_path is not None  # a patch names at least one side

            if patch.binary:
                # The file is (now) binary: drop any stale text provenance so
                # a later binary->text flip can't mis-attribute churn.
                if patch.old_path:
                    removed_paths.append(patch.old_path)
                if patch.new_path:
                    removed_paths.append(patch.new_path)
                continue

            for record in deletions[idx]:
                if not record.consumed:
                    outcome.candidates.append(
                        ChurnCandidate(
                            path=churn_path, kind=record.kind, origin=record.origin
                        )
                    )

            deleted_at = {record.old_lineno for record in deletions[idx]}
            kept = [
                origin
                for lineno, origin in enumerate(old_lists[idx], start=1)
                if lineno not in deleted_at
            ]

            additions: list[tuple[int, LineOrigin, bool]] = []
            fresh_count = 0
            fresh_here: LineOrigin | None = None
            for hunk in patch.hunks:
                for line in hunk.lines:
                    if line.origin != "+":
                        continue
                    moved_origin = carried[idx].get(line.new_lineno)
                    if moved_origin is not None:
                        additions.append((line.new_lineno, moved_origin, True))
                    else:
                        if fresh_here is None:
                            fresh_here = replace(fresh, path=churn_path)
                        additions.append((line.new_lineno, fresh_here, False))
                        fresh_count += 1
            additions.sort(key=lambda item: item[0])

            # Rebuild: kept lines fill the gaps between additions, in order.
            # When kept runs dry before an addition's position (the file has
            # an untracked region, see UNTRACKED above), pad with sentinels so
            # positions beyond it stay usable.
            new_list: list[LineOrigin] = []
            kept_iter = iter(kept)
            for new_lineno, line_origin, _ in additions:
                while len(new_list) < new_lineno - 1:
                    new_list.append(next(kept_iter, UNTRACKED))
                new_list.append(line_origin)
            new_list.extend(kept_iter)

            if patch.old_path and patch.old_path != patch.new_path:
                removed_paths.append(patch.old_path)
            if patch.new_path is None:
                assert patch.old_path is not None
                removed_paths.append(patch.old_path)
            else:
                new_lists[patch.new_path] = new_list
            if fresh_count and patch.new_path is not None:
                outcome.fresh_added[patch.new_path] = (
                    outcome.fresh_added.get(patch.new_path, 0) + fresh_count
                )

        # Commit all writes only after every patch was read, so same-commit
        # rename chains (a->b while b->c) can't read a half-updated map.
        for path in removed_paths:
            self._files.pop(path, None)
        self._files.update(new_lists)
        return outcome
