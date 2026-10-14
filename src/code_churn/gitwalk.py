"""Git access layer: first-parent walk producing pure FilePatch values.

This is the only module that imports pygit2. Each yielded commit carries its
diff against its first parent, already converted to the git-free structures
that :mod:`code_churn.origins` consumes.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pygit2
from pygit2.enums import DeltaStatus, DiffOption, SortMode

from code_churn.origins import FilePatch, PatchHunk, PatchLine

# IGNORE_WHITESPACE can only equate equal-count line runs, so the positional
# bookkeeping in OriginMap stays correct. Do NOT add IGNORE_BLANK_LINES: it
# drops blank-line insertions from the patch while later hunks still account
# for them, which desyncs the origin map.
DIFF_FLAGS = DiffOption.IGNORE_WHITESPACE
# Context lines carry no information we use; hunks shrink to just +/- lines.
CONTEXT_LINES = 0


@dataclass(frozen=True, slots=True)
class CommitDiff:
    commit_id: str
    time: int  # commit time, epoch seconds
    author: str  # "Name <email>", mailmap-resolved
    patches: tuple[FilePatch, ...]


def open_repo(path: str) -> pygit2.Repository:
    try:
        return pygit2.Repository(path)
    except pygit2.GitError as exc:
        raise ValueError(f"not a git repository: {path}") from exc


def walk_first_parent(repo: pygit2.Repository, rev: str) -> Iterator[CommitDiff]:
    """Yield first-parent history of ``rev``, oldest first, with diffs.

    First-parent simplification makes the walk a linear chain: each yielded
    commit's first parent is the previously yielded commit, so applying the
    diffs in order keeps an OriginMap positionally exact.
    """
    try:
        tip = repo.revparse_single(rev).peel(pygit2.Commit)
    except (KeyError, pygit2.GitError) as exc:
        raise ValueError(f"cannot resolve revision {rev!r}") from exc

    try:
        mailmap = pygit2.Mailmap.from_repository(repo)
    except pygit2.GitError:
        mailmap = None

    walker = repo.walk(tip.id, SortMode.TOPOLOGICAL | SortMode.TIME | SortMode.REVERSE)
    walker.simplify_first_parent()

    for commit in walker:
        if commit.parents:
            diff = commit.parents[0].tree.diff_to_tree(
                commit.tree, flags=DIFF_FLAGS, context_lines=CONTEXT_LINES
            )
        else:
            diff = commit.tree.diff_to_tree(
                swap=True, flags=DIFF_FLAGS, context_lines=CONTEXT_LINES
            )
        diff.find_similar()  # rename detection

        sig = commit.author
        if mailmap is not None:
            try:
                sig = mailmap.resolve_signature(sig)
            except pygit2.GitError:
                pass
        author = f"{sig.name} <{sig.email}>"

        yield CommitDiff(
            commit_id=str(commit.id),
            time=commit.commit_time,
            author=author,
            patches=tuple(_convert_patches(diff)),
        )


def _convert_patches(diff: pygit2.Diff) -> Iterator[FilePatch]:
    for patch in diff:
        if patch is None:
            continue
        delta = patch.delta
        old_path = delta.old_file.path if delta.status != DeltaStatus.ADDED else None
        new_path = delta.new_file.path if delta.status != DeltaStatus.DELETED else None
        if delta.is_binary:
            # No line content to track; the origin map untracks the file so a
            # later binary->text flip starts from a clean (sentinel) slate.
            yield FilePatch(old_path=old_path, new_path=new_path, hunks=(), binary=True)
            continue
        hunks = tuple(
            PatchHunk(
                lines=tuple(
                    PatchLine(
                        origin=line.origin,  # type: ignore[arg-type]
                        old_lineno=line.old_lineno,
                        new_lineno=line.new_lineno,
                        content=line.content,
                    )
                    # '=', '<', '>' are end-of-file-newline markers; only
                    # real added/removed lines matter to the origin map.
                    for line in hunk.lines
                    if line.origin in ("+", "-")
                )
            )
            for hunk in patch.hunks
        )
        yield FilePatch(old_path=old_path, new_path=new_path, hunks=hunks)
