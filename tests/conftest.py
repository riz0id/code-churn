"""Fixture: build throwaway git repos with exact, deterministic timestamps.

Commits are created straight from in-memory indexes (no worktree writes), and
every commit's time comes from the test via ``pygit2.Signature(..., time=...)``,
so window arithmetic in tests is exact — no mocking, no clock.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pygit2
import pytest

# Fixed epoch base for all test repos; tests speak in whole days from T0.
T0 = 1_700_000_000
DAY = 86_400


def day(n: float) -> int:
    return T0 + int(n * DAY)


class RepoBuilder:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.repo = pygit2.init_repository(str(path), bare=False)

    def commit(
        self,
        files: dict[str, str | bytes | None],
        *,
        when: int,
        author: str = "alice",
        message: str = "commit",
        parents: list[str] | None = None,
        update_head: bool = True,
    ) -> str:
        """Create a commit; ``files`` maps path -> content (None = delete).

        The tree is the first parent's tree with ``files`` applied. Pass
        ``parents`` explicitly to build branch/merge shapes; ``update_head``
        False creates a commit without moving the current branch (side-branch
        work for merge tests).
        """
        repo = self.repo
        if parents is None:
            parents = [] if repo.head_is_unborn else [str(repo.head.target)]

        index = pygit2.Index()
        if parents:
            base_tree = repo[parents[0]].peel(pygit2.Commit).tree
            index.read_tree(base_tree)
        for file_path, content in files.items():
            if content is None:
                index.remove(file_path)
            else:
                data = content.encode() if isinstance(content, str) else content
                blob = repo.create_blob(data)
                index.add(
                    pygit2.IndexEntry(file_path, blob, pygit2.enums.FileMode.BLOB)
                )
        tree = index.write_tree(repo)

        sig = pygit2.Signature(author, f"{author}@example.com", when, 0)
        ref = "HEAD" if update_head else None
        oid = repo.create_commit(ref, sig, sig, message, tree, parents)
        return str(oid)


@pytest.fixture
def make_repo(tmp_path: Path) -> Callable[[], RepoBuilder]:
    count = 0

    def factory() -> RepoBuilder:
        nonlocal count
        count += 1
        repo_path = tmp_path / f"repo{count}"
        repo_path.mkdir()
        return RepoBuilder(repo_path)

    return factory


@pytest.fixture
def repo(make_repo: Callable[[], RepoBuilder]) -> RepoBuilder:
    return make_repo()
