# code-churn

Measure code churn from git history. **Churn** here means: a line added in one
commit that is modified or deleted by a later commit within a configurable time
window after it was added (default 30 days). High churn signals code that
didn't survive contact with reality — it had to be reworked soon after it was
written.

## Usage

```sh
nix develop        # or `direnv allow` if you use direnv

code-churn .                          # analyze the repo in the current directory
code-churn /path/to/repo --window 90d
code-churn . --since 2025-01-01 --until 2025-06-30 --format json
code-churn . --by author --format csv --output churn.csv
```

Build and run the packaged application:

```sh
nix build
./result/bin/code-churn --help
```

## What gets reported

- **Repo summary** — total lines added, lines churned, overall churn rate.
- **Per-file** — churn per file (keyed by the file's path at the time the
  lines were added; renames are followed for tracking, not re-keyed).
- **Per-author** — both directions, reported separately:
  - *churn received*: lines this author added that were churned (with rate
    against lines they added), and
  - *churn caused*: churn events this author performed on anyone's lines.

Output formats: human-readable table (default), `--format json` (includes the
individual churn events), `--format csv`.

## How churn is measured

- **Line-level, single pass.** History is walked oldest→newest along
  first-parent, maintaining a per-file map of which commit each current line
  came from. A deleted/modified line is churn if its origin commit is within
  the window. No per-line `git blame` calls, so large repos stay fast.
- **Whitespace-only changes don't count** (diffs are taken ignoring
  whitespace), and the line's original authorship/clock is preserved through
  them.
- **Moved code doesn't count.** A line deleted in a commit whose exact
  (whitespace-normalized) content reappears as an added line in the same
  commit is treated as a move: no churn, and the moved line keeps its original
  commit/author/time. Very short lines (fewer than 4 non-whitespace
  characters, e.g. `}`) never match as moves.
- **Self-churn counts.** Editing your own recently added line is still churn.
- **A line churns at most once.** Modifying a line replaces it; the
  replacement belongs to the modifying commit and gets its own window.
- **Recent lines use a shrunk window.** A line added 10 days ago with a
  30-day window is measured over its 10 elapsed days and counts fully in
  rates — recent code is included, not excluded or flagged.
- **First-parent history.** Merge commits contribute one diff against their
  first parent, so branch content is never double-counted — but content
  brought in by a merge commit is attributed to the merge's author and
  timestamp. With squash/rebase workflows this is exact; with merge-commit
  workflows it is the standard trade-off.
- Binary files are skipped. When git's binary detection flips for a file
  across history (e.g. added as "binary", later diffed as text), lines from
  the untracked period simply have no provenance: they never count as added
  or churned.

## Development

Nix manages all dependencies: add runtime deps to `projectDeps` in
`flake.nix` (from nixpkgs' `python3Packages`), dev-only deps to `devDeps`.
`dependencies` in `pyproject.toml` intentionally stays empty.

```sh
pytest             # test suite (builds throwaway git repos via pygit2)
mypy               # strict
black src tests
```

## VS Code / Pylance

Entering the dev shell writes the current Nix Python environment's store path
into `.vscode/settings.json` (`python.defaultInterpreterPath` and
`python.analysis.extraPaths`), so Pylance resolves imports for everything in
`projectDeps`. That path changes whenever `projectDeps` or nixpkgs move, which
is why it is refreshed on every shell entry — with direnv this happens
automatically on `cd`.

Other keys in the file are preserved (the update is a JSON merge), and the file
is only rewritten when the path actually changes. Two caveats: the file must be
strict JSON — comments make the hook warn and skip rather than clobber your
settings — and it is gitignored, since store paths are machine-local. Reload the
VS Code window if Pylance does not pick up a change.
