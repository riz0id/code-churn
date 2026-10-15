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
code-churn . --exclude '**/*.ipynb' --exclude 'vendor/**'   # skip generated/vendored files
code-churn . --fit-decay                                    # fit the repo's own decay timescale
code-churn . --decay-plot churn-decay.png                   # ...and graph it (.png/.svg/.pdf)
code-churn . --group-by author                              # per-group fits + likelihood-ratio test
code-churn . --churn-heatmap churn.png                      # cross-author churn matrix ('live' for a window)
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

## Fitting a decay function

The fixed `--window` is a step function: churn at day 29 counts, day 31
doesn't, and 30 days is arbitrary for any given repo. `--fit-decay` instead
discovers the repo's own decay timescale by treating line survival as a
censored survival problem:

- every churned line is an observed **death** with exact lifetime
  (churn time − add time);
- every line still alive at the end of the walk is **right-censored** — it
  survived at least its current age.

For an exponential model `S(t) = exp(−λt)` this has a closed-form
maximum-likelihood fit: `λ = deaths / total exposure`, where total exposure is
the sum of all death lifetimes plus all censored ages. The reported half-life
is `ln(2)/λ`, and the fitted weight `exp(−λ·Δt)` is the smooth replacement for
the window's hard cutoff.

Details:

- **Fitting ignores `--window`** and walks the full history even past
  `--until`, so churn landing after the window is observed as a late death
  rather than miscounted as survival. `--since`/`--until` still select which
  lines (by origin commit) participate; `--exclude` applies as usual.
- The fit appears in the table summary and under `"decay"` in JSON
  (`half_life_days`, `lambda_per_day`, `events`, `censored`, `exposure_days`,
  `caveats`).
- **Caveats, not silence**: the fit is flagged when the observed history is
  shorter than 30 days, or the half-life extrapolates past 4× the oldest
  observation.
- `--decay-plot PATH` (implies `--fit-decay`) writes a graph to PATH: the
  model-free Kaplan–Meier survival curve of the observed data with the fitted
  exponential overlaid, plus half-life guides — if the two curves diverge
  badly, the exponential assumption is wrong for your repo. Format follows
  the extension (`.png`, `.svg`, `.pdf`).

### Stratified fits: does a variable change the lifetime?

`--group-by author` (or `--group-by filetype`, implies `--fit-decay`) fits one
decay rate per group and tests whether the grouping variable actually affects
the fitted lifetime, via a likelihood-ratio test of the shared-rate model
against the per-group model. Both MLEs are closed-form, so the statistic is
`2·(Σ_g d_g·ln(d_g/E_g) − D·ln(D/E))` with `df = G − 1` (`d` = churn events,
`E` = exposure per group):

```
decay by author:
group                      half-life   λ/day    events  surviving
alice <alice@example.com>  13.9 days   0.0500   12      30
bob <bob@example.com>      46.2 days   0.0150   3       41
likelihood-ratio test (shared λ vs per-group λ): χ²=6.21, df=1, p=0.0127
```

A small p-value (conventionally < 0.05) means the groups' code decays at
genuinely different rates — the half-life differences are signal, not noise.
Groups with no churn events are listed (half-life `-`) and still contribute
their exposure to the test. JSON output carries the same data under
`"stratified_decay"` (`groups` + `lrt`).

### Cross-author churn heatmap

`--churn-heatmap TARGET` renders a matrix of churn counts — the line's
original author on rows, the churning author on columns, so the diagonal is
self-churn and off-diagonal cells show whose code gets reworked by whom.
TARGET is an image path (`.png`/`.svg`/`.pdf`) or the literal `live` to open
an interactive matplotlib window:

```sh
code-churn . --churn-heatmap churn.png
code-churn . --churn-heatmap live
```

Axes are truncated to the 20 most-involved authors when there are more.

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
- **`--exclude GLOB` filters by add-time path.** Patterns are gitignore-style
  globs (pathlib full-match semantics): `*` stays within one path segment,
  `**` recurses — so `--exclude '**/*.c'` matches C files at any depth while
  `--exclude '*.c'` matches only the top level. A line whose add-time path
  matches (the same path the per-file report keys on) counts neither as added
  nor as churned. Useful for committed generated artifacts, which otherwise
  dominate churn rates. Two move-boundary consequences: a line born in an
  excluded file stays excluded even after being moved into an included file,
  and a line born in an included file still counts if it churns after moving
  into an excluded one.
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
