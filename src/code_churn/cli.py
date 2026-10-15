"""Command-line interface for code-churn."""

from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from typing import cast

from code_churn import __version__
from code_churn.decay import fit_exponential, fit_stratified
from code_churn.engine import analyze
from code_churn.formatters import render_csv, render_json, render_table
from code_churn.model import AnalysisConfig, GroupBy
from code_churn.report import build_report, churn_matrix

_DURATION_UNITS = {
    "h": 3_600,
    "d": 86_400,
    "w": 7 * 86_400,
}

# Image formats accepted by --decay-plot (savefig picks the backend from the
# extension).
_PLOT_EXTENSIONS = (".png", ".svg", ".pdf")

EPILOG = """\
churn definition:
  A line added in one commit that is modified or deleted by a later commit
  within --window of being added. Whitespace-only changes and moved (cut and
  pasted verbatim) lines do not count. History is walked along first parents,
  so content brought in by a merge commit is attributed to the merge's author
  and timestamp (exact for squash/rebase workflows).

exclusions:
  --exclude globs are tested against the file's repo-relative path at the time
  the line was added. Gitignore-style: `*` stays within one path segment and
  `**` recurses, so use --exclude '**/*.c' for all C files at any depth or
  --exclude 'vendor/**' for a subtree. Excluded lines count neither as added
  nor as churned.

decay fitting:
  --fit-decay models line survival as exponential decay S(t) = exp(-lambda*t)
  and reports the fitted half-life. Churned lines are observed lifetimes;
  lines still alive are right-censored at their current age, so the
  closed-form MLE is lambda = events / total exposure. Fitting ignores
  --window (the point is to discover the repo's natural decay timescale) and
  walks the full history even past --until. --decay-plot PATH additionally
  writes a graph (empirical Kaplan-Meier survival curve plus the fitted
  exponential) to PATH (.png, .svg, or .pdf) and implies --fit-decay.

  --group-by {author,filetype} stratifies the fit: one decay rate per group
  plus a likelihood-ratio test of the shared-rate model against the
  per-group model, stat = 2*(sum_g d_g*ln(d_g/E_g) - D*ln(D/E)) with
  df = G-1 (d = churn events, E = exposure). A small p-value means the
  grouping variable changes the fitted lifetime. Implies --fit-decay.

cross-author churn:
  --churn-heatmap TARGET renders a matrix of churn counts with the line's
  original author on rows and the churning author on columns (the diagonal
  is self-churn). TARGET is an image path (.png/.svg/.pdf) or the literal
  'live' to open an interactive matplotlib window.
"""


def parse_duration(text: str) -> int:
    """'30d', '12h', '2w' -> seconds."""
    match = re.fullmatch(r"(\d+)([hdw])", text.strip())
    if not match:
        raise argparse.ArgumentTypeError(
            f"invalid duration {text!r} (expected e.g. '30d', '2w', '12h')"
        )
    return int(match.group(1)) * _DURATION_UNITS[match.group(2)]


def parse_date(text: str) -> int:
    """ISO date or datetime -> epoch seconds (UTC when no zone given)."""
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid date {text!r} (expected ISO format, e.g. 2026-01-31)"
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="code-churn",
        description="Measure code churn from git history.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("repo", nargs="?", default=".", help="path to a git repository")
    parser.add_argument(
        "--window",
        type=parse_duration,
        default="30d",
        metavar="DURATION",
        help="churn window after a line is added (default: 30d)",
    )
    parser.add_argument(
        "--since", type=parse_date, metavar="DATE", help="only analyze commits on/after"
    )
    parser.add_argument(
        "--until",
        type=parse_date,
        metavar="DATE",
        help="only analyze commits on/before",
    )
    parser.add_argument("--rev", default="HEAD", help="tip revision (default: HEAD)")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="GLOB",
        help="exclude files whose add-time path matches this gitignore-style "
        "glob (repeatable; `**` recurses, e.g. '**/*.c', 'vendor/**')",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json", "csv"],
        default="table",
        help="output format (default: table)",
    )
    parser.add_argument(
        "--by",
        choices=["file", "author", "summary"],
        help="restrict output to one section (table default: all three; "
        "csv default: file)",
    )
    parser.add_argument(
        "--fit-decay",
        action="store_true",
        help="fit an exponential decay to line survival and report the "
        "half-life (ignores --window; see epilog)",
    )
    parser.add_argument(
        "--decay-plot",
        metavar="PATH",
        help="write the survival curve and fitted decay as an image "
        "(.png/.svg/.pdf); implies --fit-decay",
    )
    parser.add_argument(
        "--group-by",
        choices=["author", "filetype"],
        help="stratify the decay fit per group and run a likelihood-ratio "
        "test against the shared-rate fit; implies --fit-decay (see epilog)",
    )
    parser.add_argument(
        "--churn-heatmap",
        metavar="TARGET",
        help="render the origin-author x churn-author heatmap; TARGET is an "
        "image path (.png/.svg/.pdf) or 'live' for an interactive window",
    )
    parser.add_argument("--output", metavar="PATH", help="write output to a file")
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.window <= 0:
        print("error: --window must be positive", file=sys.stderr)
        return 2
    if args.since is not None and args.until is not None and args.since > args.until:
        print("error: --since is after --until", file=sys.stderr)
        return 2
    for pattern in args.exclude:
        try:
            if not pattern:
                raise ValueError("empty pattern")
            PurePosixPath("x").full_match(pattern)
        except ValueError:
            print(f"error: invalid --exclude pattern {pattern!r}", file=sys.stderr)
            return 2

    if args.decay_plot and Path(args.decay_plot).suffix.lower() not in _PLOT_EXTENSIONS:
        print(
            "error: --decay-plot must end in one of: " + ", ".join(_PLOT_EXTENSIONS),
            file=sys.stderr,
        )
        return 2

    heatmap_live = args.churn_heatmap == "live"
    if (
        args.churn_heatmap
        and not heatmap_live
        and Path(args.churn_heatmap).suffix.lower() not in _PLOT_EXTENSIONS
    ):
        print(
            "error: --churn-heatmap must be 'live' or end in one of: "
            + ", ".join(_PLOT_EXTENSIONS),
            file=sys.stderr,
        )
        return 2

    fit_requested = (
        args.fit_decay or args.decay_plot is not None or args.group_by is not None
    )

    config = AnalysisConfig(
        repo_path=args.repo,
        rev=args.rev,
        window_seconds=args.window,
        since=args.since,
        until=args.until,
        exclude=tuple(args.exclude),
        collect_survival=fit_requested,
        survival_group_by=cast("GroupBy | None", args.group_by),
    )
    try:
        result = analyze(config)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    decay_fit = fit_exponential(result.survival) if result.survival else None
    stratified = (
        fit_stratified(result.survival_by_group, args.group_by)
        if result.survival_by_group is not None
        else None
    )
    report = build_report(
        config,
        result,
        generated_at=int(time.time()),
        decay=decay_fit,
        stratified=stratified,
    )

    if args.decay_plot or args.churn_heatmap:
        # Lazy import: only plotting runs pay the matplotlib cost.
        from code_churn import plot

        plot.configure(live=heatmap_live)
        if heatmap_live and not plot.is_interactive():
            print(
                "warning: no interactive matplotlib backend available; "
                "nothing will be displayed",
                file=sys.stderr,
            )

    if args.decay_plot:
        survival = result.survival
        assert survival is not None  # collect_survival was set
        if not survival.event_lifetimes and not survival.censored_ages:
            print("error: no in-range lines; nothing to plot", file=sys.stderr)
            return 1
        plot.render_decay_plot(survival, decay_fit, args.decay_plot)

    if args.churn_heatmap:
        if not report.events:
            print("error: no churn events; nothing to plot", file=sys.stderr)
            return 1
        plot.render_churn_heatmap(
            churn_matrix(report.events),
            path=None if heatmap_live else args.churn_heatmap,
        )

    if args.format == "json":
        output = render_json(report)
    elif args.format == "csv":
        output = render_csv(report, args.by or "file")
    else:
        sections = [args.by] if args.by else ["summary", "file", "author"]
        output = render_table(report, sections)

    if args.output:
        Path(args.output).write_text(output)
    else:
        sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
