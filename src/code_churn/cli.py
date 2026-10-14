"""Command-line interface for code-churn."""

from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from code_churn import __version__
from code_churn.engine import analyze
from code_churn.formatters import render_csv, render_json, render_table
from code_churn.model import AnalysisConfig
from code_churn.report import build_report

_DURATION_UNITS = {
    "h": 3_600,
    "d": 86_400,
    "w": 7 * 86_400,
}

EPILOG = """\
churn definition:
  A line added in one commit that is modified or deleted by a later commit
  within --window of being added. Whitespace-only changes and moved (cut and
  pasted verbatim) lines do not count. History is walked along first parents,
  so content brought in by a merge commit is attributed to the merge's author
  and timestamp (exact for squash/rebase workflows).
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

    config = AnalysisConfig(
        repo_path=args.repo,
        rev=args.rev,
        window_seconds=args.window,
        since=args.since,
        until=args.until,
    )
    try:
        result = analyze(config)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    report = build_report(config, result, generated_at=int(time.time()))

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
