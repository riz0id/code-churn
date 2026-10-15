"""Render a RepoReport as a table, JSON, or CSV. No pygit2, stdlib only."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict
from datetime import datetime, timezone

from code_churn.decay import GroupFit, StratifiedDecay
from code_churn.model import AuthorStats, GroupStats, RepoReport

Section = str  # "summary" | "file" | "author" (validated by the CLI)


def _rate(stats: GroupStats) -> str:
    rate = stats.churn_rate
    return "-" if rate is None else f"{rate * 100:.1f}%"


def _share(share: float | None) -> str:
    return "-" if share is None else f"{share * 100:.1f}%"


def _sorted_files(report: RepoReport) -> list[tuple[str, GroupStats]]:
    return sorted(report.files.items(), key=lambda item: (-item[1].churned, item[0]))


def _sorted_authors(report: RepoReport) -> list[tuple[str, AuthorStats]]:
    return sorted(
        report.authors.items(),
        key=lambda item: (-item[1].received.churned, item[0]),
    )


def _render_columns(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    lines = [
        "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip(),
        "  ".join("-" * widths[i] for i in range(len(headers))),
    ]
    for row in rows:
        lines.append(
            "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip()
        )
    return "\n".join(lines)


def _summary_lines(report: RepoReport) -> list[str]:
    config = report.config
    window_days = config.window_seconds / 86_400
    date = datetime.fromtimestamp(report.generated_at, tz=timezone.utc)
    lines = [
        f"repo: {config.repo_path} @ {config.rev}",
        f"window: {window_days:g} days"
        + (
            f", range: {_iso(config.since)} .. {_iso(config.until)}"
            if config.since is not None or config.until is not None
            else ""
        ),
        f"generated: {date:%Y-%m-%d %H:%M} UTC",
    ]
    if config.exclude:
        lines.append("exclude: " + ", ".join(config.exclude))
    lines += [
        "",
        f"lines added:   {report.totals.added}",
        f"lines churned: {report.totals.churned}",
        f"churn rate:    {_rate(report.totals)}",
    ]
    if config.collect_survival:
        lines += _decay_lines(report)
    if report.stratified_decay is not None:
        lines += _stratified_lines(report)
    return lines


def _decay_lines(report: RepoReport) -> list[str]:
    fit = report.decay
    if fit is None:
        return ["fitted half-life: - (no churn events observed)"]
    lines = [
        f"fitted half-life: {fit.half_life_seconds / 86_400:.1f} days "
        f"(λ≈{fit.lam * 86_400:.4g}/day; "
        f"{fit.events} churned, {fit.censored} surviving)"
    ]
    lines += [f"  caveat: {caveat}" for caveat in fit.caveats]
    return lines


def _sorted_groups(stratified: StratifiedDecay) -> list[tuple[str, GroupFit]]:
    return sorted(
        stratified.groups.items(), key=lambda item: (-item[1].events, item[0])
    )


def _stratified_lines(report: RepoReport) -> list[str]:
    stratified = report.stratified_decay
    assert stratified is not None
    headers = ["group", "half-life", "λ/day", "events", "surviving"]
    rows = []
    caveat_lines = []
    for label, group in _sorted_groups(stratified):
        fit = group.fit
        if fit is None:
            half_life, lam = "-", "-"
        else:
            half_life = f"{fit.half_life_seconds / 86_400:.1f} days"
            lam = f"{fit.lam * 86_400:.4g}"
            caveat_lines += [f"  caveat ({label}): {c}" for c in fit.caveats]
        rows.append([label, half_life, lam, str(group.events), str(group.censored)])
    lines = [
        "",
        f"decay by {stratified.group_by}:",
        _render_columns(headers, rows),
    ]
    lines += caveat_lines
    lrt = stratified.lrt
    if lrt is None:
        lines.append("likelihood-ratio test: - (needs at least two groups with data)")
    else:
        p = "p<0.001" if lrt.p_value < 0.001 else f"p={lrt.p_value:.4g}"
        lines.append(
            "likelihood-ratio test (shared λ vs per-group λ): "
            f"χ²={lrt.statistic:.4g}, df={lrt.df}, {p}"
        )
    return lines


def _stratified_payload(report: RepoReport) -> dict[str, object] | None:
    stratified = report.stratified_decay
    if stratified is None:
        return None
    lrt = stratified.lrt
    return {
        "group_by": stratified.group_by,
        "groups": [
            {
                "group": label,
                "events": group.events,
                "censored": group.censored,
                "exposure_days": group.exposure_seconds / 86_400,
                "half_life_days": (
                    None if group.fit is None else group.fit.half_life_seconds / 86_400
                ),
                "lambda_per_day": (
                    None if group.fit is None else group.fit.lam * 86_400
                ),
                "caveats": [] if group.fit is None else list(group.fit.caveats),
            }
            for label, group in _sorted_groups(stratified)
        ],
        "lrt": (
            None
            if lrt is None
            else {
                "statistic": lrt.statistic,
                "df": lrt.df,
                "p_value": lrt.p_value,
            }
        ),
    }


def _decay_payload(report: RepoReport) -> dict[str, object] | None:
    fit = report.decay
    if fit is None:
        return None
    return {
        "half_life_days": fit.half_life_seconds / 86_400,
        "lambda_per_day": fit.lam * 86_400,
        "events": fit.events,
        "censored": fit.censored,
        "exposure_days": fit.exposure_seconds / 86_400,
        "caveats": list(fit.caveats),
    }


def _iso(epoch: int | None) -> str:
    if epoch is None:
        return "-"
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d")


def _file_rows(report: RepoReport) -> list[list[str]]:
    return [
        [path, str(stats.added), str(stats.churned), _rate(stats)]
        for path, stats in _sorted_files(report)
    ]


def _author_rows(report: RepoReport) -> list[list[str]]:
    return [
        [
            author,
            str(stats.received.added),
            str(stats.received.churned),
            _rate(stats.received),
            str(stats.caused),
            _share(report.caused_share(author)),
        ]
        for author, stats in _sorted_authors(report)
    ]


FILE_HEADERS = ["file", "added", "churned", "churn rate"]
AUTHOR_HEADERS = [
    "author",
    "added",
    "churned",
    "churn rate",
    "caused",
    "caused share",
]


def render_table(report: RepoReport, sections: list[Section]) -> str:
    parts: list[str] = []
    if "summary" in sections:
        parts.append("\n".join(_summary_lines(report)))
    if "file" in sections and report.files:
        parts.append("per file:\n" + _render_columns(FILE_HEADERS, _file_rows(report)))
    if "author" in sections and report.authors:
        parts.append(
            "per author:\n" + _render_columns(AUTHOR_HEADERS, _author_rows(report))
        )
    return "\n\n".join(parts) + "\n"


def render_json(report: RepoReport) -> str:
    payload = {
        "config": {
            "repo_path": report.config.repo_path,
            "rev": report.config.rev,
            "window_seconds": report.config.window_seconds,
            "since": report.config.since,
            "until": report.config.until,
            "exclude": list(report.config.exclude),
        },
        "generated_at": report.generated_at,
        "summary": {
            "added": report.totals.added,
            "churned": report.totals.churned,
            "churn_rate": report.totals.churn_rate,
        },
        # Present only when fitting was requested; null when nothing to fit.
        **({"decay": _decay_payload(report)} if report.config.collect_survival else {}),
        # Present only when a stratified fit was requested.
        **(
            {"stratified_decay": _stratified_payload(report)}
            if report.stratified_decay is not None
            else {}
        ),
        "files": [
            {
                "path": path,
                "added": stats.added,
                "churned": stats.churned,
                "churn_rate": stats.churn_rate,
            }
            for path, stats in _sorted_files(report)
        ],
        "authors": [
            {
                "author": author,
                "added": stats.received.added,
                "churned": stats.received.churned,
                "churn_rate": stats.received.churn_rate,
                "caused": stats.caused,
                "caused_share": report.caused_share(author),
            }
            for author, stats in _sorted_authors(report)
        ],
        "events": [asdict(event) for event in report.events],
    }
    return json.dumps(payload, indent=2) + "\n"


def render_csv(report: RepoReport, section: Section) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    if section == "file":
        writer.writerow(["path", "added", "churned", "churn_rate"])
        for path, stats in _sorted_files(report):
            writer.writerow([path, stats.added, stats.churned, stats.churn_rate])
    elif section == "author":
        writer.writerow(
            ["author", "added", "churned", "churn_rate", "caused", "caused_share"]
        )
        for author, astats in _sorted_authors(report):
            writer.writerow(
                [
                    author,
                    astats.received.added,
                    astats.received.churned,
                    astats.received.churn_rate,
                    astats.caused,
                    report.caused_share(author),
                ]
            )
    else:  # summary
        writer.writerow(["added", "churned", "churn_rate"])
        writer.writerow(
            [report.totals.added, report.totals.churned, report.totals.churn_rate]
        )
    return buffer.getvalue()
