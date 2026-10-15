"""Rebuild the decay fit at historical cutoffs for the animated timeline.

Stdlib only, no pygit2, no matplotlib. Each monthly cutoff T reconstructs the
survival dataset as it was observable at T: churn that had already happened is
a death with an exact lifetime, and every line alive at T is right-censored at
its age then. Refitting per cutoff reuses the closed-form censored-exponential
MLE directly; at O(n) per frame this is dwarfed by the git walk that produced
the data. (If profiling ever shows it matters: sorting deaths by death time
and births by birth time gives prefix sums for deaths-so-far and exposure,
making each cutoff O(log n) — at the cost of re-deriving fit_exponential's
caveat logic by hand.)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from code_churn.decay import DecayFit, SurvivalData, fit_exponential

Lifespan = tuple[int, int | None]  # (birth, death | None = alive)


@dataclass(frozen=True, slots=True)
class TimelineFrame:
    cutoff: int  # epoch seconds; last instant included in this frame
    label: str  # "%Y-%m" of the frame's calendar month (UTC)
    fit: DecayFit | None  # censored-exponential fit at this cutoff
    deaths: int  # cumulative deaths observed by cutoff
    alive: int  # lines born <= cutoff and not yet dead
    churned_in_month: int  # deaths falling inside this frame's month


@dataclass(frozen=True, slots=True)
class Timeline:
    frames: list[TimelineFrame]
    first_birth: int  # epoch seconds of the oldest line birth
    observation_end: int  # newest walked commit time


def survival_at(lifespans: list[Lifespan], cutoff: int) -> SurvivalData:
    """The survival dataset as it was observable at `cutoff`.

    Deaths at/before cutoff are events (lifetime = death - birth, clamped
    >= 0 against non-monotonic clocks); lines born at/before cutoff and not
    yet dead are right-censored at cutoff - birth. Lines born after cutoff
    do not exist yet.
    """
    event_lifetimes: list[int] = []
    censored_ages: list[int] = []
    for birth, death in lifespans:
        if birth > cutoff:
            continue
        if death is not None and death <= cutoff:
            event_lifetimes.append(max(0, death - birth))
        else:
            censored_ages.append(cutoff - birth)
    return SurvivalData(event_lifetimes=event_lifetimes, censored_ages=censored_ages)


def month_cutoffs(first_birth: int, observation_end: int) -> list[tuple[int, str]]:
    """UTC calendar-month cutoffs covering [first_birth, observation_end].

    One (cutoff, "%Y-%m") pair per month; cutoff is the month's last epoch
    second, except the final (possibly partial) month, cut at
    observation_end.
    """
    cutoffs: list[tuple[int, str]] = []
    start = datetime.fromtimestamp(first_birth, tz=timezone.utc)
    cursor = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    while int(cursor.timestamp()) <= observation_end:
        next_start = datetime(
            cursor.year + (cursor.month == 12),
            cursor.month % 12 + 1,
            1,
            tzinfo=timezone.utc,
        )
        cutoff = min(int(next_start.timestamp()) - 1, observation_end)
        cutoffs.append((cutoff, f"{cursor:%Y-%m}"))
        cursor = next_start
    return cutoffs


def build_timeline(lifespans: list[Lifespan], observation_end: int) -> Timeline:
    """One frame per calendar month, each refit from data observable then."""
    if not lifespans:
        return Timeline(
            frames=[], first_birth=observation_end, observation_end=observation_end
        )

    first_birth = min(birth for birth, _ in lifespans)
    frames: list[TimelineFrame] = []
    previous_deaths = 0
    for cutoff, label in month_cutoffs(first_birth, observation_end):
        data = survival_at(lifespans, cutoff)
        deaths = len(data.event_lifetimes)
        frames.append(
            TimelineFrame(
                cutoff=cutoff,
                label=label,
                fit=fit_exponential(data),
                deaths=deaths,
                alive=len(data.censored_ages),
                # Cumulative deaths are monotone in cutoff, so the diff is
                # exactly this month's deaths.
                churned_in_month=deaths - previous_deaths,
            )
        )
        previous_deaths = deaths

    return Timeline(
        frames=frames, first_birth=first_birth, observation_end=observation_end
    )
