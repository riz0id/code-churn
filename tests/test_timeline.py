"""Unit tests for the expanding-window timeline construction (pure)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from code_churn.timeline import build_timeline, month_cutoffs, survival_at
from tests.conftest import DAY, day

LIFESPANS = [(0, 40 * DAY), (0, None), (20 * DAY, None)]


def test_survival_at_before_death_censors_the_doomed_line() -> None:
    # The line that will die at day 40 is still alive at the day-30 cutoff:
    # it must be censored at its current age, not counted as a death.
    data = survival_at(LIFESPANS, 30 * DAY)
    assert data.event_lifetimes == []
    assert sorted(data.censored_ages) == [10 * DAY, 30 * DAY, 30 * DAY]


def test_survival_at_death_cutoff_records_the_event() -> None:
    data = survival_at(LIFESPANS, 40 * DAY)
    assert data.event_lifetimes == [40 * DAY]
    assert sorted(data.censored_ages) == [20 * DAY, 40 * DAY]


def test_survival_at_unborn_lines_are_absent() -> None:
    data = survival_at(LIFESPANS, 10 * DAY)
    assert data.event_lifetimes == []
    assert sorted(data.censored_ages) == [10 * DAY, 10 * DAY]


def test_survival_at_clamps_nonmonotonic_clock() -> None:
    # Death recorded before birth (non-monotonic commit clock): lifetime 0.
    data = survival_at([(100, 50)], 200)
    assert data.event_lifetimes == [0]


def test_month_cutoffs_spans_year_rollover() -> None:
    # T0 = 2023-11-14 UTC; day(60) lands in January 2024.
    cutoffs = month_cutoffs(day(0), day(60))
    assert [label for _, label in cutoffs] == ["2023-11", "2023-12", "2024-01"]
    december_start = int(datetime(2023, 12, 1, tzinfo=timezone.utc).timestamp())
    assert cutoffs[0][0] == december_start - 1
    # The final partial month is cut at observation_end exactly.
    assert cutoffs[-1][0] == day(60)


def test_month_cutoffs_single_month() -> None:
    cutoffs = month_cutoffs(day(0), day(1))
    assert cutoffs == [(day(1), "2023-11")]


def test_build_timeline_frames_hand_computed() -> None:
    timeline = build_timeline([(day(0), day(35)), (day(0), None)], day(40))
    assert timeline.first_birth == day(0)
    assert timeline.observation_end == day(40)
    assert [frame.label for frame in timeline.frames] == ["2023-11", "2023-12"]

    november, december = timeline.frames
    # No deaths observed yet in November: both lines censored, no fit.
    assert november.fit is None
    assert (november.deaths, november.alive) == (0, 2)
    assert november.churned_in_month == 0

    # December's cutoff is observation_end: one death at 35 days, one line
    # censored at 40 days -> lambda = 1 / (75 days of exposure).
    assert december.cutoff == day(40)
    assert (december.deaths, december.alive) == (1, 1)
    assert december.churned_in_month == 1
    assert december.fit is not None
    assert december.fit.lam == pytest.approx(1 / (75 * DAY))


def test_build_timeline_empty_input() -> None:
    timeline = build_timeline([], day(0))
    assert timeline.frames == []
    assert timeline.observation_end == day(0)
