"""Unit tests for the survival fit and Kaplan-Meier estimator (pure)."""

from __future__ import annotations

import math

import pytest

from code_churn.decay import (
    DAY,
    SurvivalData,
    chi2_sf,
    fit_exponential,
    fit_stratified,
    kaplan_meier,
)


def test_mle_matches_closed_form() -> None:
    # 2 deaths, 60 seconds of total exposure -> lambda = 2/60 exactly.
    data = SurvivalData(event_lifetimes=[10, 20], censored_ages=[30])
    fit = fit_exponential(data)
    assert fit is not None
    assert fit.lam == pytest.approx(2 / 60)
    assert fit.half_life_seconds == pytest.approx(30 * math.log(2))
    assert (fit.events, fit.censored, fit.exposure_seconds) == (2, 1, 60)


def test_no_events_or_no_exposure_returns_none() -> None:
    assert fit_exponential(SurvivalData([], [100])) is None
    assert fit_exponential(SurvivalData([], [])) is None
    assert fit_exponential(SurvivalData([0], [0])) is None  # zero exposure


def test_clean_fit_has_no_caveats() -> None:
    data = SurvivalData(event_lifetimes=[40 * DAY], censored_ages=[40 * DAY])
    fit = fit_exponential(data)
    assert fit is not None
    # half-life = ln2 * 80d ~ 55d, within 4x the oldest (160d); history 40d.
    assert fit.caveats == ()


def test_short_history_and_extrapolation_caveats() -> None:
    # Oldest observation is 1 day (short history), and heavy censoring pushes
    # the half-life far past it (extrapolation).
    data = SurvivalData(event_lifetimes=[DAY], censored_ages=[DAY] * 100)
    fit = fit_exponential(data)
    assert fit is not None
    assert any("short history" in c for c in fit.caveats)
    assert any("extrapolated" in c for c in fit.caveats)


def test_kaplan_meier_hand_computed() -> None:
    # n=5. Death at 10 (5 at risk): S=4/5. Censored at 15. Death at 20
    # (3 at risk): S=4/5 * 2/3. Censored at 25 and 30 change nothing.
    data = SurvivalData(event_lifetimes=[10, 20], censored_ages=[15, 25, 30])
    points = kaplan_meier(data)
    assert [t for t, _ in points] == [0, 10, 20]
    assert [s for _, s in points] == pytest.approx([1.0, 0.8, 0.8 * 2 / 3])


def test_kaplan_meier_tie_convention() -> None:
    # Censored observations at a death's exact time are still at risk for it.
    data = SurvivalData(event_lifetimes=[10], censored_ages=[10, 10])
    points = kaplan_meier(data)
    assert [t for t, _ in points] == [0, 10]
    assert [s for _, s in points] == pytest.approx([1.0, 2 / 3])


def test_chi2_sf_analytic_identities() -> None:
    # df=2 is exactly exp(-x/2); df=1 is erfc(sqrt(x/2)); df=4 is
    # (1 + x/2)*exp(-x/2). The x values exercise both the series and the
    # continued-fraction branches of the incomplete gamma.
    for x in (0.5, 2.0, 10.0):
        assert chi2_sf(x, 2) == pytest.approx(math.exp(-x / 2))
    for x in (0.1, 3.84, 15.0):
        assert chi2_sf(x, 1) == pytest.approx(math.erfc(math.sqrt(x / 2)))
    for x in (1.0, 8.0):
        assert chi2_sf(x, 4) == pytest.approx((1 + x / 2) * math.exp(-x / 2))


def test_chi2_sf_bounds_and_monotonicity() -> None:
    assert chi2_sf(0.0, 3) == 1.0
    assert chi2_sf(-1.0, 3) == 1.0
    values = [chi2_sf(x, 3) for x in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)]
    assert values == sorted(values, reverse=True)


def test_stratified_lrt_hand_computed() -> None:
    # Group a: 1 death, exposure 20. Group b: 1 death, exposure 40.
    # D=2, E=60; stat = 2*(ln(1/20) + ln(1/40) - 2*ln(2/60)).
    stratified = fit_stratified(
        {"a": SurvivalData([10], [10]), "b": SurvivalData([10], [30])},
        "author",
    )
    assert stratified is not None
    assert stratified.group_by == "author"
    fit_a = stratified.groups["a"].fit
    fit_b = stratified.groups["b"].fit
    assert fit_a is not None and fit_a.lam == pytest.approx(1 / 20)
    assert fit_b is not None and fit_b.lam == pytest.approx(1 / 40)
    lrt = stratified.lrt
    assert lrt is not None
    expected = 2 * (math.log(1 / 20) + math.log(1 / 40) - 2 * math.log(2 / 60))
    assert lrt.statistic == pytest.approx(expected)
    assert lrt.df == 1
    assert 0.0 < lrt.p_value < 1.0


def test_stratified_equal_rates_give_zero_statistic() -> None:
    stratified = fit_stratified(
        {"a": SurvivalData([10], []), "b": SurvivalData([10], [])},
        "author",
    )
    assert stratified is not None
    lrt = stratified.lrt
    assert lrt is not None
    assert lrt.statistic == pytest.approx(0.0, abs=1e-12)
    assert lrt.p_value == pytest.approx(1.0)


def test_stratified_zero_death_group_counts_toward_test() -> None:
    # Group b has exposure but no deaths: no per-group fit, yet its exposure
    # joins the pooled rate and it counts toward the degrees of freedom.
    # D=1, E=90; stat = 2*(ln(1/20) - ln(1/90)) = 2*ln(4.5).
    stratified = fit_stratified(
        {"a": SurvivalData([10], [10]), "b": SurvivalData([], [70])},
        "author",
    )
    assert stratified is not None
    assert stratified.groups["b"].fit is None
    assert stratified.groups["b"].exposure_seconds == 70
    lrt = stratified.lrt
    assert lrt is not None
    assert lrt.df == 1
    assert lrt.statistic == pytest.approx(2 * math.log(4.5))


def test_stratified_zero_exposure_group_excluded_from_test() -> None:
    # Group c has neither deaths nor exposure: listed, but not a stratum of
    # the test (df still 1 for the two groups with data).
    stratified = fit_stratified(
        {
            "a": SurvivalData([10], [10]),
            "b": SurvivalData([10], [30]),
            "c": SurvivalData([], []),
        },
        "author",
    )
    assert stratified is not None
    assert set(stratified.groups) == {"a", "b", "c"}
    assert stratified.groups["c"].fit is None
    lrt = stratified.lrt
    assert lrt is not None
    assert lrt.df == 1


def test_stratified_single_group_has_no_lrt() -> None:
    stratified = fit_stratified({"a": SurvivalData([10], [])}, "author")
    assert stratified is not None
    assert stratified.lrt is None
    assert set(stratified.groups) == {"a"}


def test_stratified_empty_input_returns_none() -> None:
    assert fit_stratified({}, "author") is None
