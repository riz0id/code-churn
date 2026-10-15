"""Fit a decay function to line survival. Stdlib only, no pygit2.

Every churned line is an observed "death" with an exact lifetime; every line
still alive at the end of the walk is a right-censored observation. For an
exponential survival model S(t) = exp(-lambda*t) with right-censoring, the
maximum-likelihood estimate is closed-form:

    lambda = deaths / total exposure
    total exposure = sum of death lifetimes + sum of censored ages

The Kaplan-Meier estimator gives the model-free empirical survival curve the
fit can be judged against.

Stratified fits test whether a grouping variable (author, file type) changes
the decay rate: fit one shared lambda versus one lambda per group and compare
with a likelihood-ratio test. Both MLEs are closed-form, so the statistic is
too (d_g = deaths in group g, E_g = exposure in group g; D, E their totals
over the G groups with exposure; d_g = 0 terms drop out):

    stat = 2 * (sum_g d_g*ln(d_g/E_g) - D*ln(D/E)),  df = G - 1

Under the shared-rate null the statistic is chi-square distributed with df
degrees of freedom; a small p-value means the grouping variable changes the
reliability lifetime.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

DAY = 86_400

# Fits are reported with caveats (never suppressed) when the data is thin:
# under 30 days of observed history, or a half-life so far past the oldest
# observation that it is pure extrapolation (both after git-halflife).
MIN_HISTORY_SECONDS = 30 * DAY
EXTRAPOLATION_FACTOR = 4


@dataclass(frozen=True, slots=True)
class SurvivalData:
    event_lifetimes: list[int]  # seconds from add to churn, one per death
    censored_ages: list[int]  # seconds from add to observation end, survivors


@dataclass(frozen=True, slots=True)
class DecayFit:
    lam: float  # decay rate, per second
    half_life_seconds: float
    events: int
    censored: int
    exposure_seconds: int
    caveats: tuple[str, ...]


def fit_exponential(data: SurvivalData) -> DecayFit | None:
    """Censored-exponential MLE; None when there is nothing to fit."""
    events = len(data.event_lifetimes)
    exposure = sum(data.event_lifetimes) + sum(data.censored_ages)
    if events == 0 or exposure <= 0:
        return None

    lam = events / exposure
    half_life = math.log(2) / lam

    oldest = max(max(data.event_lifetimes), max(data.censored_ages, default=0))
    caveats: list[str] = []
    if oldest < MIN_HISTORY_SECONDS:
        caveats.append(f"short history: oldest observation is {oldest / DAY:.1f} days")
    if half_life > EXTRAPOLATION_FACTOR * oldest:
        caveats.append(
            "extrapolated: fitted half-life exceeds "
            f"{EXTRAPOLATION_FACTOR}x the oldest observation"
        )

    return DecayFit(
        lam=lam,
        half_life_seconds=half_life,
        events=events,
        censored=len(data.censored_ages),
        exposure_seconds=exposure,
        caveats=tuple(caveats),
    )


@dataclass(frozen=True, slots=True)
class LRTResult:
    statistic: float
    df: int
    p_value: float


@dataclass(frozen=True, slots=True)
class GroupFit:
    """Per-stratum observation counts plus the fit (None when no deaths)."""

    events: int
    censored: int
    exposure_seconds: int
    fit: DecayFit | None


@dataclass(frozen=True, slots=True)
class StratifiedDecay:
    group_by: str  # "author" | "filetype"
    groups: dict[str, GroupFit]  # insertion order = sorted labels
    lrt: LRTResult | None  # None when < 2 groups with exposure or no deaths


def fit_stratified(
    survival_by_group: dict[str, SurvivalData], group_by: str
) -> StratifiedDecay | None:
    """Per-group censored-exponential MLEs plus the shared-rate LRT.

    Groups with zero exposure stay in the result but are excluded from the
    test; groups with exposure but no deaths still contribute exposure to the
    pooled rate and count toward the degrees of freedom.
    """
    if not survival_by_group:
        return None

    groups: dict[str, GroupFit] = {}
    counts: list[tuple[int, int]] = []  # (d_g, E_g) over groups with exposure
    for label in sorted(survival_by_group):
        data = survival_by_group[label]
        d = len(data.event_lifetimes)
        exposure = sum(data.event_lifetimes) + sum(data.censored_ages)
        groups[label] = GroupFit(
            events=d,
            censored=len(data.censored_ages),
            exposure_seconds=exposure,
            fit=fit_exponential(data),
        )
        if exposure > 0:
            counts.append((d, exposure))

    lrt: LRTResult | None = None
    total_deaths = sum(d for d, _ in counts)
    total_exposure = sum(e for _, e in counts)
    if len(counts) >= 2 and total_deaths >= 1:
        stat = 2.0 * (
            sum(d * math.log(d / e) for d, e in counts if d > 0)
            - total_deaths * math.log(total_deaths / total_exposure)
        )
        stat = max(0.0, stat)
        df = len(counts) - 1
        lrt = LRTResult(statistic=stat, df=df, p_value=chi2_sf(stat, df))

    return StratifiedDecay(group_by=group_by, groups=groups, lrt=lrt)


def _gamma_p_series(s: float, x: float) -> float:
    """Regularized lower incomplete gamma P(s, x) by power series (x < s+1)."""
    term = 1.0 / s
    total = term
    n = s
    for _ in range(500):
        n += 1.0
        term *= x / n
        total += term
        if abs(term) < abs(total) * 1e-15:
            break
    return total * math.exp(-x + s * math.log(x) - math.lgamma(s))


def _gamma_q_contfrac(s: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(s, x) by Lentz's continued
    fraction (x >= s+1)."""
    tiny = 1e-300
    b = x + 1.0 - s
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - s)
        b += 2.0
        d = an * d + b
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = b + an / (c if abs(c) > tiny else tiny)
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h * math.exp(-x + s * math.log(x) - math.lgamma(s))


def chi2_sf(x: float, df: int) -> float:
    """P(chi2_df > x): the regularized upper incomplete gamma Q(df/2, x/2)."""
    if x <= 0.0:
        return 1.0
    s = df / 2.0
    half_x = x / 2.0
    if half_x < s + 1.0:
        return max(0.0, min(1.0, 1.0 - _gamma_p_series(s, half_x)))
    return max(0.0, min(1.0, _gamma_q_contfrac(s, half_x)))


def kaplan_meier(data: SurvivalData) -> list[tuple[int, float]]:
    """Product-limit survival curve as step points (age seconds, S(age)).

    Starts at (0, 1.0); one point per distinct death time. The standard tie
    convention applies: an observation censored at time t is still at risk
    for deaths at t.
    """
    deaths = Counter(data.event_lifetimes)
    censored = Counter(data.censored_ages)
    at_risk = len(data.event_lifetimes) + len(data.censored_ages)

    points = [(0, 1.0)]
    survival = 1.0
    for t in sorted(deaths.keys() | censored.keys()):
        d = deaths.get(t, 0)
        if d:
            survival *= 1.0 - d / at_risk
            points.append((t, survival))
        at_risk -= d + censored.get(t, 0)
    return points
