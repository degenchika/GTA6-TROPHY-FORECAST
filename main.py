"""GTA VI trophy rarity forecaster.

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, five
quantiles (5%, 25%, 50%, 75%, 95%) of its earn rate on Exophase at the case's
target time, from the house recorder's history of reads of the game's trophy
page.  Standard library only; no network; deterministic.

How it works
------------
An earn rate is a ratio:  rate = earners / players_tracked.  Both sides move,
and they move for different reasons, so the program models them separately
and divides at the target:

* players_tracked -- the denominator.  After a launch it grows fast and then
  decelerates as the launch surge passes.  Modelled as a power law in time
  since release:  log P grows linearly in log(1 + hours since release).

* earners of trophy i -- the numerator.  A trophy's earners grow as the
  player base plays through the game: early content (Chapter 1, fishing, the
  first side activities) is earned quickly by almost everyone, late content
  slowly and by few.  Modelled the same way: log E_i grows linearly in
  log(1 + hours since release), fitted per trophy.

Both fits use only the case's own history (the last 24 hours of distinct
reads, or all of them when the history is shorter), by least squares.  The
fitted slopes are shrunk toward a prior by how much the data actually says
(a t-statistic weight), so a flat or tiny history falls back to the prior
instead of trusting a meaningless fit:

* prior for players: slope 2.0 -- Exophase's first refresh of the game's
  counts, which the rules record, multiplied the tracked players by about
  six in four and a half hours, and the tracked base kept growing steeply
  into the second day; a power law with slope ~2 in log(1 + hours) matches
  that shape.
* prior for a trophy's earners: slope 3 * ln(first rate / 0.78), floored at
  0 -- trophies that were already common on launch night are the early
  ones, and their earners keep growing fastest; trophies nobody had yet are
  assumed to grow only with the player base.

The forecast rate is E(target) / P(target).  The spread around the median
combines the random-walk term (0.5 logit per sqrt-day, as in the house
baseline), the fit's own residual propagated to the target, a floor that
grows with the horizon, and extra width when the prior (rather than the
data) is doing the work.  Quantiles are the median's logit shifted by the
normal quantiles times that spread.

Edge cases: a page that has not refreshed shows the same numbers in every
read, so the history collapses to a single distinct snapshot and the prior
takes over entirely -- that is exactly the practice cases.  For such a stale
page the projection also allows one refresh jump in the tracked players
(about sixfold, the first refresh the rules record), but only when the
page's age plus the horizon reaches the refresh interval of about four and
a half hours: a short horizon ending before the next refresh is due must
not jump.  Rates of 0% or 100% are handled by clamping before any logit
(the scorer clamps the same way), and the projected rate is kept inside
[0.005, 99.995] percent.

Usage (outside the grader):  mkdir -p out && MAKEFUN_INPUT_DIR=input \
MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a helper
# module shipped beside this file would import.  (This entry ships none.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- tuning constants (fixed in advance, see the module docstring) ---------
# Slope of log players_tracked against log(1 + hours since release), used
# as the prior when the history says nothing about player growth.
PRIOR_PLAYERS_SLOPE = 2.0
# Reference launch-night rate (the median trophy's) for the earners prior.
PRIOR_RATE_REFERENCE = 0.78
# How strongly a trophy's launch-night commonness lifts its earners slope.
PRIOR_EARNERS_COEF = 3.0
# Largest earners slope the prior may assign.
PRIOR_EARNERS_MAX = 4.0
# Random-walk spread, logit units per square-root day (house baseline's).
WALK_SIGMA = 0.5
# Horizon floor on the spread: 0.55 * ln(1 + horizon in days).
FLOOR_COEF = 0.55
# Extra spread when the prior, not the data, drives a trophy's slope.
PRIOR_SPREAD = 1.2
# Spread kept for the small chance a refresh lands even when none is due.
NO_REFRESH_SPREAD = 0.15
# Refresh jump for a stale page: Exophase refreshes a game's counts every
# few hours, and the first refresh the rules record multiplied the tracked
# players by about six (638 -> 3,868).  The page was unchanged from 23:48Z
# to 04:00Z and refreshed by 04:30Z, so a refresh comes about every four
# and a half hours.  When the history shows a page that has not refreshed
# at all, the next refresh is due once the page's age plus the horizon
# reaches that interval -- and only then is a jump the right median, since
# a refresh before the target is what the target will show.  A short
# horizon ending well before the next refresh must NOT jump (the holdout
# below caught exactly that mistake).
REFRESH_JUMP = 6.0
# A refresh is due about this often (hours), from the launch-night record.
REFRESH_INTERVAL_HOURS = 4.5
# A page counts as stale when it holds this few distinct snapshots, and
# the jump only applies on a short horizon (longer ones grow by the smooth
# power law instead).
REFRESH_MAX_SNAPS = 1
REFRESH_MAX_HORIZON_HOURS = 6.0
# How far the projected rate may move from the last read, as a ratio.
MOVE_MIN = 0.002
MOVE_MAX = 30.0
# Window of recent history the fits use (hours before the checkpoint).
FIT_WINDOW_HOURS = 24.0
# Damping on the projected slopes: a log-linear fit extrapolates the
# recent growth rate, but real trophy growth saturates (a trophy's rate
# approaches the share of players who will ever earn it), so the fitted
# slope overstates how far the rate travels over a multi-day horizon.
# Slopes are scaled by this much before projecting.
SLOPE_DAMP = 0.7
# Shrinkage strength: weight = t^2 / (t^2 + SHRINK_K).
SHRINK_K = 4.0


def parse_moment(text):
    """An ISO 8601 timestamp (or a plain date) as an aware datetime (UTC)."""
    text = str(text).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def clamp_rate(pct):
    """A percent into the range the logit can take (as the scorer does)."""
    return min(99.99, max(0.01, pct))


def logit_of(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def percent_of(logit_value):
    return 100.0 / (1.0 + math.exp(-logit_value))


def round4(value):
    """Round to 4 decimals, half up, as JavaScript's Math.round does it."""
    scaled = value * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    out = whole / 10000.0
    return int(out) if float(out).is_integer() else out


def least_squares(xs, ys):
    """Ordinary least squares of y on x.

    Returns (slope, intercept, resid_rms, se_slope, x_mean, sxx).  With fewer
    than two points, or no spread in x, the slope is 0 and its standard
    error is infinite (the data says nothing about it).
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, float("inf"), 0.0, 0.0
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    sxx = sum((x - x_mean) ** 2 for x in xs)
    if n < 2 or sxx <= 0.0:
        return 0.0, y_mean, 0.0, float("inf"), x_mean, sxx
    sxy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean
    if n > 2:
        ssr = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
        resid_rms = math.sqrt(ssr / (n - 2))
    else:
        resid_rms = 0.0
    se_slope = resid_rms / math.sqrt(sxx) if resid_rms > 0.0 else 0.0
    return slope, intercept, resid_rms, se_slope, x_mean, sxx


def shrink_toward_prior(slope, se_slope, prior):
    """Blend a fitted slope with its prior by the fit's t-statistic.

    A slope the data pins down (large |t|) is kept; a slope the data cannot
    see (small |t|, or no variation at all) falls back to the prior.  Also
    returns the data weight, which the spread uses to widen when the prior
    is doing the work.  The data's slope is damped first (SLOPE_DAMP):
    a log-linear fit extrapolates the recent growth rate, but real trophy
    growth saturates, so the fitted slope overstates how far the rate
    travels over a multi-day horizon.  The prior is not damped: it is
    already calibrated to the long-horizon shape.
    """
    if se_slope == 0.0:
        # No residual at all: either a perfect fit through two points (trust
        # it only a little) or a flat series (trust it not at all).
        if slope == 0.0:
            return prior, 0.0
        return 0.5 * slope * SLOPE_DAMP + 0.5 * prior, 0.5
    if math.isinf(se_slope):
        return prior, 0.0
    t_stat = abs(slope) / se_slope
    weight = (t_stat * t_stat) / (t_stat * t_stat + SHRINK_K)
    return weight * slope * SLOPE_DAMP + (1.0 - weight) * prior, weight


def distinct_snapshots(reads):
    """Collapse consecutive reads with identical counts into one snapshot.

    Exophase refreshes on its own schedule, so the recorder often reads the
    same page several times in a row; those repeats carry no information
    about growth and would overweight the flat stretches in a fit.
    """
    out = []
    previous_key = None
    for read in reads:
        key = (read.get("players_tracked"), tuple(read.get("rates") or ()))
        if key != previous_key:
            out.append(read)
            previous_key = key
    return out


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(
            f"{name} asks for levels {case['levels']}, not {LEVELS}"
        )
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]

    release = parse_moment(case["game"].get("release_date", "2026-11-19"))
    checkpoint = parse_moment(case["at"])
    target = parse_moment(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0

    def hours_since_release(moment):
        return (moment - release).total_seconds() / 3600.0

    def x_of(moment):
        return math.log1p(max(0.0, hours_since_release(moment)))

    # Recent window of distinct snapshots for the fits.
    snaps = distinct_snapshots(history)
    window_start = checkpoint.timestamp() - FIT_WINDOW_HOURS * 3600.0
    recent = [r for r in snaps if parse_moment(r["slot"]).timestamp() >= window_start]
    if len(recent) < 3:
        recent = snaps
    x_recent = [x_of(parse_moment(r["slot"])) for r in recent]
    x_last = x_of(parse_moment(history[-1]["slot"]))
    x_target = x_of(target)

    # Staleness: how long the displayed page has gone without a refresh.
    # The displayed counts only move when Exophase refreshes (about every
    # REFRESH_INTERVAL_HOURS), so when the page is flat and the next
    # refresh falls after the target, the target shows today's numbers:
    # persistence, with a narrow spread, is the right forecast.
    first_slot = parse_moment(snaps[0]["slot"])
    stale_hours = (checkpoint - first_slot).total_seconds() / 3600.0
    horizon_hours = case["horizon_minutes"] / 60.0
    no_refresh_expected = (
        len(snaps) <= REFRESH_MAX_SNAPS
        and stale_hours + horizon_hours < REFRESH_INTERVAL_HOURS
    )

    # --- players_tracked: fit log P against log(1 + hours since release).
    log_p = []
    for read in recent:
        players = read.get("players_tracked")
        if players is None or players <= 0:
            continue
        log_p.append(math.log(players))
    # Keep the x values aligned with the kept reads.
    x_p = [x for x, read in zip(x_recent, recent)
           if read.get("players_tracked") and read["players_tracked"] > 0]
    p_slope, _p_int, p_resid, p_se, _p_xm, p_sxx = least_squares(x_p, log_p)
    p_slope, p_weight = shrink_toward_prior(
        p_slope, p_se, PRIOR_PLAYERS_SLOPE
    )
    last_players = history[-1].get("players_tracked") or 1
    if no_refresh_expected:
        log_p_target = math.log(last_players)
    else:
        log_p_target = math.log(last_players) + p_slope * (x_target - x_last)
    # A stale page: the displayed counts have not refreshed in the whole
    # history, so the next refresh -- not smooth growth -- is what the
    # target will show.  Take the larger of the smooth projection and one
    # refresh jump (the first refresh the rules record).  This only runs
    # when a refresh is actually due (stale + horizon reaches the
    # interval); otherwise no_refresh_expected above already froze the
    # projection.
    if not no_refresh_expected and (
        len(snaps) <= REFRESH_MAX_SNAPS
        and stale_hours + horizon_hours >= REFRESH_INTERVAL_HOURS
        and horizon_hours <= REFRESH_MAX_HORIZON_HOURS
    ):
        log_p_target = max(
            log_p_target,
            math.log(last_players) + math.log(REFRESH_JUMP),
        )
    # Players cannot shrink and cannot explode without bound.
    log_p_target = min(
        log_p_target,
        math.log(last_players) + math.log(500.0),
    )
    log_p_target = max(log_p_target, math.log(last_players))
    players_target = math.exp(log_p_target)

    # --- each trophy: fit log earners the same way.
    first_rates = snaps[0].get("rates") or []
    rows = []
    for i, trophy in enumerate(trophies):
        # Most recent non-null rate for this trophy (the anchor).
        anchor_rate = None
        for read in reversed(history):
            rates = read.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                anchor_rate = float(rates[i])
                break
        if anchor_rate is None:
            anchor_rate = 1.0
        anchor_rate = min(99.99, max(0.005, anchor_rate))
        earners_last = anchor_rate / 100.0 * last_players

        # Earners series over the recent window (skip nulls and zeros).
        xs_e, ys_e = [], []
        for x_val, read in zip(x_recent, recent):
            rates = read.get("rates") or []
            players = read.get("players_tracked")
            if i >= len(rates) or rates[i] is None or not players:
                continue
            rate = min(99.99, max(0.005, float(rates[i])))
            xs_e.append(x_val)
            ys_e.append(math.log(rate / 100.0 * players))
        e_slope, _e_int, e_resid, e_se, e_xmean, e_sxx = least_squares(xs_e, ys_e)

        # Prior slope from how common the trophy was on launch night.
        first_rate = 0.78
        if i < len(first_rates) and first_rates[i] is not None:
            first_rate = min(99.99, max(0.01, float(first_rates[i])))
        prior_slope = PRIOR_EARNERS_COEF * math.log(first_rate / PRIOR_RATE_REFERENCE)
        prior_slope = min(PRIOR_EARNERS_MAX, max(0.0, prior_slope))
        e_slope, e_weight = shrink_toward_prior(e_slope, e_se, prior_slope)
        if no_refresh_expected:
            # No refresh before the target: the displayed earners do not
            # move either, whatever the underlying growth is.
            e_slope, e_weight = 0.0, 1.0

        log_e_target = math.log(earners_last) + e_slope * (x_target - x_last)
        rate_target = math.exp(log_e_target) / players_target * 100.0

        # Keep the move from the last read inside sane bounds.
        ratio = rate_target / anchor_rate if anchor_rate > 0 else 1.0
        ratio = min(MOVE_MAX, max(MOVE_MIN, ratio))
        rate_target = anchor_rate * ratio
        rate_target = min(99.995, max(0.005, rate_target))
        # A saturated trophy stays saturated: a rate at (essentially) 100%
        # means every tracked player earned it, and the rules record that
        # the one rate at 100% did not fall when the page refreshed.  Do
        # not let a refresh jump dilute it.
        if anchor_rate >= 99.0:
            rate_target = 99.99

        # --- spread: random walk + fit residual + horizon floor + prior width.
        walk = WALK_SIGMA * math.sqrt(max(0.0, horizon_days))
        floor = FLOOR_COEF * math.log1p(max(0.0, horizon_days))
        if e_sxx > 0.0 and len(xs_e) > 2:
            leverage = 1.0 + ((x_target - e_xmean) ** 2) / e_sxx + 1.0 / len(xs_e)
            fit_spread = e_resid * math.sqrt(max(1.0, leverage))
            # The players fit adds its own uncertainty to every trophy.
            if p_sxx > 0.0:
                p_leverage = 1.0 + ((x_target - _p_xm) ** 2) / p_sxx
                fit_spread = math.sqrt(
                    fit_spread * fit_spread
                    + (p_resid * math.sqrt(max(1.0, p_leverage))) ** 2
                )
        else:
            fit_spread = 0.0
        if no_refresh_expected:
            # Narrow spread: only the small chance a refresh lands early.
            prior_width = NO_REFRESH_SPREAD
        else:
            prior_width = PRIOR_SPREAD * (1.0 - e_weight)
        spread = math.sqrt(
            walk * walk + fit_spread * fit_spread + floor * floor
            + prior_width * prior_width
        )
        spread = min(3.0, max(0.15, spread))

        median_logit = logit_of(rate_target)
        quantiles = []
        for z in Z_SCORES:
            q = percent_of(median_logit + z * spread)
            quantiles.append(round4(min(100.0, max(0.0, q))))
        quantiles.sort()
        rows.append({"id": trophy["id"], "q": quantiles})

    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    inputs = sorted(
        p for p in INPUT_DIR.iterdir()
        if p.is_file() and p.name.endswith(".in")
    )
    for path in inputs:
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out_path = OUTPUT_DIR / f"{path.stem}.out"
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
