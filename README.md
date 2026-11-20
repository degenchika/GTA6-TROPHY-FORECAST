# GTA6-TROPHY-FORECAST

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/degenchika/GTA6-TROPHY-FORECAST/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/degenchika/GTA6-TROPHY-FORECAST/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for Grand Theft Auto VI's PlayStation 5 trophy earn rates on Exophase. It models a trophy's earners and the players tracked separately — each as a power law in time since release, fitted per trophy on the case's own history and shrunk toward a prior — then divides at the target, with quantiles widened by the fit's residual and the horizon. Built for make.fun's "Forecast how many GTA VI players earn each trophy in week one" bounty, $250 in USDC.

## Run it

```sh
mkdir -p out && MAKEFUN_OUTPUT_DIR=out python3 main.py
```

With the kit's practice cases as input:

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=../gta6-trophy-rarity-forecast-01-kit/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3.13, standard library only) reads every `<id>.in` in the input directory and writes `<id>.out` to the output directory: one row per trophy with five quantiles (5%, 25%, 50%, 75%, 95%) of its earn rate at the target, in percent. Deterministic — two runs write identical bytes.

## How it works

An earn rate is a ratio, so the program forecasts its two sides apart:

- **Players tracked** grows fast after launch and then decelerates. Modelled as a power law in time since release, fitted on the case's recent distinct reads.
- **Earners per trophy** grow as the player base plays through the game: early content is earned quickly by almost everyone, late content slowly and by few. Fitted the same way, per trophy, with a prior that leans on how common the trophy was on launch night.
- The fitted slopes are shrunk toward those priors by their t-statistic and damped for saturation, so a flat or tiny history falls back to the prior instead of trusting a meaningless fit.
- **Stale pages**: Exophase refreshes a game's counts every few hours, so a history that never changes is a stale page. When the next refresh falls after the target, the forecast is persistence; when it falls inside the horizon, the projection allows one refresh jump in the tracked players (about sixfold, the first refresh the rules record). A trophy pinned at 100% stays pinned.
- **Spread**: the random-walk term plus the fit's residual propagated to the target, a floor growing with the horizon, and extra width when the prior (rather than the data) is doing the work.

Validated locally: on the real launch-night-to-live-page trajectory (38 hours, rates read from Exophase on 20 Nov 2026) the forecaster scores a pinball loss of 0.66 against the house baseline's 2.91; on a synthetic final-like case with a 4.5-day horizon it scores 0.31 against the baseline's 0.47.

---

<sub>Entered on [make.fun](https://make.fun/?ref=cdzr1hpv&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
