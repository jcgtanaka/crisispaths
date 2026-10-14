# Methodology

This page explains, in plain language, what crisispaths does between a
scenario spec and a table of path metrics, and which numbers come from data
and which are judgment. It describes the implemented model; it does not claim
the model is right.

Scenarios are conditional what-ifs. Nothing here produces a forecast and no
probability is attached to any scenario.

## 1. Pipeline

1. A **scenario spec** (YAML, or composed from a family) says which factors are
   shocked, by how much, over how long, and what happens afterwards.
2. **Calibration** estimates, once, how your assets load on the factors in calm
   and stress periods, plus residual volatility, correlation and tail
   structure.
3. **Generation** draws one reproducible path per `(seed, i)`: factor paths
   first, then asset returns from the loadings plus fat-tailed residuals.
4. The **harness** runs your strategy (and a benchmark) on history followed by
   the simulated path and records metrics.
5. The **runner and report** repeat this for many paths and summarise with
   bootstrap intervals, always conditional on the scenario.

## 2. Factors: anchored and conditional

A scenario names a small set of factors (equity `EQ`, technology `TECH`,
dollar `USD`, rates `RATES`, credit `CREDIT`, volatility `VOL`, haven `HAVEN`).

- **Anchored factors** have a drawn cumulative move over the crash window. The
  spec gives a distribution for it (fixed, uniform, lognormal loss, or a
  replayed analog window). The factor path is built so that its cumulative move
  over the crash equals the drawn target exactly; innovations are de-meaned and
  Student-t with unit variance, scaled by a volatility envelope `v_t` (a draw
  from `vol_multiplier` during the crash, decaying back to 1 with a half-life
  afterwards).
- **Conditional factors** are not anchored. They follow
  `f_c = Gamma f_a + eta`, where `Gamma` is the regression (no intercept) of
  the unanchored factors on the anchored ones over the stress windows. Without
  an intercept the map carries the crisis-conditional mean, so for example
  Treasuries rally when `rates_credit=history_consistent` because they rallied
  in every historical window.

Assets: `r_i = sum_k beta_ik f_k + eps_i`, with `eps` drawn from a t-copula
with Student-t margins whose scale is the calibrated residual volatility times
the envelope, capped per asset at the largest rolling volatility the asset
itself showed in the calibration windows.

## 3. Loading estimation

Per asset and per regime (calm, stress):

- Ridge regression on the factors and, by default, one lead and one lag
  (Dimson), reporting the sum. The calm prior is zero; the stress prior is the
  calm coefficient vector, so with few stress observations the stress loading
  stays near the calm one. The shrinkage share and standard errors are
  reported.
- An exposure test: if the pooled joint F-test p-value is above 0.10 and
  adjusted R2 below 0.02 the asset is flagged *unaffected* and its loadings are
  set to zero. That is a model claim, not a fact; the report says so.

### Stress imputation (assets with no stress data)

An asset with no data in the stress windows (for example BTC-USD in 2000-2002)
borrows its stress loadings from its nearest neighbours in calm-loading space:

`beta_stress_i = s * beta_calm_i`

where `s` is ONE scalar per imputed asset, a through-origin least-squares fit
over the neighbours and all factors,
`s = sum(stress_nk * calm_nk) / sum(calm_nk^2)`, clipped to `[0.5, 2.0]`. The
clip bounds are a judgment. A per-factor ratio was rejected because a
near-zero neighbour calm beta makes the ratio explode and flip signs. Imputed
rows carry no standard errors, and the report marks them `IMPUTED` with `s`.

## 4. Calm drift

After a V recovery and before the onset the generator block-bootstraps rows
from the calm windows. Raw calm rows carry each asset's historical trend. By
default (`calm_drift="zero"`) each column's calm mean is removed so the blocks
keep volatility, correlation and tail structure but no trend;
`calm_drift="historical"` keeps the raw rows. Scenarios that never draw from
the pool are bit-identical either way.

## 5. Aftermath modes

- `analog`: replay (exact) or block-bootstrap the analog window's factor path.
- `prolonged_bear`: constant drift (annual, a judgment parameter).
- `v_recovery`: a drawn fraction of the log loss is recovered over a stated
  period, then joint calm bootstrap. It is a counterfactual.
- `stagnation`: zero drift with a volatility floor.

## 6. Frequency handling

Generation always runs at the calibration frequency (daily). Other bar clocks
are reached after the fact:

- **Intraday**: the daily return `d` of each session is split into `S` bars by
  a Brownian bridge, `r_k = x_k + w_k (d - sum_j x_j)` with
  `x_k = sigma sqrt(w_k) z_k`, `z` Student-t (degrees of freedom estimated from
  the intraday data, default 5), `w` the estimated variance profile of the
  session (sums to 1, shrunk toward the pooled profile for short histories) and
  `sigma = max(|d|, daily vol)`. The bars sum to `d` exactly. Cross-asset
  correlation inside the day is the daily residual correlation.
- **Coarser**: daily log returns are summed in blocks.

Minute bars are never simulated directly: the central limit theorem would wash
out the fat daily crash tails and memory would explode.

Limits: no jumps, halts or liquidity gaps inside a session; the Epps effect is
not modelled; flat cost model; strategies that react to intraday moves look
better than reality. See [limitations.md](limitations.md).

## 7. Calibrated versus judgment

| Calibrated from data | Judgment (`calibrated: false`, with a rationale) |
|---|---|
| Loadings per regime, their standard errors and shrinkage | Shock sizes of hypothetical crashes (for example 40% median equity loss) |
| Residual volatility, correlation, vol caps | Sign and size of USD, rates and credit moves no window settles |
| Conditional map `Gamma` and `eta` covariance | Aftermath drift and recovery fractions |
| Analog crash windows (replayed history) | Crash length, volatility multiple, copula and factor degrees of freedom |
| Intraday variance profile and t degrees of freedom (when data are given) | Imputation clip bounds `[0.5, 2.0]` |

The report lists every judgment parameter of every scenario and, for a sweep,
shows how far the results move when a judgment axis changes.

## 8. Metrics and inference

Per path: terminal wealth, drawdown from start and from the all-time peak,
time to trough, time to recover (right-censored at the horizon), ruin (equity
at or below `ruin_level` of start), turnover and cost. Across paths: quantiles
and CVaR with percentile-bootstrap intervals, ruin with a Wilson interval,
recovery with Kaplan-Meier. No expected return across scenarios is reported.

### Short math

Asset log return `r_t = B f_t + eps_t`. For an imputed asset
`B_stress = s B_calm`, `s = <stress, calm> / <calm, calm>` over the
neighbours, clipped to `[0.5, 2]`. A strategy's weights `w_t` set at the close
of bar `t` earn `R_{t+1}`; equity updates as
`E_{t+1} = E_t (1 + w_t' R_{t+1} - c |w_t - w_{t-1}|_1)` with
`c = costs_bps * cost_multiplier / 1e4`. Drawdown from the peak at `t` is
`1 - E_t / max(history_peak, max_{u<=t} E_u)`.
