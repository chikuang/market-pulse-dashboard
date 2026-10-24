"""Predeclared evaluation windows, causal regime labels and paired inference."""
import numpy as np
import pandas as pd

BACKTEST_START = "2020-01-01"
MIN_BACKTEST_YEARS = 5
BOOTSTRAP_REPETITIONS = 4000
MIN_SLICE_N = 20
POLICY = {
    "start": BACKTEST_START, "minimum_years": MIN_BACKTEST_YEARS,
    "bull": "close >= own-market SMA200 at forecast origin",
    "bear": "close < own-market SMA200 at forecast origin",
    "high_volatility": "20-session realized volatility >= prior 756-session 75th percentile (minimum 252 observations; threshold shifted one session)",
    "regime_overlap": "Bull/bear partition the sample; high/normal volatility are a separate, overlapping partition",
    "eras": {"early": "origin before 2024-01-01", "recent": "origin on/after 2024-01-01; already inspected in prior research, not an untouched holdout"},
    "minimum_slice_n": MIN_SLICE_N,
    "stability_rule": "Overall relative MAE gain >=1%, positive unadjusted 95% lower bound, Holm-adjusted one-sided p<0.05, positive gains in bull/bear/high/normal volatility and both eras with >=20 forecasts each, >=5 complete origin-years and gains in >=2/3 of those years",
    "inference": "Paired circular moving-block bootstrap, 5 chronological forecasts per block, 4000 repetitions, seed=42. Resample regime masks jointly on the original timeline; do not concatenate disjoint regime episodes.",
    "multiplicity": "Holm across 42 overall tests (7 combinations x 3 primary markets x 2 horizons). Nasdaq is a separate 14-test exploratory family; slice intervals are unadjusted diagnostics.",
    "selection": "Research screening only; no automatic model switch and no genuinely untouched holdout",
}


def evaluation_start(index, horizon, min_train, test_window, backtest_start=None):
    if backtest_start is None:
        return max(min_train+horizon, len(index)-test_window-horizon)
    position = int(index.searchsorted(pd.Timestamp(backtest_start)))
    if position >= len(index):
        raise ValueError("Backtest start is after available data")
    # Do not move the anchor as new days arrive, or shift it to conceal insufficient training.
    return position


def validate_span(records, minimum_years):
    if not records:
        raise ValueError("No out-of-sample forecasts")
    if minimum_years and pd.Timestamp(records[0]['date']) + pd.DateOffset(years=minimum_years) > pd.Timestamp(records[-1]['target_date']):
        raise ValueError(f"At least {minimum_years} calendar years of actual out-of-sample coverage are required")
    return (pd.Timestamp(records[-1]['target_date'])-pd.Timestamp(records[0]['date'])).days/365.2425


def regime_labels(frame):
    volatility = frame.volatility_20
    threshold = volatility.rolling(756, min_periods=252).quantile(.75).shift(1)
    known = frame.sma200.notna() & volatility.notna() & threshold.notna()
    return pd.DataFrame({"known": known, "bull": frame.close >= frame.sma200,
                         "high_volatility": volatility >= threshold,
                         "realized_volatility": volatility, "volatility_threshold": threshold}, index=frame.index)


def bootstrap_indices(n, repetitions=BOOTSTRAP_REPETITIONS, block=5):
    rng = np.random.default_rng(42)
    starts = rng.integers(0, n, size=(repetitions, int(np.ceil(n/block))))
    return ((starts[:, :, None]+np.arange(block)) % n).reshape(repetitions, -1)[:, :n]


def paired_improvement(base_loss, loss, repetitions=BOOTSTRAP_REPETITIONS, block=5, mask=None, indices=None):
    difference = np.asarray(base_loss)-np.asarray(loss)
    mask = np.ones(len(difference), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    selected = difference[mask]
    if not len(selected):
        return {k: None for k in ('mae_difference', 'relative_mae_improvement', 'ci_low', 'ci_high', 'p_one_sided', 'first_half_difference', 'second_half_difference')}
    observed = float(selected.mean())
    denominator = np.asarray(base_loss)[mask].mean()
    out = {"mae_difference": observed, "relative_mae_improvement": float(observed/denominator) if denominator else None,
           "ci_low": None, "ci_high": None, "p_one_sided": None,
           "first_half_difference": float(selected[:max(1, len(selected)//2)].mean()),
           "second_half_difference": float(selected[len(selected)//2:].mean())}
    if len(selected) < MIN_SLICE_N:
        return out
    indices = bootstrap_indices(len(difference), repetitions, block) if indices is None else indices
    sampled_mask = mask[indices]
    counts = sampled_mask.sum(axis=1)
    valid = counts > 0
    means = ((difference*mask)[indices].sum(axis=1)[valid]/counts[valid])
    if len(means) < .95*len(indices):
        return out
    out['ci_low'], out['ci_high'] = map(float, np.quantile(means, [.025, .975]))
    # Center the paired bootstrap distribution under H0: expected loss improvement = 0.
    out['p_one_sided'] = float((1+np.sum(means-observed >= observed))/(len(means)+1))
    return out


def summarize_slice(actual_log, predictions, mean_log, mask, indices):
    mask = np.asarray(mask, dtype=bool)
    actual = np.expm1(actual_log)
    baseline = abs(np.expm1(predictions['technical'])-actual)
    out = {"n": int(mask.sum()), "sufficient": bool(mask.sum() >= MIN_SLICE_N), "variants": {},
           "zero_mae": float(abs(actual[mask]).mean()) if mask.any() else None,
           "mean_mae": float(abs(np.expm1(np.asarray(mean_log)[mask])-actual[mask]).mean()) if mask.any() else None,
           "always_up_accuracy": float((actual[mask]>0).mean()) if mask.any() else None}
    for key, pred_log in predictions.items():
        pred = np.expm1(pred_log)
        error = pred-actual
        out['variants'][key] = {
            "metrics": {"mae": float(abs(error[mask]).mean()) if mask.any() else None,
                        "rmse": float(np.sqrt(np.mean(error[mask]**2))) if mask.any() else None,
                        "direction_accuracy": float(np.mean((pred[mask]>0)==(actual[mask]>0))) if mask.any() else None},
            "improvement": paired_improvement(baseline, abs(error), mask=mask, indices=indices)}
    return out


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p, kind='stable')
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]))
    return adjusted.tolist()


def stability_checks(comparison, key, adjusted_p):
    variant = comparison['variants'][key]
    required = [comparison['slices'][s] for s in ('bull', 'bear', 'high_volatility', 'normal_volatility', 'early', 'recent')]
    years = [s for s in comparison['annual'].values() if s['complete'] and s['n'] >= 8]
    wins = sum(s['variants'][key]['improvement']['mae_difference'] > 0 for s in years)
    gain = variant['improvement']['relative_mae_improvement']
    ci = variant['improvement']['ci_low']
    checks = {
        "overall_gain_at_least_1pct": gain is not None and gain >= .01,
        "overall_ci_above_zero": ci is not None and ci > 0,
        "holm_evidence": adjusted_p < .05,
        "all_slices_sufficient": all(s['sufficient'] for s in required),
        "all_slices_improve": all(s['variants'][key]['improvement']['mae_difference'] is not None and s['variants'][key]['improvement']['mae_difference'] > 0 for s in required),
        "five_complete_years": len(years) >= 5,
        "two_thirds_years_improve": len(years) > 0 and wins/len(years) >= 2/3,
    }
    return {"passes": all(checks.values()), "checks": checks, "holm_p": adjusted_p,
            "positive_years": wins, "eligible_years": len(years)}


def add_stability_assessment(markets):
    from .config import PRIMARY
    for codes, family_name in [(PRIMARY, 'primary_42'), ([c for c in markets if c not in PRIMARY], 'exploratory_nasdaq_14')]:
        tests = [(code, c, key) for code in codes if code in markets for c in markets[code].get('macro_comparisons', []) for key in c['variants'] if key != 'technical']
        adjusted = holm_adjust([c['variants'][key]['improvement']['p_one_sided'] if c['variants'][key]['improvement']['p_one_sided'] is not None else 1. for _, c, key in tests])
        for (_, comparison, key), p in zip(tests, adjusted):
            comparison['variants'][key]['stability'] = {**stability_checks(comparison, key, p), 'family': family_name, 'family_size': len(tests)}
    return {str(h): [key for key in next(iter(markets.values()))['macro_comparisons'][0]['variants'] if key != 'technical' and
            all(next(c for c in markets[code]['macro_comparisons'] if c['horizon'] == h)['variants'][key]['stability']['passes'] for code in PRIMARY)] for h in (5, 20)}
