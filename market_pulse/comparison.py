"""Fixed full-factorial macro ablations on common, chronological samples."""
import hashlib
import numpy as np
import pandas as pd

from .config import FEATURES
from .macro import GROUPS
from .models import pipeline
from .evaluation import (BOOTSTRAP_REPETITIONS, evaluation_start, validate_span, regime_labels,
                         bootstrap_indices, paired_improvement, summarize_slice)

VARIANTS = {"technical": "原技術模型（共同樣本）", "rates": "技術＋利率", "vix": "技術＋VIX", "fx": "技術＋匯率",
            "rates_vix": "技術＋利率＋VIX", "rates_fx": "技術＋利率＋匯率", "vix_fx": "技術＋VIX＋匯率", "all": "技術＋全部"}
VARIANT_GROUPS = {"technical": [], "rates": ['rates'], "vix": ['vix'], "fx": ['fx'],
                  "rates_vix": ['rates', 'vix'], "rates_fx": ['rates', 'fx'], "vix_fx": ['vix', 'fx'], "all": list(GROUPS)}


def compare_models(frame, horizon=5, min_train=504, test_window=504, backtest_start=None, minimum_years=0):
    all_features = FEATURES + sum(GROUPS.values(), [])
    common = np.isfinite(frame[all_features].to_numpy()).all(axis=1)
    y = np.log(frame.close.shift(-horizon)/frame.close)
    regimes = regime_labels(frame)
    columns = {key: FEATURES + sum((GROUPS[g] for g in groups), []) for key, groups in VARIANT_GROUPS.items()}
    start = evaluation_start(frame.index, horizon, min_train, test_window, backtest_start)
    common_records, predictions, actual, means = [], {key: [] for key in VARIANTS}, [], []
    for origin in range(start, len(frame)-horizon, horizon):
        train = np.flatnonzero(common[:origin-horizon] & np.isfinite(y.iloc[:origin-horizon])) if origin >= horizon else np.array([], dtype=int)
        if not common[origin] or len(train) < min_train or not regimes.known.iloc[origin]:
            continue
        state = regimes.iloc[origin]
        row = {"date": str(frame.index[origin].date()), "target_date": str(frame.index[origin+horizon].date()),
               "last_train_label_date": str(frame.index[train[-1]+horizon].date()), "train_n": len(train),
               "training_dates_sha256": hashlib.sha256(frame.index[train].strftime("%Y-%m-%d").str.cat(sep=",").encode()).hexdigest(),
               "actual_log": float(y.iloc[origin]), "mean_baseline_log": float(y.iloc[train].mean()), "predicted_log": {},
               "trend_regime": 'bull' if state.bull else 'bear', "high_volatility": bool(state.high_volatility),
               "realized_volatility": float(state.realized_volatility), "volatility_threshold": float(state.volatility_threshold)}
        for key, cols in columns.items():
            model = pipeline().fit(frame.iloc[train][cols], y.iloc[train])
            prediction = float(model.predict(frame.iloc[[origin]][cols])[0])
            predictions[key].append(prediction)
            row["predicted_log"][key] = prediction
        common_records.append(row)
        actual.append(float(y.iloc[origin])); means.append(row["mean_baseline_log"])
    if len(common_records) < 20:
        raise ValueError("At least 20 common non-overlapping forecasts are required for macro comparison")
    years = validate_span(common_records, minimum_years)
    indices = bootstrap_indices(len(common_records))
    overall = summarize_slice(actual, predictions, means, np.ones(len(actual), dtype=bool), indices)
    dates = pd.DatetimeIndex([r['date'] for r in common_records])
    bull = np.array([r['trend_regime'] == 'bull' for r in common_records])
    high_vol = np.array([r['high_volatility'] for r in common_records])
    masks = {'bull': bull, 'bear': ~bull, 'high_volatility': high_vol, 'normal_volatility': ~high_vol,
             'early': dates < pd.Timestamp('2024-01-01'), 'recent': dates >= pd.Timestamp('2024-01-01')}
    slices = {key: summarize_slice(actual, predictions, means, mask, indices) for key, mask in masks.items()}
    annual = {}
    for year in sorted(set(dates.year)):
        mask = dates.year == year
        annual[str(year)] = {**summarize_slice(actual, predictions, means, mask, indices),
                            'complete': bool(year < frame.index[-1].year and (year > dates[0].year or dates[0] <= pd.Timestamp(year=year, month=1, day=10)))}
    last = len(frame)-1
    train = np.flatnonzero(common[:last-horizon] & np.isfinite(y.iloc[:last-horizon]))
    if not common[last] or len(train) < min_train:
        raise ValueError("Latest macro inputs are incomplete or stale; refusing to publish a partial comparison")
    variants = {}
    for key, cols in columns.items():
        model = pipeline().fit(frame.iloc[train][cols], y.iloc[train])
        latest = float(model.predict(frame.iloc[[last]][cols])[0])
        variants[key] = {"name": VARIANTS[key], "features": cols, "feature_count": len(cols),
                         **overall['variants'][key],
                         "forecast": {"origin": str(frame.index[last].date()), "return": float(np.expm1(latest)),
                                      "train_n": len(train), "model_components": int(model.named_steps["pca"].n_components_),
                                      "last_train_label_date": str(frame.index[train[-1]+horizon].date())}}
    return {"horizon": horizon, "n": len(common_records), "test_start": common_records[0]["date"],
            "test_end": common_records[-1]["target_date"], "requested_start": backtest_start, "coverage_years": years,
            "variants": variants, "records": common_records, "slices": slices, "annual": annual,
            "zero_mae": overall['zero_mae'], "mean_mae": overall['mean_mae'], "always_up_accuracy": overall['always_up_accuracy'],
            "common_feature_rows": int(common.sum()),
            "bootstrap": {"method": "paired circular moving blocks with jointly resampled masks", "block_forecasts": 5,
                          "repetitions": BOOTSTRAP_REPETITIONS, "seed": 42, "confidence": .95,
                          "multiple_comparisons_adjusted": "Overall primary p-values adjusted by Holm; slice confidence intervals unadjusted"},
            "selection": "No automatic model selection; original technical forecast remains the default"}
