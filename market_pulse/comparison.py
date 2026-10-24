"""Fixed, matched-sample macro ablations; no selection on evaluation outcomes."""
import hashlib
import numpy as np

from .config import FEATURES
from .macro import GROUPS
from .models import pipeline

VARIANTS = {"technical": "原技術模型（共同樣本）", "rates": "技術＋利率", "vix": "技術＋VIX", "fx": "技術＋匯率", "all": "技術＋全部"}


def paired_improvement(base_loss, loss, repetitions=2000, block=5):
    """Circular moving-block bootstrap of paired absolute loss differences."""
    difference = np.asarray(base_loss) - np.asarray(loss)
    n = len(difference)
    rng = np.random.default_rng(42)
    starts = rng.integers(0, n, size=(repetitions, int(np.ceil(n/block))))
    positions = ((starts[:, :, None] + np.arange(block)) % n).reshape(repetitions, -1)[:, :n]
    lo, hi = np.quantile(difference[positions].mean(axis=1), [.025, .975])
    return {"mae_difference": float(difference.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "relative_mae_improvement": float(difference.mean()/np.mean(base_loss)) if np.mean(base_loss) else None,
            "first_half_difference": float(difference[:n//2].mean()), "second_half_difference": float(difference[n//2:].mean())}


def compare_models(frame, horizon=5, min_train=504, test_window=504):
    all_features = FEATURES + sum(GROUPS.values(), [])
    common = np.isfinite(frame[all_features].to_numpy()).all(axis=1)
    y = np.log(frame.close.shift(-horizon)/frame.close)
    columns = {key: FEATURES + (sum(GROUPS.values(), []) if key == "all" else GROUPS.get(key, [])) for key in VARIANTS}
    # Preserve the original exchange date index; missing macro inputs never shorten a target horizon.
    start = max(min_train + horizon, len(frame) - test_window - horizon)
    common_records, predictions, actual, means = [], {key: [] for key in VARIANTS}, [], []
    for origin in range(start, len(frame)-horizon, horizon):
        train = np.flatnonzero(common[:origin-horizon] & np.isfinite(y.iloc[:origin-horizon]))
        if not common[origin] or len(train) < min_train:
            continue
        row = {"date": str(frame.index[origin].date()), "target_date": str(frame.index[origin+horizon].date()),
               "last_train_label_date": str(frame.index[train[-1]+horizon].date()), "train_n": len(train),
               "training_dates_sha256": hashlib.sha256(frame.index[train].strftime("%Y-%m-%d").str.cat(sep=",").encode()).hexdigest(),
               "actual_log": float(y.iloc[origin]), "mean_baseline_log": float(y.iloc[train].mean()), "predicted_log": {}}
        for key, cols in columns.items():
            model = pipeline().fit(frame.iloc[train][cols], y.iloc[train])
            prediction = float(model.predict(frame.iloc[[origin]][cols])[0])
            predictions[key].append(prediction)
            row["predicted_log"][key] = prediction
        common_records.append(row)
        actual.append(float(y.iloc[origin])); means.append(row["mean_baseline_log"])
    if len(common_records) < 20:
        raise ValueError("At least 20 common non-overlapping forecasts are required for macro comparison")
    actual_return = np.expm1(actual)
    base_loss = abs(np.expm1(predictions["technical"])-actual_return)
    last = len(frame)-1
    train = np.flatnonzero(common[:last-horizon] & np.isfinite(y.iloc[:last-horizon]))
    if not common[last] or len(train) < min_train:
        raise ValueError("Latest macro inputs are incomplete or stale; refusing to publish a partial comparison")
    variants = {}
    for key, cols in columns.items():
        predicted_return = np.expm1(predictions[key])
        loss = abs(predicted_return-actual_return)
        model = pipeline().fit(frame.iloc[train][cols], y.iloc[train])
        latest = float(model.predict(frame.iloc[[last]][cols])[0])
        variants[key] = {"name": VARIANTS[key], "features": cols, "feature_count": len(cols),
                         "metrics": {"mae": float(loss.mean()), "rmse": float(np.sqrt(np.mean((predicted_return-actual_return)**2))),
                                     "direction_accuracy": float(np.mean((predicted_return>0)==(actual_return>0)))},
                         "improvement": paired_improvement(base_loss, loss),
                         "forecast": {"origin": str(frame.index[last].date()), "return": float(np.expm1(latest)),
                                      "train_n": len(train), "model_components": int(model.named_steps["pca"].n_components_),
                                      "last_train_label_date": str(frame.index[train[-1]+horizon].date())}}
    return {"horizon": horizon, "n": len(common_records), "test_start": common_records[0]["date"],
            "test_end": common_records[-1]["target_date"], "variants": variants, "records": common_records,
            "zero_mae": float(np.mean(abs(actual_return))), "mean_mae": float(np.mean(abs(np.expm1(means)-actual_return))),
            "always_up_accuracy": float(np.mean(actual_return>0)), "common_feature_rows": int(common.sum()),
            "bootstrap": {"method": "paired circular moving blocks", "block_forecasts": 5, "repetitions": 2000, "seed": 42,
                          "confidence": .95, "multiple_comparisons_adjusted": False},
            "selection": "No automatic model selection; original technical forecast remains the default"}
