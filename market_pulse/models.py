"""Chronological, purged expanding-window forecasts. No shuffled train/test split."""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .config import FEATURES, PRIMARY


def pipeline():
    return make_pipeline(StandardScaler(), PCA(n_components=.9, svd_solver="full"), Ridge(alpha=10.0))


def training_positions(origin, horizon, minimum=504):
    # Training label for row j ends at j+h; require j+h < origin.
    end = origin - horizon
    if end < minimum:
        raise ValueError("Not enough mature training labels")
    return np.arange(end)


def forecast(frame, horizon=5, min_train=504, test_window=504):
    complete = frame[FEATURES].notna().all(axis=1)
    first = np.flatnonzero(complete.to_numpy())
    if not len(first):
        raise ValueError("No complete technical features")
    # Do not silently compress internal gaps and change the meaning of a trading-day horizon.
    sample = frame.iloc[first[0]:].copy()
    if not sample[FEATURES].notna().all().all():
        raise ValueError("Missing technical features inside the modeling window")
    x = sample[FEATURES]
    y = np.log(sample.close.shift(-horizon) / sample.close)
    last = len(sample) - 1
    start = max(min_train + horizon, len(sample) - test_window - horizon)
    rows, errors, coverage = [], [], []
    for origin in range(start, len(sample) - horizon, horizon):
        train = training_positions(origin, horizon, min_train)
        model = pipeline().fit(x.iloc[train], y.iloc[train])
        pred = float(model.predict(x.iloc[[origin]])[0])
        actual = float(y.iloc[origin])
        if len(errors) >= 20:
            lo, hi = np.quantile(errors, [.1, .9])
            coverage.append(float(pred + lo <= actual <= pred + hi))
        rows.append({"date": str(x.index[origin].date()), "target_date": str(x.index[origin + horizon].date()), "last_train_label_date": str(x.index[train[-1] + horizon].date()), "predicted_log": pred, "actual_log": actual, "mean_baseline_log": float(y.iloc[train].mean())})
        errors.append(actual - pred)
    if len(rows) < 20:
        raise ValueError("At least 20 non-overlapping out-of-sample forecasts are required")
    records = pd.DataFrame(rows)
    pred_r, actual_r, mean_r = [np.expm1(records[c].to_numpy()) for c in ["predicted_log", "actual_log", "mean_baseline_log"]]
    metrics = {"n": len(rows), "mae": float(np.mean(abs(pred_r - actual_r))), "zero_mae": float(np.mean(abs(actual_r))), "mean_mae": float(np.mean(abs(mean_r - actual_r))), "rmse": float(np.sqrt(np.mean((pred_r - actual_r)**2))), "direction_accuracy": float(np.mean((pred_r > 0) == (actual_r > 0))), "always_up_accuracy": float(np.mean(actual_r > 0)), "prequential_coverage": float(np.mean(coverage)) if coverage else None, "coverage_n": len(coverage), "test_start": rows[0]["date"], "test_end": rows[-1]["target_date"]}
    train = training_positions(last, horizon, min_train)
    model = pipeline().fit(x.iloc[train], y.iloc[train])
    pred = float(model.predict(x.iloc[[last]])[0])
    lo, hi = np.quantile(errors, [.1, .9])
    pca = model.named_steps["pca"]
    return {"horizon": horizon, "origin": str(sample.index[-1].date()), "return": float(np.expm1(pred)), "low": float(np.expm1(pred + lo)), "high": float(np.expm1(pred + hi)), "price": float(sample.close.iloc[-1] * np.exp(pred)), "price_low": float(sample.close.iloc[-1] * np.exp(pred + lo)), "price_high": float(sample.close.iloc[-1] * np.exp(pred + hi)), "metrics": metrics, "model_components": int(pca.n_components_), "train_n": len(train), "last_train_label_date": str(x.index[train[-1] + horizon].date()), "records": rows, "beats_baselines": metrics["mae"] < min(metrics["zero_mae"], metrics["mean_mae"])}


def market_pca(frames, window=756):
    # Individual-market returns first, THEN intersect dates: holidays never become fabricated zeroes.
    daily = pd.concat({k: np.log(frames[k].close).diff() for k in PRIMARY}, axis=1).dropna().tail(window)
    if len(daily) < 60:
        raise ValueError("Insufficient overlapping dates for PCA")
    z = StandardScaler().fit_transform(daily)
    pca = PCA().fit(z)
    components = pca.components_.copy()
    # Deterministic orientation; eigenvector signs have no financial meaning.
    for i, component in enumerate(components):
        if component[np.argmax(abs(component))] < 0:
            components[i] *= -1
    scores = z @ components.T
    return {"markets": PRIMARY, "n": len(daily), "start": str(daily.index[0].date()), "end": str(daily.index[-1].date()), "explained_variance": pca.explained_variance_ratio_.tolist(), "weights": components.tolist(), "loadings": (components.T * np.sqrt(pca.explained_variance_ * (len(daily) - 1) / len(daily))).tolist(), "correlation": daily.corr().to_numpy().tolist(), "dates": daily.index.strftime("%Y-%m-%d").tolist(), "scores": scores.tolist()}
