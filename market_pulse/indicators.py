import numpy as np
import pandas as pd


def wilder(series, period=14):
    """Wilder smoothing initialized by the first complete period's arithmetic mean."""
    result = pd.Series(np.nan, index=series.index, dtype=float)
    valid = series.dropna()
    if len(valid) < period:
        return result
    value = float(valid.iloc[:period].mean())
    result.loc[valid.index[period - 1]] = value
    for date, current in valid.iloc[period:].items():
        value = ((period - 1) * value + current) / period
        result.loc[date] = value
    return result


def indicators(prices):
    x = prices.copy()
    close = x.close
    for period in (1, 5, 20):
        x[f"return_{period}"] = close.pct_change(period, fill_method=None)
    for period in (20, 60, 200):
        x[f"sma{period}"] = close.rolling(period).mean()
        x[f"sma{period}_gap"] = close / x[f"sma{period}"] - 1
    # Taiwan recursive KD(9,3,3); initialize prior K,D at 50, preserve them on a flat range.
    low, high = x.low.rolling(9).min(), x.high.rolling(9).max()
    rsv = 100 * (close - low) / (high - low).replace(0, np.nan)
    k, d = 50., 50.
    ks, ds = [], []
    for i, value in enumerate(rsv):
        if i < 8:
            ks.append(np.nan); ds.append(np.nan)
            continue
        if pd.notna(value):
            k = 2 * k / 3 + value / 3
            d = 2 * d / 3 + k / 3
        ks.append(k); ds.append(d)
    x["rsv"], x["k"], x["d"] = rsv, ks, ds
    x["j"] = 3 * x.k - 2 * x.d
    change = close.diff()
    gain, loss = wilder(change.clip(lower=0)), wilder(-change.clip(upper=0))
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    x["rsi"] = rsi.mask((loss == 0) & (gain > 0), 100).mask((loss == 0) & (gain == 0), 50)
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    x["macd"] = ema12 - ema26
    x["macd_signal"] = x.macd.ewm(span=9, adjust=False, min_periods=9).mean()
    x["macd_hist"] = x.macd - x.macd_signal
    x["macd_pct"], x["macd_hist_pct"] = x.macd / close, x.macd_hist / close
    std = close.rolling(20).std(ddof=0)
    x["bb_upper"], x["bb_lower"] = x.sma20 + 2 * std, x.sma20 - 2 * std
    x["bb_position"] = ((close - x.bb_lower) / (4 * std).replace(0, np.nan)).mask(std == 0, .5)
    x["volatility_20"] = np.log(close).diff().rolling(20).std(ddof=1) * np.sqrt(252)
    tr = pd.concat([x.high - x.low, (x.high - close.shift()).abs(), (x.low - close.shift()).abs()], axis=1).max(axis=1)
    x["atr"] = wilder(tr)
    x["atr_pct"] = x.atr / close
    x["drawdown"] = close / close.cummax() - 1
    # Index volume can be missing, zero, or provider-dependent; never used by the forecast.
    x["volume_ratio"] = x.volume / x.volume.rolling(20).mean().replace(0, np.nan)
    return x.replace([np.inf, -np.inf], np.nan)
