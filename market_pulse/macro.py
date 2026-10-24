"""Auditable daily macro snapshots and conservative, backward-only time alignment."""
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from pathlib import Path
import hashlib
import json
import time
from urllib.parse import quote, urlencode

import numpy as np
import pandas as pd
import requests

from .config import MARKETS

SEED_COMMIT = "def6a4d5a9a75e1a123476750a80b9aa5c8576da"
SEED_URL = f"https://raw.githubusercontent.com/Leo-CZ/fed-rates-dashboard/{SEED_COMMIT}/data/fred/rates_daily.csv"
VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
RATE_COLUMNS = ["DGS2", "DGS10", "DGS30", "T10YIE", "DFII10", "DFII30"]
GROUPS = {
    "rates": ["yield2", "yield10", "yield30", "spread10_2", "spread30_10", "breakeven10", "real10", "real30", "yield10_change5"],
    "vix": ["vix_level", "vix_change5", "vix_relative20"],
    "fx": ["usdcad_return5", "usdcad_return20", "usdtwd_return5", "usdtwd_return20"],
}
SOURCE_KEYS = ["rates", "vix", "usdcad", "usdtwd", "rates_seed"]
POLICY = {
    "availability": "Observation date + 2 calendar days, 00:00 UTC (assumption, not observed release timestamps)",
    "lag_calendar_days": 2,
    "maximum_observation_age_days": 10,
    "vintage": "Latest retrieved vintage; revisions are not point-in-time reconstructed. Not an ALFRED vintage backtest.",
    "alignment": "Backward as-of at each exchange local close; no backfill, interpolation, or future values",
}


def get_text(url, attempts=3, headers=None):
    for attempt in range(attempts):
        try:
            response = requests.get(url, headers=headers, timeout=(10, 20))
            response.raise_for_status()
            return response.text
        except requests.RequestException as exc:
            if attempt == attempts-1:
                raise RuntimeError(f"Macro download failed: {url}: {exc}") from exc
            time.sleep(2 ** attempt)


def download_fred(url):
    # Both URLs are the same authoritative series. Some FRED query/cache routes time out.
    try:
        return parse_rates(get_text(url, attempts=1)), url
    except RuntimeError:
        alternate = url.split("&coed=")[0]
        return parse_rates(get_text(alternate, attempts=2)), alternate


def parse_rates(text, seed=False):
    frame = pd.read_csv(StringIO(text), na_values=["."])
    frame = frame.rename(columns={"observation_date": "date", "BE10": "T10YIE"})
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.set_index("date")
    columns = RATE_COLUMNS if seed else [c for c in RATE_COLUMNS if c in frame]
    if not columns:
        raise ValueError("No expected FRED columns")
    return frame[columns].apply(pd.to_numeric, errors="raise")


def validate_macro(frame, key):
    expected = RATE_COLUMNS if key in ("rates", "rates_seed") else ["close"]
    if list(frame.columns) != expected or frame.empty or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"Invalid macro schema or dates: {key}")
    if frame.index.isna().any() or np.isinf(frame.to_numpy()).any():
        raise ValueError(f"Invalid macro values: {key}")
    if key not in ("rates", "rates_seed") and (frame.close.dropna() <= 0).any():
        raise ValueError(f"Nonpositive macro price: {key}")
    if len(frame.dropna()) < 800:
        raise ValueError(f"Insufficient macro history: {key}")
    return frame


def save_snapshot(directory, key, frame, metadata):
    frame = validate_macro(frame, key)
    frame.index.name = "date"
    text = frame.to_csv(float_format="%.12g")
    digest = hashlib.sha256(text.encode()).hexdigest()
    filename = f"macro-{key}-{digest[:16]}.csv"
    (Path(directory) / filename).write_text(text, encoding="utf-8")
    valid = frame.dropna()
    return {**metadata, "file": filename, "sha256": digest, "rows": len(frame),
            "first_date": str(valid.index[0].date()), "last_date": str(valid.index[-1].date())}


def parse_fx(payload, symbol, limit):
    results = payload.get("chart", {}).get("result")
    if not results or results[0].get("meta", {}).get("symbol") != symbol:
        raise ValueError(f"Unexpected FX response: {symbol}")
    item = results[0]
    dates = pd.to_datetime(item["timestamp"], unit="s", utc=True).normalize().tz_localize(None)
    # Only close is consumed: Yahoo FX OHLC ranges may not enclose the reported close.
    frame = pd.DataFrame({"close": item["indicators"]["quote"][0]["close"]}, index=dates)
    frame = frame[frame.index < limit]
    missing = int(frame.close.isna().sum())
    frame = frame.dropna().sort_index()
    frame.index.name = "date"
    frame.attrs["missing_close_rows_dropped"] = missing
    return frame


def refresh_macro(directory, start, now, as_of=None):
    print("Downloading upstream rates, FRED, Cboe VIX and USD exchange rates…", flush=True)
    limit = min(pd.Timestamp(now).tz_convert("UTC").normalize().tz_localize(None),
                pd.Timestamp(as_of) + pd.Timedelta(days=1)) if as_of else pd.Timestamp(now).tz_convert("UTC").normalize().tz_localize(None)
    seed_text = get_text(SEED_URL)
    seed = parse_rates(seed_text, seed=True)
    seed = seed[(seed.index >= start) & (seed.index < limit)]
    end = str((limit-pd.Timedelta(days=1)).date())
    urls = [f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}&coed={end}" for series in RATE_COLUMNS]
    with ThreadPoolExecutor(max_workers=6) as executor:
        downloaded = list(executor.map(download_fred, urls))
        fresh = pd.concat([frame for frame, _ in downloaded], axis=1)[RATE_COLUMNS]
        urls = [url for _, url in downloaded]
    fresh = fresh[(fresh.index >= start) & (fresh.index < limit)]
    overlap = fresh.index.intersection(seed.index)
    differences = int(((fresh.loc[overlap] - seed.loc[overlap]).abs() > 1e-8).sum().sum())
    # Preserve the upstream history; original FRED observations extend and revise it.
    rates = fresh.combine_first(seed).sort_index()[RATE_COLUMNS]
    entries = {
        "rates_seed": save_snapshot(directory, "rates_seed", seed, {"source_url": SEED_URL, "repository_commit": SEED_COMMIT, "original_sha256": hashlib.sha256(seed_text.encode()).hexdigest(), "units": "percent per year"}),
        "rates": save_snapshot(directory, "rates", rates, {"source_urls": urls, "seed_commit": SEED_COMMIT, "overlap_revisions": differences, "seed_only_values": int((rates.notna() & fresh.reindex(rates.index).isna()).sum().sum()), "units": "percent per year; differences in percentage points"}),
    }
    vix = pd.read_csv(StringIO(get_text(VIX_URL)))
    vix["date"] = pd.to_datetime(vix.DATE, format="%m/%d/%Y")
    vix = vix.set_index("date")[["CLOSE"]].rename(columns={"CLOSE": "close"})
    vix = vix[(vix.index >= start) & (vix.index < limit)]
    entries["vix"] = save_snapshot(directory, "vix", vix, {"source_url": VIX_URL, "units": "VIX index points (30-day implied volatility, annualized %)"})
    for key, symbol, currency in [("usdcad", "CAD=X", "CAD"), ("usdtwd", "TWD=X", "TWD")]:
        params = {"period1": int(pd.Timestamp(start, tz="UTC").timestamp()), "period2": int(limit.tz_localize("UTC").timestamp()), "interval": "1d"}
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol, safe='')}?{urlencode(params)}"
        frame = parse_fx(json.loads(get_text(url, headers={"User-Agent": "Mozilla/5.0"})), symbol, limit)
        entries[key] = save_snapshot(directory, key, frame, {"source_url": url, "symbol": symbol, "missing_close_rows_dropped": frame.attrs["missing_close_rows_dropped"], "units": f"{currency} per 1 USD; positive return = USD appreciation"})
    for key in SOURCE_KEYS[:-1]:
        age = (limit - pd.Timestamp(entries[key]["last_date"])).days
        if age > POLICY["maximum_observation_age_days"]:
            raise ValueError(f"Stale macro download: {key} ({age} days)")
    return {"policy": POLICY, "sources": entries}


def load_macro(directory, manifest):
    if "macro" not in manifest:
        return None  # Old snapshots retain their original technical-only report.
    entries = manifest["macro"]["sources"]
    if set(entries) != set(SOURCE_KEYS):
        raise ValueError("Incomplete macro snapshot")
    frames = {}
    for key, meta in entries.items():
        path = Path(directory) / meta["file"]
        if path.resolve().parent != Path(directory).resolve() or hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
            raise ValueError(f"Macro checksum/path mismatch: {key}")
        frame = validate_macro(pd.read_csv(path, index_col="date", parse_dates=True), key)
        if str(frame.dropna().index[-1].date()) != meta["last_date"]:
            raise ValueError(f"Macro date mismatch: {key}")
        frames[key] = frame
    return frames


def source_features(frames):
    r = frames["rates"].dropna()
    rates = pd.DataFrame({"yield2": r.DGS2, "yield10": r.DGS10, "yield30": r.DGS30,
        "spread10_2": r.DGS10-r.DGS2, "spread30_10": r.DGS30-r.DGS10,
        "breakeven10": r.T10YIE, "real10": r.DFII10, "real30": r.DFII30,
        "yield10_change5": r.DGS10.diff(5)})
    v = frames["vix"].close.dropna()
    out = {"rates": rates, "vix": pd.DataFrame({"vix_level": v, "vix_change5": v.diff(5), "vix_relative20": v/v.rolling(20).mean()-1})}
    for key in ["usdcad", "usdtwd"]:
        p = frames[key].close.dropna()
        out[key] = pd.DataFrame({f"{key}_return5": np.log(p/p.shift(5)), f"{key}_return20": np.log(p/p.shift(20))})
    return out


def align_macro(index, code, frames):
    spec = MARKETS[code]
    hour, minute = map(int, spec["close"].split(":"))
    # Localize each session independently so North American DST is respected.
    origins = (index + pd.Timedelta(hours=hour, minutes=minute)).tz_localize(spec["timezone"]).tz_convert("UTC")
    features, audit = [], {}
    for key, source in source_features(frames).items():
        source = source.copy()
        source["observation_date"] = source.index.tz_localize("UTC")
        source["available_at"] = source.observation_date + pd.Timedelta(days=POLICY["lag_calendar_days"])
        joined = pd.merge_asof(pd.DataFrame({"origin_at": origins}), source.reset_index(drop=True),
                               left_on="origin_at", right_on="available_at", direction="backward")
        age = (joined.origin_at - joined.observation_date).dt.total_seconds()/86400
        eligible = age <= POLICY["maximum_observation_age_days"]
        cols = [c for c in source if c not in ("observation_date", "available_at")]
        aligned = joined[cols].where(eligible, np.nan, axis=0)
        aligned.index = index
        features.append(aligned)
        audit[key] = {"dates": index.strftime("%Y-%m-%d").tolist(),
                      "observation_dates": joined.observation_date.dt.strftime("%Y-%m-%d").where(joined.observation_date.notna(), None).tolist(),
                      "available_at_utc": joined.available_at.dt.strftime("%Y-%m-%dT%H:%M:%SZ").where(joined.available_at.notna(), None).tolist(),
                      "age_days": age.where(age.notna(), None).tolist()}
    return pd.concat(features, axis=1), audit


def macro_summary(frames, manifest):
    summaries = {}
    for key in SOURCE_KEYS[:-1]:
        frame = frames[key].dropna()
        latest = frame.iloc[-1]
        history = frame.tail(756)
        summaries[key] = {"date": str(frame.index[-1].date()), "values": latest.to_dict(),
                          "units": manifest["macro"]["sources"][key]["units"],
                          "history": {"dates": history.index.strftime("%Y-%m-%d").tolist(), **{c: history[c].tolist() for c in frame}}}
    return {"policy": POLICY, "series": summaries, "feature_groups": GROUPS}
