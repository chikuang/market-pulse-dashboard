"""Public Yahoo daily data; no API key, no silent synthetic-data fallback."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo
import hashlib
import json
import time
import numpy as np
import pandas as pd
import requests
from .config import MARKETS


def validate_ohlcv(frame):
    required = ["open", "high", "low", "close", "volume"]
    if not set(required).issubset(frame):
        raise ValueError("OHLCV columns are required")
    if frame.empty or not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("Prices must be nonempty, unique and sorted by date")
    prices = frame[required[:4]]
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("OHLC prices must be finite and positive")
    if ((frame.high < prices.max(axis=1) - 1e-6) | (frame.low > prices.min(axis=1) + 1e-6)).any():
        raise ValueError("Inconsistent daily OHLC range")
    if (frame.volume.dropna() < 0).any():
        raise ValueError("Negative volume")
    if len(frame) < 800:
        raise ValueError("At least 800 daily bars are required for training and evaluation")
    return frame


def parse_chart(payload, spec, cutoff):
    result = payload.get("chart", {}).get("result")
    if not result:
        raise ValueError(f"Yahoo returned no data: {payload.get('chart', {}).get('error')}")
    item = result[0]
    actual_symbol = item.get("meta", {}).get("symbol")
    if actual_symbol != spec["symbol"]:
        raise ValueError(f"Wrong ticker in response: {actual_symbol}")
    quotes = item["indicators"]["quote"][0]
    dates = pd.to_datetime(item["timestamp"], unit="s", utc=True).tz_convert(spec["timezone"])
    frame = pd.DataFrame(quotes, index=dates.normalize().tz_localize(None))
    frame.index.name = "date"
    # Only past local calendar days: deliberately exclude today's possibly incomplete bar.
    local_today = cutoff.astimezone(ZoneInfo(spec["timezone"])).date()
    frame = frame[frame.index.date < local_today]
    missing_rows = int(frame[["open", "high", "low", "close"]].isna().any(axis=1).sum())
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    frame = frame[["open", "high", "low", "close", "volume"]].sort_index()
    validate_ohlcv(frame)
    frame.attrs["missing_ohlc_rows_dropped"] = missing_rows
    return frame


def market_cutoff(now, spec, as_of=None):
    """Inclusive historical --as-of in each exchange's own calendar."""
    if as_of is None:
        return now
    requested = datetime.fromisoformat(as_of).replace(tzinfo=ZoneInfo(spec["timezone"])) + timedelta(days=1)
    return min(now, requested.astimezone(timezone.utc))


def download_market(spec, start, cutoff):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(spec['symbol'], safe='')}"
    params = {"period1": int(pd.Timestamp(start, tz="UTC").timestamp()), "period2": int(cutoff.timestamp()), "interval": "1d", "events": "history"}
    last_error = None
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0 (MarketPulse research)"}, timeout=30)
            response.raise_for_status()
            frame = parse_chart(response.json(), spec, cutoff)
            return frame, response.url
        except (requests.RequestException, ValueError, KeyError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Download failed for {spec['symbol']}: {last_error}")


def refresh(directory, start="2016-01-01", as_of=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    frames, entries = {}, {}
    for code, spec in MARKETS.items():
        print(f"Downloading {spec['symbol']}…", flush=True)
        cutoff = market_cutoff(now, spec, as_of)
        frame, url = download_market(spec, start, cutoff)
        text = frame.to_csv(float_format="%.10g")
        digest = hashlib.sha256(text.encode()).hexdigest()
        # Content-addressed files keep the previous manifest usable after a partial refresh.
        filename = f"{code}-{digest[:16]}.csv"
        (directory / filename).write_text(text, encoding="utf-8")
        frames[code] = frame
        entries[code] = {**spec, "file": filename, "sha256": digest, "rows": len(frame), "first_date": str(frame.index[0].date()), "last_date": str(frame.index[-1].date()), "source_url": url, "cutoff_utc": cutoff.isoformat(), "missing_ohlc_rows_dropped": frame.attrs["missing_ohlc_rows_dropped"]}
    manifest = {"provider": "Yahoo Finance chart endpoint (unofficial)", "fetched_at_utc": now.isoformat(), "requested_as_of": as_of, "price_basis": "Unadjusted price indices; excludes dividends and FX conversion", "markets": entries}
    temp = directory / "manifest.tmp"
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(directory / "manifest.json")
    return load_data(directory)


def load_data(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    frames = {}
    for code in MARKETS:
        meta = manifest["markets"][code]
        path = directory / meta["file"]
        if path.resolve().parent != directory.resolve():
            raise ValueError("CSV must be inside the data directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
            raise ValueError(f"Checksum mismatch: {code}")
        frame = pd.read_csv(path, index_col="date", parse_dates=True)
        frames[code] = validate_ohlcv(frame)
    return frames, manifest
