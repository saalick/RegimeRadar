"""
Pull historical and live OHLCV bars for NQ proxy (QQQ).
Uses yfinance; supports 5m (and 1m with 7d limit for free tier).
"""

import os
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TICKER, BAR_INTERVAL, LOOKBACK_DAYS, HISTORICAL_MONTHS, DATA_CACHE_DIR


def _ensure_cache_dir():
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)


def fetch_historical_bars(months: int = None, interval: str = None) -> pd.DataFrame:
    """
    Fetch historical intraday data for training.
    yfinance limits: 5m = last 60 days only; 1m = last 7 days.
    """
    months = months or HISTORICAL_MONTHS
    interval = interval or BAR_INTERVAL
    end = datetime.now()
    # yfinance 5m only allows last 60 days; cap request
    max_days = 59 if interval == "5m" else (7 if interval == "1m" else 60)
    requested_days = min(months * 31, max_days)
    start = end - timedelta(days=requested_days)

    _ensure_cache_dir()
    cache_file = os.path.join(DATA_CACHE_DIR, f"{TICKER}_{interval}_hist.parquet")

    # Try cache first (optional: use fresh if older than 1 day)
    if os.path.exists(cache_file):
        try:
            df = pd.read_parquet(cache_file)
            if df.index.max() >= (end - timedelta(days=3)).timestamp():
                return df
        except Exception:
            pass

    # yfinance intraday: 1m max 7d, 5m max 60d in one go
    chunks = []
    chunk_days = 7 if interval == "1m" else min(59, (end - start).days)
    current_start = start
    while current_start < end:
        current_end = min(current_start + timedelta(days=chunk_days), end)
        t = yf.Ticker(TICKER)
        hist = t.history(start=current_start, end=current_end, interval=interval, auto_adjust=True)
        if hist is not None and not hist.empty:
            hist = hist.tz_localize(None)
            chunks.append(hist)
        current_start = current_end
        if not chunks and current_start >= end:
            break

    if not chunks:
        return pd.DataFrame()

    df = pd.concat(chunks).sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # Normalize columns to lowercase
    df.columns = [c.lower() for c in df.columns]
    if "open" not in df.columns and "Open" in df.columns:
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})

    try:
        df.to_parquet(cache_file)
    except Exception:
        pass
    return df


def fetch_live_bars(lookback_days: int = None) -> pd.DataFrame:
    """
    Fetch last N trading days + today for live feature computation.
    Returns OHLCV with DatetimeIndex (no tz).
    """
    lookback_days = lookback_days or LOOKBACK_DAYS
    end = datetime.now()
    start = end - timedelta(days=lookback_days + 2)  # buffer

    t = yf.Ticker(TICKER)
    # 5m gives more history in one request
    df = t.history(start=start, end=end, interval=BAR_INTERVAL, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.tz_localize(None)
    df.columns = [c.lower() for c in df.columns]
    return df.sort_index()


def get_training_data() -> pd.DataFrame:
    """Convenience: historical bars for model training."""
    return fetch_historical_bars(months=HISTORICAL_MONTHS, interval=BAR_INTERVAL)


def get_live_data() -> pd.DataFrame:
    """Convenience: recent bars for real-time inference."""
    return fetch_live_bars(lookback_days=LOOKBACK_DAYS)


if __name__ == "__main__":
    print("Fetching historical data...")
    hist = fetch_historical_bars(months=2)
    print(f"Historical: {len(hist)} bars, from {hist.index.min()} to {hist.index.max()}")

    print("\nFetching live window...")
    live = fetch_live_bars(lookback_days=5)
    print(f"Live: {len(live)} bars")
