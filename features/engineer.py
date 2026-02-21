"""
Feature engineering for regime classifier.
~10–12 lean features: ATR, ATR ratio, EMA slopes, RSI, volume ratio,
range position, VWAP crosses, opening range breakout.
"""

import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    ATR_PERIOD, RSI_PERIOD, EMA_FAST, EMA_SLOW, VOLUME_MA_PERIOD,
    OPENING_RANGE_MINUTES, BAR_INTERVAL,
)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    typical = (high + low + close) / 3
    return (typical * volume).cumsum() / volume.cumsum().replace(0, np.nan)


def bars_in_minutes(minutes: int) -> int:
    """Number of bars in given minutes for current BAR_INTERVAL."""
    m = int(BAR_INTERVAL.replace("m", "")) if "m" in BAR_INTERVAL else 60
    return max(1, minutes // m)


def compute_features(ohlcv: pd.DataFrame) -> pd.Series:
    """
    Compute one row of features from OHLCV (full window).
    Assumes ohlcv has columns: open, high, low, close, volume and DatetimeIndex.
    Returns a Series of feature values for the latest bar (or empty if insufficient data).
    """
    cols = [c.lower() for c in ohlcv.columns]
    ohlcv = ohlcv.copy()
    ohlcv.columns = cols
    if "close" not in ohlcv.columns:
        return pd.Series(dtype=float)

    need = max(ATR_PERIOD, RSI_PERIOD, EMA_SLOW, VOLUME_MA_PERIOD) + 20
    if len(ohlcv) < need:
        return pd.Series(dtype=float)

    high = ohlcv["high"]
    low = ohlcv["low"]
    close = ohlcv["close"]
    volume = ohlcv["volume"] if "volume" in ohlcv.columns else pd.Series(1, index=ohlcv.index)

    # 1) ATR (14)
    atr = _atr(high, low, close, ATR_PERIOD)
    atr_curr = atr.iloc[-1]

    # 2) ATR ratio: today's ATR vs 5-day avg ATR
    ohlcv_d = ohlcv.resample("D").agg({"high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    if len(ohlcv_d) >= 2:
        atr_daily = _atr(ohlcv_d["high"], ohlcv_d["low"], ohlcv_d["close"], ATR_PERIOD)
        atr_5d_avg = atr_daily.rolling(5).mean().iloc[-1]
        atr_ratio = (atr_curr / atr_5d_avg) if atr_5d_avg and atr_5d_avg > 0 else 1.0
    else:
        atr_ratio = 1.0

    # 3) EMA slopes (8 and 21)
    ema8 = _ema(close, EMA_FAST)
    ema21 = _ema(close, EMA_SLOW)
    n_slope = 5
    ema8_slope = (ema8.iloc[-1] - ema8.iloc[-1 - n_slope]) / (close.iloc[-1] + 1e-10) * 100
    ema21_slope = (ema21.iloc[-1] - ema21.iloc[-1 - n_slope]) / (close.iloc[-1] + 1e-10) * 100

    # 4) RSI (14)
    rsi = _rsi(close, RSI_PERIOD)
    rsi_curr = rsi.iloc[-1]

    # 5) Volume ratio: current bar vol / 20-bar avg
    vol_ma = volume.rolling(VOLUME_MA_PERIOD).mean().iloc[-1]
    volume_ratio = (volume.iloc[-1] / vol_ma) if vol_ma and vol_ma > 0 else 1.0

    # 6) Range position: where is price within today's high-low?
    today = ohlcv.index[-1].date()
    today_bars = ohlcv[ohlcv.index.date == today]
    if len(today_bars) > 0:
        day_high = today_bars["high"].max()
        day_low = today_bars["low"].min()
        day_range = day_high - day_low
        if day_range > 0:
            range_position = (close.iloc[-1] - day_low) / day_range
        else:
            range_position = 0.5
    else:
        range_position = 0.5

    # 7) VWAP crosses today
    vwap = _vwap(high, low, close, volume)
    today_mask = ohlcv.index.date == today
    if today_mask.sum() > 1:
        price_today = close.loc[today_mask]
        vwap_today = vwap.loc[today_mask]
        cross = (price_today - vwap_today).diff()
        vwap_crosses = (np.diff(np.sign(cross.dropna())) != 0).sum()
        if np.isnan(vwap_crosses):
            vwap_crosses = 0
    else:
        vwap_crosses = 0

    # 8) Opening range (first 15 min) breakout status
    n_or_bars = bars_in_minutes(OPENING_RANGE_MINUTES)
    today_bars = ohlcv[ohlcv.index.date == today]
    if len(today_bars) >= n_or_bars:
        or_high = today_bars["high"].iloc[:n_or_bars].max()
        or_low = today_bars["low"].iloc[:n_or_bars].min()
        last_close = close.iloc[-1]
        if last_close > or_high:
            or_breakout = 1.0   # broke above
        elif last_close < or_low:
            or_breakout = -1.0  # broke below
        else:
            or_breakout = 0.0   # inside
    else:
        or_breakout = 0.0

    # Optional: trend strength (close vs open of day)
    if len(today_bars) > 0:
        day_open = today_bars["open"].iloc[0]
        move_pct = (close.iloc[-1] - day_open) / (day_open + 1e-10) * 100
    else:
        move_pct = 0.0

    return pd.Series({
        "atr": atr_curr,
        "atr_ratio": atr_ratio,
        "ema8_slope": ema8_slope,
        "ema21_slope": ema21_slope,
        "rsi": rsi_curr,
        "volume_ratio": volume_ratio,
        "range_position": range_position,
        "vwap_crosses": float(vwap_crosses),
        "or_breakout": or_breakout,
        "move_pct_from_open": move_pct,
    })


def compute_features_for_day(ohlcv: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    """
    Compute feature row using only data up to (and including) the given date.
    Used for building training set: one row per day.
    """
    cutoff = date + pd.Timedelta(days=1)
    df = ohlcv[ohlcv.index < cutoff].copy()
    if len(df) < max(ATR_PERIOD, RSI_PERIOD, EMA_SLOW) + 10:
        return pd.Series(dtype=float)
    return compute_features(df)


FEATURE_NAMES = [
    "atr", "atr_ratio", "ema8_slope", "ema21_slope", "rsi", "volume_ratio",
    "range_position", "vwap_crosses", "or_breakout", "move_pct_from_open",
]
