"""
Auto-label historical days into regimes using simple rules:
Trend Day, Choppy, Volatile, Quiet.
"""

import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TREND_DAY_MIN_MOVE_PCT,
    CHOPPY_MAX_TREND_PCT,
    VOLATILE_ATR_RATIO,
    QUIET_ATR_RATIO,
    VWAP_CROSS_THRESHOLD,
    ATR_PERIOD,
)
from features.engineer import _atr, _vwap, FEATURE_NAMES, compute_features_for_day


def _daily_ohlcv(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Resample to one row per trading day (OHLCV)."""
    c = {x: x for x in ohlcv.columns}
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    cols = [k for k in agg if k in ohlcv.columns]
    if not cols:
        return pd.DataFrame()
    return ohlcv[cols].resample("D").agg({k: agg[k] for k in cols}).dropna(how="all")


def _atr_ratio_per_day(ohlcv: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
    """Per-day ATR (14) and 5-day average ATR for ratio."""
    atr_d = _atr(daily["high"], daily["low"], daily["close"], ATR_PERIOD)
    atr_5d = atr_d.rolling(5).mean()
    ratio = atr_d / atr_5d.replace(0, np.nan)
    return ratio


def _vwap_crosses_per_day(ohlcv: pd.DataFrame) -> pd.Series:
    """Count VWAP crosses per day."""
    high = ohlcv["high"]
    low = ohlcv["low"]
    close = ohlcv["close"]
    vol = ohlcv["volume"] if "volume" in ohlcv.columns else pd.Series(1, index=ohlcv.index)
    vwap = _vwap(high, low, close, vol)
    diff = (close - vwap)
    # cross when sign changes
    sign_changes = (np.diff(np.sign(diff)) != 0).astype(int)
    by_date = pd.Series(sign_changes, index=ohlcv.index[1:]).groupby(ohlcv.index[1:].date).sum()
    return by_date


def label_day(row: pd.Series, vwap_crosses: int, atr_ratio: float) -> str:
    """
    Label a single day: trend_up, choppy, volatile, quiet.
    row: daily OHLC (open, high, low, close); move_pct = (close-open)/open*100.
    """
    open_p = row["open"]
    close_p = row["close"]
    move_pct = (close_p - open_p) / (open_p + 1e-10) * 100
    abs_move = abs(move_pct)

    # 1) Volatile: high ATR ratio
    if atr_ratio >= VOLATILE_ATR_RATIO:
        return "volatile"

    # 2) Quiet: low ATR ratio
    if atr_ratio <= QUIET_ATR_RATIO:
        return "quiet"

    # 3) Trend day: strong directional move
    if abs_move >= TREND_DAY_MIN_MOVE_PCT:
        return "trend_up"

    # 4) Choppy: small move and/or many VWAP crosses
    if abs_move <= CHOPPY_MAX_TREND_PCT or vwap_crosses >= VWAP_CROSS_THRESHOLD:
        return "choppy"

    # Default: if moderate move and not many crosses, call it trend
    if abs_move >= TREND_DAY_MIN_MOVE_PCT * 0.6:
        return "trend_up"
    return "choppy"


def label_historical_days(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Build a DataFrame with one row per trading day: date, label, and optional features.
    """
    ohlcv = ohlcv.copy()
    ohlcv.columns = [c.lower() for c in ohlcv.columns]
    daily = _daily_ohlcv(ohlcv)
    if daily.empty or len(daily) < 5:
        return pd.DataFrame()

    atr_ratio = _atr_ratio_per_day(ohlcv, daily)
    vwap_crosses = _vwap_crosses_per_day(ohlcv)

    labels = []
    for d in daily.index:
        date = d.date() if hasattr(d, "date") else d
        row = daily.loc[d]
        ar = atr_ratio.get(d, np.nan)
        if np.isnan(ar) or ar <= 0:
            ar = 1.0
        vc = vwap_crosses.get(date, 0)
        if pd.isna(vc):
            vc = 0
        lab = label_day(row, int(vc), float(ar))
        labels.append(lab)

    result = daily.copy()
    result["label"] = labels
    result["atr_ratio"] = daily.index.map(lambda d: atr_ratio.get(d, np.nan))
    result["vwap_crosses"] = daily.index.map(lambda d: vwap_crosses.get(d.date() if hasattr(d, "date") else d, 0))
    return result


def build_training_dataset(ohlcv: pd.DataFrame, labeled_daily: pd.DataFrame) -> pd.DataFrame:
    """
    For each labeled day, compute features at end-of-day (or last bar) and attach label.
    Returns DataFrame with columns = FEATURE_NAMES + ['label'].
    """
    rows = []
    for d in labeled_daily.index:
        feats = compute_features_for_day(ohlcv, pd.Timestamp(d))
        if feats.empty or feats.isna().all():
            continue
        lab = labeled_daily.loc[d, "label"]
        feats["label"] = lab
        rows.append(feats)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out[FEATURE_NAMES + ["label"]]


if __name__ == "__main__":
    from data.fetch_data import fetch_historical_bars
    ohlcv = fetch_historical_bars(months=3)
    if ohlcv.empty:
        print("No data")
    else:
        labeled = label_historical_days(ohlcv)
        print(labeled[["open", "close", "label", "atr_ratio", "vwap_crosses"]].tail(15))
        train_df = build_training_dataset(ohlcv, labeled)
        print("\nTraining sample:", train_df.shape)
        print(train_df["label"].value_counts())
