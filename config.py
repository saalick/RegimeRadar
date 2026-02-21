"""
NQ Regime Detector — configuration.
Tickers, timeframes, feature params, and regime thresholds.
"""

# --- Data ---
TICKER = "QQQ"  # NQ proxy; use "^NDX" or "NQ=F" for futures if available
BAR_INTERVAL = "5m"  # "1m" or "5m"
LOOKBACK_DAYS = 10  # rolling window for features
HISTORICAL_MONTHS = 12  # for training

# --- Feature params ---
ATR_PERIOD = 14
RSI_PERIOD = 14
EMA_FAST = 8
EMA_SLOW = 21
VOLUME_MA_PERIOD = 20
OPENING_RANGE_MINUTES = 15  # first 15 min = opening range

# --- Regime labels (used in model) ---
REGIMES = {
    "trend_up": "Trend Day",
    "choppy": "Mean-Reversion / Choppy",
    "volatile": "Volatile / News-Driven",
    "quiet": "Quiet / Low-Vol",
}

REGIME_COLORS = {
    "trend_up": "#22c55e",   # green
    "choppy": "#ef4444",     # red
    "volatile": "#eab308",   # yellow
    "quiet": "#3b82f6",      # blue
}

REGIME_EMOJI = {
    "trend_up": "🟢",
    "choppy": "🔴",
    "volatile": "🟡",
    "quiet": "🔵",
}

PLAYBOOKS = {
    "trend_up": "Favor breakout entries, trail stops, don't fade.",
    "choppy": "Fade extremes, target VWAP, tight stops.",
    "volatile": "Reduce size, widen stops, or sit out.",
    "quiet": "Watch for range break, small size until confirmation.",
}

# --- Labeling rules (for auto-labeling historical days) ---
TREND_DAY_MIN_MOVE_PCT = 1.5  # open-to-close move % for trend
CHOPPY_MAX_TREND_PCT = 0.5   # max directional move for chop
VOLATILE_ATR_RATIO = 1.5     # ATR vs 5d avg for "volatile"
QUIET_ATR_RATIO = 0.7        # ATR vs 5d avg for "quiet"
VWAP_CROSS_THRESHOLD = 5     # min crosses to consider choppy

# --- Paths ---
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "regime_model.pkl")
DATA_CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")

# --- Market hours (Eastern) for live inference ---
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0
INFERENCE_INTERVAL_MINUTES = 5
