"""
NQ Momentum Regime Detector — Streamlit dashboard.
Live regime classification, confidence, playbooks, chart, and history.
"""

import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

# Project root
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
import config as cfg
from data.fetch_data import fetch_live_bars, fetch_historical_bars
from features.engineer import compute_features, _ema, _atr, _vwap, bars_in_minutes
from model.label import label_historical_days, build_training_dataset

st.set_page_config(page_title="NQ Regime Detector", layout="wide", initial_sidebar_state="expanded")

# --- Load model ---
@st.cache_resource
def load_model():
    if not os.path.exists(cfg.MODEL_PATH):
        return None
    with open(cfg.MODEL_PATH, "rb") as f:
        return pickle.load(f)


def run_inference(ohlcv: pd.DataFrame):
    """Compute features and return regime + confidence."""
    payload = load_model()
    if payload is None:
        return None, None, None
    feats = compute_features(ohlcv)
    if feats.empty or feats.isna().all():
        return None, None, None
    fnames = payload["feature_names"]
    X = np.array([[feats.get(k, np.nan) for k in fnames]])
    X = np.nan_to_num(X, nan=0.0)
    X_scaled = payload["scaler"].transform(X)
    pred = payload["model"].predict(X_scaled)[0]
    proba = payload["model"].predict_proba(X_scaled)[0]
    classes = list(payload["model"].classes_)
    conf = float(proba[classes.index(pred)])
    return pred, conf, feats


def chart_data(ohlcv: pd.DataFrame):
    """Add VWAP, EMAs, opening range for display."""
    if ohlcv.empty or len(ohlcv) < 20:
        return ohlcv, None, None, None, None, None
    ohlcv = ohlcv.copy()
    ohlcv.columns = [c.lower() for c in ohlcv.columns]
    high, low, close = ohlcv["high"], ohlcv["low"], ohlcv["close"]
    vol = ohlcv["volume"] if "volume" in ohlcv.columns else pd.Series(1, index=ohlcv.index)
    ohlcv["ema8"] = _ema(close, 8)
    ohlcv["ema21"] = _ema(close, 21)
    ohlcv["vwap"] = _vwap(high, low, close, vol)
    today = ohlcv.index[-1].date() if hasattr(ohlcv.index[-1], "date") else ohlcv.index[-1]
    today_bars = ohlcv[ohlcv.index.date == today]
    n_or = bars_in_minutes(cfg.OPENING_RANGE_MINUTES)
    or_high = or_low = None
    if len(today_bars) >= n_or:
        or_high = today_bars["high"].iloc[:n_or].max()
        or_low = today_bars["low"].iloc[:n_or].min()
    return ohlcv, or_high, or_low, today, today_bars, n_or


# --- UI ---
st.title("NQ Momentum Regime Detector")
st.caption(f"Ticker: {cfg.TICKER} · Bar: {cfg.BAR_INTERVAL} · Context engine (not buy/sell signals)")

sidebar = st.sidebar
sidebar.header("Settings")
auto_refresh = sidebar.checkbox("Auto-refresh every 5 min", value=False)
if auto_refresh:
    sidebar.caption("Dashboard will refresh periodically during the session.")

# Fetch live data
with st.spinner("Loading data..."):
    ohlcv = fetch_live_bars()
if ohlcv.empty:
    st.warning("No live data. Check ticker and market hours (or use cached historical).")
    st.stop()

# Inference
regime, confidence, feat_series = run_inference(ohlcv)
if regime is None and load_model() is None:
    st.error("Model not found. Run: `python model/train.py` first.")
elif regime is None:
    st.warning("Could not compute features (need more bars).")
else:
    # Big regime badge
    emoji = cfg.REGIME_EMOJI.get(regime, "⚪")
    name = cfg.REGIMES.get(regime, regime)
    color = cfg.REGIME_COLORS.get(regime, "#6b7280")
    st.markdown(
        f'<div style="text-align:center; padding:1.5rem; border-radius:12px; background:{color}22; border:2px solid {color};">'
        f'<span style="font-size:3rem;">{emoji}</span><br>'
        f'<span style="font-size:1.8rem; font-weight:700;">{name}</span></div>',
        unsafe_allow_html=True,
    )
    # Confidence meter
    st.progress(float(confidence), text=f"Confidence: {confidence*100:.0f}%")
    # Playbook for current regime
    playbook = cfg.PLAYBOOKS.get(regime, "")
    st.info(f"**Playbook:** {playbook}")

    # Mini playbook cards for all regimes
    st.caption("Playbooks by regime")
    cols = st.columns(4)
    for i, (r, name) in enumerate(cfg.REGIMES.items()):
        with cols[i]:
            color = cfg.REGIME_COLORS.get(r, "#6b7280")
            is_current = r == regime
            st.markdown(
                f'<div style="padding:0.6rem; border-radius:8px; border:2px solid {color}; '
                f'background:{"#f0fdf4" if is_current else "#fafafa"};">'
                f'<strong>{cfg.REGIME_EMOJI.get(r, "")} {name}</strong><br>'
                f'<small>{cfg.PLAYBOOKS.get(r, "")}</small></div>',
                unsafe_allow_html=True,
            )

# Chart: price + VWAP, EMAs, opening range
st.subheader("Live chart")
ohlcv_plot, or_high, or_low, today, today_bars, n_or = chart_data(ohlcv)
if ohlcv_plot is not None and not ohlcv_plot.empty:
    plot_df = ohlcv_plot[["close", "ema8", "ema21", "vwap"]].copy()
    plot_df.columns = ["Close", "EMA 8", "EMA 21", "VWAP"]
    st.line_chart(plot_df, height=400)
    if or_high is not None and or_low is not None and today_bars is not None:
        st.caption(f"Opening range (first {cfg.OPENING_RANGE_MINUTES} min): low={or_low:.2f}, high={or_high:.2f}")

# Regime history (last 10 days)
st.subheader("Regime history (last 10 days)")
try:
    hist_bars = fetch_historical_bars(months=1)
    if not hist_bars.empty:
        labeled = label_historical_days(hist_bars)
        if not labeled.empty:
            hist_display = labeled[["open", "close", "label"]].tail(10).copy()
            hist_display["move_%"] = ((hist_display["close"] - hist_display["open"]) / hist_display["open"] * 100).round(2)
            hist_display.index = [str(d)[:10] for d in hist_display.index]
            st.dataframe(hist_display, width="stretch")
        else:
            st.caption("No labeled history available.")
    else:
        st.caption("No historical data for history table.")
except Exception as e:
    st.caption(f"Could not load history: {e}")

# Optional: show raw features
with st.expander("Current feature values"):
    if feat_series is not None and not feat_series.empty:
        st.json(feat_series.to_dict())
    else:
        st.write("Not available.")

# Footer
st.divider()
st.caption("NQ Regime Detector — Retrain with `python model/train.py`. Not financial advice.")

# Auto-refresh
if auto_refresh:
    import time
    time.sleep(300)
    st.rerun()
