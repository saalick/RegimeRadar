
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

st.set_page_config(
    page_title="NQ Regime Detector",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Warm Earth global CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Main background */
  .stApp { background-color: #fdf8f0; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background-color: #f5ebe0 !important;
    border-right: 2px solid #e7d5c0;
  }

  /* Headers */
  h1, h2, h3, h4 { color: #2c1a0e !important; }

  /* Metric / info boxes */
  [data-testid="stMetric"] {
    background: #fef3c7;
    border-radius: 10px;
    padding: 0.5rem 1rem;
    border-left: 4px solid #b45309;
  }

  /* Progress bar (confidence) */
  .stProgress > div > div { background-color: #b45309 !important; }
  .stProgress { background-color: #e7d5c0 !important; border-radius: 8px; }

  /* Info/alert box */
  [data-testid="stAlert"] {
    background-color: #fef9c3 !important;
    border-left: 4px solid #b45309 !important;
    color: #2c1a0e !important;
  }

  /* Dataframe */
  [data-testid="stDataFrame"] { border: 1px solid #e7d5c0; border-radius: 8px; }

  /* Divider */
  hr { border-color: #e7d5c0 !important; }

  /* Expander */
  [data-testid="stExpander"] {
    background: #fef3c7;
    border: 1px solid #e7d5c0;
    border-radius: 8px;
  }

  /* Caption / small text */
  .stCaption, small { color: #6b5a47 !important; }

  /* Buttons */
  .stButton > button {
    background-color: #b45309;
    color: #fdf8f0;
    border: none;
    border-radius: 6px;
  }
  .stButton > button:hover { background-color: #92400e; }

  /* Checkbox */
  [data-testid="stCheckbox"] { accent-color: #b45309; }
</style>
""", unsafe_allow_html=True)

# ── Model loader ───────────────────────────────────────────────────────────────
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
    ohlcv["ema8"]  = _ema(close, 8)
    ohlcv["ema21"] = _ema(close, 21)
    ohlcv["vwap"]  = _vwap(high, low, close, vol)
    today = ohlcv.index[-1].date() if hasattr(ohlcv.index[-1], "date") else ohlcv.index[-1]
    today_bars = ohlcv[ohlcv.index.date == today]
    n_or = bars_in_minutes(cfg.OPENING_RANGE_MINUTES)
    or_high = or_low = None
    if len(today_bars) >= n_or:
        or_high = today_bars["high"].iloc[:n_or].max()
        or_low  = today_bars["low"].iloc[:n_or].min()
    return ohlcv, or_high, or_low, today, today_bars, n_or


# ── UI Header ──────────────────────────────────────────────────────────────────
st.markdown(
    '<h1 style="color:#2c1a0e; letter-spacing:-0.5px;">🌾 NQ Momentum Regime Detector</h1>',
    unsafe_allow_html=True,
)
st.caption(f"Ticker: {cfg.TICKER} · Bar: {cfg.BAR_INTERVAL} · Context engine — not buy/sell signals")

# Sidebar
sidebar = st.sidebar
sidebar.markdown(
    '<div style="font-size:1.1rem; font-weight:700; color:#2c1a0e; margin-bottom:0.5rem;">⚙️ Settings</div>',
    unsafe_allow_html=True,
)
auto_refresh = sidebar.checkbox("Auto-refresh every 5 min", value=False)
if auto_refresh:
    sidebar.caption("Dashboard will refresh periodically during the session.")

# ── Data fetch ─────────────────────────────────────────────────────────────────
with st.spinner("Loading data..."):
    ohlcv = fetch_live_bars()
if ohlcv.empty:
    st.warning("No live data. Check ticker and market hours (or use cached historical).")
    st.stop()

# ── Inference ──────────────────────────────────────────────────────────────────
regime, confidence, feat_series = run_inference(ohlcv)
if regime is None and load_model() is None:
    st.error("Model not found. Run: `python model/train.py` first.")
elif regime is None:
    st.warning("Could not compute features (need more bars).")
else:
    emoji  = cfg.REGIME_EMOJI.get(regime, "⚪")
    name   = cfg.REGIMES.get(regime, regime)
    color  = cfg.REGIME_COLORS.get(regime, "#78716c")
    bg     = cfg.REGIME_BG.get(regime, "#fef3c7")

    # ── Regime badge ──────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="
            text-align:center;
            padding:1.8rem 2rem;
            border-radius:16px;
            background:{bg};
            border:2.5px solid {color};
            box-shadow:0 4px 18px {color}30;
            margin-bottom:1rem;
        ">
            <span style="font-size:3.2rem;">{emoji}</span><br>
            <span style="font-size:2rem; font-weight:800; color:{color}; letter-spacing:-0.5px;">{name}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Confidence bar
    st.progress(
        float(confidence),
        text=f"Confidence: {confidence*100:.0f}%",
    )

    # Playbook callout
    playbook = cfg.PLAYBOOKS.get(regime, "")
    st.markdown(
        f"""
        <div style="
            background:#fef9c3;
            border-left:5px solid {color};
            border-radius:0 10px 10px 0;
            padding:0.8rem 1.2rem;
            margin:0.8rem 0;
            color:#2c1a0e;
        ">
            <strong>Playbook:</strong> {playbook}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Playbook cards (all regimes) ──────────────────────────────────────────
    st.markdown(
        '<p style="color:#6b5a47; font-size:0.82rem; margin-top:1.2rem;">Playbooks by regime</p>',
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    for i, (r, rname) in enumerate(cfg.REGIMES.items()):
        with cols[i]:
            c  = cfg.REGIME_COLORS.get(r, "#78716c")
            bg_card = cfg.REGIME_BG.get(r, "#fef3c7")
            is_current = r == regime
            border_width = "3px" if is_current else "1.5px"
            shadow = f"box-shadow:0 2px 10px {c}40;" if is_current else ""
            st.markdown(
                f"""
                <div style="
                    padding:0.7rem 0.8rem;
                    border-radius:10px;
                    border:{border_width} solid {c};
                    background:{bg_card if is_current else '#fdf8f0'};
                    {shadow}
                ">
                    <strong style="color:{c};">{cfg.REGIME_EMOJI.get(r,'')} {rname}</strong><br>
                    <small style="color:#6b5a47;">{cfg.PLAYBOOKS.get(r,'')}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ── Chart ──────────────────────────────────────────────────────────────────────
st.markdown("### 📈 Live Chart")
ohlcv_plot, or_high, or_low, today, today_bars, n_or = chart_data(ohlcv)
if ohlcv_plot is not None and not ohlcv_plot.empty:
    plot_df = ohlcv_plot[["close", "ema8", "ema21", "vwap"]].copy()
    plot_df.columns = ["Close", "EMA 8", "EMA 21", "VWAP"]
    st.line_chart(
        plot_df,
        height=400,
        color=["#2c1a0e", "#b45309", "#c2410c", "#78716c"],  # dark brown, amber, rust, stone
    )
    if or_high is not None and or_low is not None and today_bars is not None:
        st.caption(
            f"Opening range (first {cfg.OPENING_RANGE_MINUTES} min): "
            f"low = {or_low:.2f} · high = {or_high:.2f}"
        )

# ── Regime history ─────────────────────────────────────────────────────────────
st.markdown("### 🗓️ Regime History (last 10 days)")
try:
    hist_bars = fetch_historical_bars(months=1)
    if not hist_bars.empty:
        labeled = label_historical_days(hist_bars)
        if not labeled.empty:
            hist_display = labeled[["open", "close", "label"]].tail(10).copy()
            hist_display["move_%"] = (
                (hist_display["close"] - hist_display["open"])
                / hist_display["open"] * 100
            ).round(2)
            hist_display.index = [str(d)[:10] for d in hist_display.index]
            st.dataframe(hist_display, use_container_width=True)
        else:
            st.caption("No labeled history available.")
    else:
        st.caption("No historical data for history table.")
except Exception as e:
    st.caption(f"Could not load history: {e}")

# ── Feature inspector ──────────────────────────────────────────────────────────
with st.expander("🔍 Current feature values"):
    if feat_series is not None and not feat_series.empty:
        st.json(feat_series.to_dict())
    else:
        st.write("Not available.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "🌾 NQ Regime Detector · Retrain with `python model/train.py` · Not financial advice."
)

# Auto-refresh
if auto_refresh:
    import time
    time.sleep(300)
    st.rerun()
