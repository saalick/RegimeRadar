# NQ Momentum Regime Detector & Alert System

A lightweight, real-time dashboard that classifies the current NASDAQ (NQ) market into **3–4 regimes** (Trend Day, Choppy, Volatile, Quiet) using a simple ML model. It’s a **context engine**: it tells you how the market is behaving so you can pick the right playbook, not a buy/sell signal bot.

## Features

- **Data**: 5-minute (or 1-minute) OHLCV for QQQ (NQ proxy) via yfinance. Note: yfinance 5m data is limited to the last 60 days; training uses that window.
- **~10 features**: ATR, ATR ratio, EMA slopes, RSI, volume ratio, range position, VWAP crosses, opening range breakout
- **Classifier**: Random Forest on auto-labeled historical days
- **Streamlit UI**: Regime badge, confidence meter, playbook cards, live chart (VWAP, EMAs, opening range), regime history table

## Quick start

```bash
cd nq-dashboard
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

1. **Train the model** (uses last 12 months of data):

```bash
python model/train.py
```

2. **Run the dashboard**:

```bash
streamlit run app.py
```

Open the URL (e.g. http://localhost:8501). Enable “Auto-refresh every 5 min” in the sidebar for periodic updates.

## Project layout

```
nq-dashboard/
├── data/
│   └── fetch_data.py       # Historical + live bars (yfinance)
├── features/
│   └── engineer.py         # ATR, RSI, EMAs, VWAP, opening range, etc.
├── model/
│   ├── label.py            # Auto-label days (trend/choppy/volatile/quiet)
│   ├── train.py            # Train Random Forest, save regime_model.pkl
│   └── regime_model.pkl    # Saved model (after first train)
├── app.py                  # Streamlit dashboard
├── config.py               # Ticker, intervals, thresholds, playbooks
├── requirements.txt
└── README.md
```

## Regimes

| Regime | Description | Playbook |
|--------|-------------|----------|
| Trend Day | Directional move ~1.5%+ open→close, few pullbacks | Favor breakouts, trail stops, don’t fade |
| Choppy / Mean-reversion | Range-bound, oscillates around VWAP | Fade extremes, target VWAP, tight stops |
| Volatile / News-driven | High ATR, erratic moves | Reduce size, widen stops, or sit out |
| Quiet / Low-vol | Compressed range, low volume | Watch for range break, small size until confirmation |

## Configuration

Edit `config.py` to change:

- `TICKER`: e.g. `"QQQ"`, `"NQ=F"`, `"^NDX"`
- `BAR_INTERVAL`: `"5m"` or `"1m"` (yfinance 1m has a 7-day limit)
- `LOOKBACK_DAYS`, `HISTORICAL_MONTHS`, ATR/RSI/EMA periods, and labeling thresholds

## Retraining

Retrain weekly or monthly for best results:

```bash
python model/train.py
```

Then restart the Streamlit app so it loads the new model.

## Stretch ideas

- Confusion matrix page (accuracy over last 30 days)
- Telegram/Discord alert when regime changes mid-day
- “Today vs history”: “Today looks most similar to [date], which ended as a Trend Day”

## Disclaimer

Not financial advice. For educational and research use only.
