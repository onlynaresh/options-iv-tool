# 📈 Options IV & Price Tool

A Streamlit app for visualizing options implied volatility, historical volatility, and pricing trends.

## Features

- **Stock & HV** — Candlestick chart with 20/60-day historical volatility
- **IV Snapshot** — IV smile across strikes for any expiration
- **Contract Drilldown** — Historical price + IV trend for a specific contract
- **Option Valuation** — Cheap/expensive indicator based on IV percentile rank
- **Interactive Charts** — Zoom (1W/1M/3M/6M/All), scroll-to-zoom, crosshair hover with dates

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo → Deploy

## Deploy with Docker

```bash
docker build -t options-iv-tool .
docker run -p 8501:8501 options-iv-tool
```

## Data Source

Uses [Yahoo Finance](https://finance.yahoo.com/) via `yfinance`. Historical IV is back-calculated using Black-Scholes from historical option prices.
