"""
Options IV & Price Trend Charting Tool
Run: streamlit run app.py
"""

import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from iv_utils import calc_historical_volatility, compute_iv_timeseries, format_number, format_percent


# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Options IV & Price Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a3e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #e0e0ff;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e1e3f 0%, #2a2a5a 100%);
        border: 1px solid rgba(100, 100, 255, 0.2);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    div[data-testid="stMetric"] label {
        color: #8888cc !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 24px;
        font-weight: 500;
    }

    /* Headers */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
    }

    /* Dataframe */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Timeframe mapping
# ──────────────────────────────────────────────
TIMEFRAMES = {
    "1 Week": 7,
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "1 Year": 365,
}


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Options IV Tool")
    st.markdown("---")

    ticker_input = st.text_input(
        "Ticker Symbol",
        value="AAPL",
        placeholder="e.g. AAPL, TSLA, SPY",
        help="Enter a valid stock ticker symbol",
    ).upper().strip()

    timeframe = st.selectbox(
        "Timeframe",
        options=list(TIMEFRAMES.keys()),
        index=1,  # default 1 Month
        help="Historical lookback period for charts",
    )

    load_btn = st.button("🚀 Load Data", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown(
        "<small style='color:#666'>Data from Yahoo Finance via yfinance. "
        "IV is a point-in-time snapshot.</small>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# Data loading (cached)
# ──────────────────────────────────────────────
import time

def _retry_on_rate_limit(func, max_retries=4, base_delay=3):
    """Retry a yfinance call with exponential backoff on rate limit errors."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            err_str = str(e).lower()
            if "RateLimit" in type(e).__name__ or "rate" in err_str or "429" in err_str or "too many requests" in err_str:
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
                    continue
            raise
    return None


@st.cache_data(ttl=1800, show_spinner=False)
def load_stock_data(ticker: str, days: int):
    """Fetch stock price history."""
    def _fetch():
        time.sleep(1.0)  # Throttling
        tk = yf.Ticker(ticker)
        end = datetime.now()
        start = end - timedelta(days=days + 10)
        hist = tk.history(start=start, end=end)
        if hist.empty:
            return None, "Empty history returned"
        info = tk.info
        return hist, info
    try:
        return _retry_on_rate_limit(_fetch)
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=1800, show_spinner=False)
def load_options_chain(ticker: str):
    """Fetch all available options expiration dates."""
    def _fetch():
        time.sleep(1.0)  # Throttling
        tk = yf.Ticker(ticker)
        expirations = tk.options
        if not expirations:
            return None
        return expirations
    try:
        return _retry_on_rate_limit(_fetch)
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def get_option_chain(ticker: str, expiration: str):
    """Get calls and puts for a specific expiration."""
    def _fetch():
        time.sleep(1.0)  # Throttling
        tk = yf.Ticker(ticker)
        chain = tk.option_chain(expiration)
        return chain.calls, chain.puts
    return _retry_on_rate_limit(_fetch)


@st.cache_data(ttl=3600, show_spinner=False)
def load_option_contract_history(contract_symbol: str, days: int = -1):
    """Fetch historical price data for a specific option contract using Ticker.history."""
    def _fetch():
        time.sleep(1.5)  # Throttling (more generous for heavy history call)
        tk = yf.Ticker(contract_symbol)
        
        if days <= 0:
            data = tk.history(period="max")
        else:
            end = datetime.now()
            start = end - timedelta(days=days)
            data = tk.history(start=start, end=end)
            
        if data.empty:
            return None
        return data
        
    try:
        return _retry_on_rate_limit(_fetch)
    except Exception:
        return None


# ──────────────────────────────────────────────
# Chart builders
# ──────────────────────────────────────────────
PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,15,35,0.8)",
    font=dict(family="Inter, sans-serif", color="#e0e0ff"),
    margin=dict(l=60, r=30, t=50, b=40),
    legend=dict(
        bgcolor="rgba(0,0,0,0.3)",
        bordercolor="rgba(100,100,255,0.3)",
        borderwidth=1,
    ),
    xaxis=dict(gridcolor="rgba(100,100,255,0.1)", zeroline=False),
    yaxis=dict(gridcolor="rgba(100,100,255,0.1)", zeroline=False),
    hovermode="x unified",
)

# Plotly chart config with scroll-to-zoom enabled
PLOTLY_CONFIG = dict(
    scrollZoom=True,
    displayModeBar=True,
    modeBarButtonsToAdd=["drawline", "eraseshape"],
    displaylogo=False,
)

# Reusable range selector buttons for time-series charts
RANGE_SELECTOR = dict(
    buttons=[
        dict(count=7, label="1W", step="day", stepmode="backward"),
        dict(count=1, label="1M", step="month", stepmode="backward"),
        dict(count=3, label="3M", step="month", stepmode="backward"),
        dict(count=6, label="6M", step="month", stepmode="backward"),
        dict(step="all", label="All"),
    ],
    bgcolor="rgba(30,30,63,0.8)",
    activecolor="#7c4dff",
    font=dict(color="#e0e0ff"),
)


def build_stock_chart(hist: pd.DataFrame, hv_20: pd.Series, hv_60: pd.Series):
    """Build the candlestick + HV overlay chart."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.6, 0.4],
        subplot_titles=["Price", "Historical Volatility (Annualized)"],
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=hist.index,
            open=hist["Open"],
            high=hist["High"],
            low=hist["Low"],
            close=hist["Close"],
            name="Price",
            increasing_line_color="#00d4aa",
            decreasing_line_color="#ff4757",
        ),
        row=1, col=1,
    )

    # Volume as bar chart behind candlestick
    colors = [
        "#00d4aa" if c >= o else "#ff4757"
        for c, o in zip(hist["Close"], hist["Open"])
    ]
    fig.add_trace(
        go.Bar(
            x=hist.index,
            y=hist["Volume"],
            name="Volume",
            marker_color=colors,
            opacity=0.15,
            yaxis="y3",
            hovertemplate="Vol: %{y:,.0f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # HV lines
    fig.add_trace(
        go.Scatter(
            x=hv_20.index, y=hv_20.values,
            name="HV 20-day",
            line=dict(color="#7c4dff", width=2),
            hovertemplate="%{x|%b %d, %Y}<br>HV 20d: %{y:.1%}<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=hv_60.index, y=hv_60.values,
            name="HV 60-day",
            line=dict(color="#ff9800", width=2, dash="dot"),
            hovertemplate="%{x|%b %d, %Y}<br>HV 60d: %{y:.1%}<extra></extra>",
        ),
        row=2, col=1,
    )

    fig.update_layout(
        **PLOT_LAYOUT,
        height=650,
        xaxis_rangeslider_visible=False,
        showlegend=True,
    )
    fig.update_xaxes(rangeselector=RANGE_SELECTOR, row=1, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volatility", tickformat=".0%", row=2, col=1)

    return fig


def build_iv_smile(calls: pd.DataFrame, puts: pd.DataFrame, current_price: float, expiration: str):
    """Build the IV smile chart."""
    fig = go.Figure()

    # Filter to reasonable strike range (±30% from current price)
    strike_min = current_price * 0.7
    strike_max = current_price * 1.3

    c = calls[(calls["strike"] >= strike_min) & (calls["strike"] <= strike_max)].copy()
    p = puts[(puts["strike"] >= strike_min) & (puts["strike"] <= strike_max)].copy()

    fig.add_trace(go.Scatter(
        x=c["strike"], y=c["impliedVolatility"],
        name="Calls IV",
        mode="lines+markers",
        line=dict(color="#00d4aa", width=2.5),
        marker=dict(size=6),
        hovertemplate="Strike: $%{x:.0f}<br>IV: %{y:.1%}<extra>Calls</extra>",
    ))

    fig.add_trace(go.Scatter(
        x=p["strike"], y=p["impliedVolatility"],
        name="Puts IV",
        mode="lines+markers",
        line=dict(color="#ff4757", width=2.5),
        marker=dict(size=6),
        hovertemplate="Strike: $%{x:.0f}<br>IV: %{y:.1%}<extra>Puts</extra>",
    ))

    # Current price vertical line
    fig.add_vline(
        x=current_price,
        line_dash="dash",
        line_color="#ffd700",
        annotation_text=f"Spot ${current_price:.2f}",
        annotation_position="top",
        annotation_font_color="#ffd700",
    )

    fig.update_layout(
        **PLOT_LAYOUT,
        height=500,
        title=f"IV Smile — Exp: {expiration}",
        xaxis_title="Strike Price ($)",
        yaxis_title="Implied Volatility",
        yaxis_tickformat=".0%",
    )

    return fig


def build_contract_chart(data: pd.DataFrame, contract_symbol: str):
    """Build chart for a specific option contract's price history."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=["Contract Price", "Volume"],
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Close"],
            name="Close",
            mode="lines",
            line=dict(color="#00d4aa", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(0,212,170,0.1)",
            hovertemplate="%{x|%b %d, %Y}<br>Close: $%{y:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )

    if "High" in data.columns and "Low" in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data["High"],
                name="High",
                mode="lines",
                line=dict(color="rgba(0,212,170,0.3)", width=1, dash="dot"),
                hovertemplate="High: $%{y:.2f}<extra></extra>",
                showlegend=False,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data["Low"],
                name="Low",
                mode="lines",
                line=dict(color="rgba(255,71,87,0.3)", width=1, dash="dot"),
                fill="tonexty",
                fillcolor="rgba(100,100,255,0.05)",
                hovertemplate="Low: $%{y:.2f}<extra></extra>",
                showlegend=False,
            ),
            row=1, col=1,
        )

    if "Volume" in data.columns:
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data["Volume"],
                name="Volume",
                marker_color="#7c4dff",
                opacity=0.6,
                hovertemplate="Vol: %{y:,.0f}<extra></extra>",
            ),
            row=2, col=1,
        )

    fig.update_layout(
        **PLOT_LAYOUT,
        height=550,
        title=f"Contract: {contract_symbol}",
    )
    fig.update_xaxes(rangeselector=RANGE_SELECTOR, row=1, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


def build_iv_trend_chart(iv_series: pd.Series, hv_20: pd.Series, contract_symbol: str):
    """Build IV trend over time chart with HV comparison."""
    fig = go.Figure()

    # IV trend line
    iv_clean = iv_series.dropna()
    if not iv_clean.empty:
        fig.add_trace(go.Scatter(
            x=iv_clean.index,
            y=iv_clean.values,
            name="Implied Volatility",
            mode="lines+markers",
            line=dict(color="#00d4aa", width=2.5),
            marker=dict(size=5),
            fill="tozeroy",
            fillcolor="rgba(0,212,170,0.08)",
            hovertemplate="%{x|%b %d, %Y}<br>IV: %{y:.1%}<extra></extra>",
        ))

    # HV overlay for comparison
    if hv_20 is not None and not hv_20.empty:
        fig.add_trace(go.Scatter(
            x=hv_20.index,
            y=hv_20.values,
            name="HV 20-day (stock)",
            mode="lines",
            line=dict(color="#7c4dff", width=2, dash="dash"),
            hovertemplate="%{x|%b %d, %Y}<br>HV 20d: %{y:.1%}<extra></extra>",
        ))

    fig.update_layout(
        **PLOT_LAYOUT,
        height=450,
        title=f"IV Trend — {contract_symbol}",
        xaxis_title="Date",
        yaxis_title="Volatility",
        yaxis_tickformat=".0%",
    )
    fig.update_xaxes(rangeselector=RANGE_SELECTOR)

    return fig


# ──────────────────────────────────────────────
# Main content
# ──────────────────────────────────────────────
st.markdown(f"# 📈 Options IV & Price Tool")

if not load_btn and "loaded_ticker" not in st.session_state:
    st.markdown(
        """
        <div style='text-align:center; padding: 80px 20px; color: #8888cc;'>
            <h2>Enter a ticker and click Load Data to get started</h2>
            <p style='font-size: 1.1rem; margin-top: 12px;'>
                Analyze implied volatility, historical volatility,<br>
                and option contract price trends.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# Determine which ticker to use
if load_btn:
    st.session_state["loaded_ticker"] = ticker_input
    st.session_state["loaded_timeframe"] = timeframe

active_ticker = st.session_state.get("loaded_ticker", ticker_input)
active_timeframe = st.session_state.get("loaded_timeframe", timeframe)
days = TIMEFRAMES[active_timeframe]

# ── Load stock data ──
with st.spinner(f"Loading data for {active_ticker}..."):
    hist, info = load_stock_data(active_ticker, days)

if hist is None:
    if info: # Should be error message string in second return value if hist is None
         st.error(f"Could not load data for {active_ticker}. Error: {info}")
    else:
         st.error(f"Could not load data for {active_ticker}. Check the ticker symbol and try again.")
    st.stop()

current_price = hist["Close"].iloc[-1]
# Trim to requested timeframe
cutoff = datetime.now() - timedelta(days=days)
hist = hist[hist.index >= cutoff.strftime("%Y-%m-%d")]

if hist.empty:
    st.error(f"❌ No data available for **{active_ticker}** in the selected timeframe.")
    st.stop()

# ── Compute HV ──
# Need extra history for rolling calc, so reload with buffer
full_hist, _ = load_stock_data(active_ticker, days + 80)
hv_20 = calc_historical_volatility(full_hist["Close"], window=20)
hv_60 = calc_historical_volatility(full_hist["Close"], window=60)
hv_20 = hv_20[hv_20.index >= cutoff.strftime("%Y-%m-%d")]
hv_60 = hv_60[hv_60.index >= cutoff.strftime("%Y-%m-%d")]

# ── Header metrics ──
current_price = hist["Close"].iloc[-1]
price_change = hist["Close"].iloc[-1] - hist["Close"].iloc[0]
price_change_pct = price_change / hist["Close"].iloc[0]
current_hv20 = hv_20.dropna().iloc[-1] if not hv_20.dropna().empty else None

company_name = info.get("shortName", active_ticker) if info else active_ticker

st.markdown(f"### {company_name} ({active_ticker})")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Current Price", f"${current_price:.2f}", f"{price_change:+.2f} ({price_change_pct:+.1%})")
with col2:
    st.metric("Period High", f"${hist['High'].max():.2f}")
with col3:
    st.metric("Period Low", f"${hist['Low'].min():.2f}")
with col4:
    hv_display = f"{current_hv20:.1%}" if current_hv20 is not None else "N/A"
    st.metric("HV (20-day)", hv_display)

st.markdown("---")

# ──────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Stock & HV", "🎯 IV Snapshot", "🔍 Contract Drilldown"])

# ── Tab 1: Stock + HV ──
with tab1:
    st.plotly_chart(
        build_stock_chart(hist, hv_20, hv_60),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )
    with st.expander("ℹ️ About Historical Volatility"):
        st.markdown("""
        **Historical Volatility (HV)** measures the actual realized volatility of the stock
        based on past price movements. It's calculated as the annualized standard deviation
        of log returns over a rolling window.

        - **HV 20-day** (purple) — short-term volatility, ~1 month of trading
        - **HV 60-day** (orange, dashed) — medium-term volatility, ~3 months of trading

        HV is useful as a **baseline comparison** against implied volatility (IV). When IV is
        significantly higher than HV, options may be "expensive" relative to realized moves.
        """)

# ── Tab 2: IV Snapshot ──
with tab2:
    with st.spinner("Loading options chain..."):
        expirations = load_options_chain(active_ticker)

    if expirations is None or len(expirations) == 0:
        st.warning(f"⚠️ No options data available for **{active_ticker}**.")
    else:
        selected_exp = st.selectbox(
            "Expiration Date",
            options=expirations,
            index=0,
            key="iv_expiration",
        )

        calls, puts = get_option_chain(active_ticker, selected_exp)

        # IV Smile chart
        st.plotly_chart(
            build_iv_smile(calls, puts, current_price, selected_exp),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

        # Summary metrics
        atm_calls = calls.iloc[(calls["strike"] - current_price).abs().argsort()[:3]]
        atm_puts = puts.iloc[(puts["strike"] - current_price).abs().argsort()[:3]]

        col_a, col_b = st.columns(2)
        with col_a:
            avg_call_iv = atm_calls["impliedVolatility"].mean()
            st.metric("ATM Call IV (avg)", format_percent(avg_call_iv))
        with col_b:
            avg_put_iv = atm_puts["impliedVolatility"].mean()
            st.metric("ATM Put IV (avg)", format_percent(avg_put_iv))

        # Options chain tables
        st.markdown("#### Options Chain")
        display_cols = [
            "contractSymbol", "strike", "lastPrice", "bid", "ask",
            "volume", "openInterest", "impliedVolatility",
        ]

        chain_tab1, chain_tab2 = st.tabs(["Calls", "Puts"])
        with chain_tab1:
            available_cols = [c for c in display_cols if c in calls.columns]
            display_calls = calls[available_cols].copy()
            display_calls["impliedVolatility"] = display_calls["impliedVolatility"].apply(
                lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
            )
            st.dataframe(display_calls, use_container_width=True, hide_index=True)

        with chain_tab2:
            available_cols = [c for c in display_cols if c in puts.columns]
            display_puts = puts[available_cols].copy()
            display_puts["impliedVolatility"] = display_puts["impliedVolatility"].apply(
                lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
            )
            st.dataframe(display_puts, use_container_width=True, hide_index=True)

# ── Tab 3: Contract Drilldown ──
with tab3:
    with st.spinner("Loading options chain..."):
        expirations2 = load_options_chain(active_ticker)

    if expirations2 is None or len(expirations2) == 0:
        st.warning(f"⚠️ No options data available for **{active_ticker}**.")
    else:
        col_exp, col_type = st.columns(2)
        with col_exp:
            drill_exp = st.selectbox(
                "Expiration Date",
                options=expirations2,
                index=0,
                key="drill_expiration",
            )
        with col_type:
            option_type = st.radio(
                "Option Type",
                ["Call", "Put"],
                horizontal=True,
                key="drill_type",
            )

        drill_calls, drill_puts = get_option_chain(active_ticker, drill_exp)
        chain_data = drill_calls if option_type == "Call" else drill_puts

        if chain_data.empty:
            st.warning("No contracts available for this selection.")
        else:
            # Create readable labels for the strike selector
            chain_data = chain_data.copy()
            chain_data["label"] = chain_data.apply(
                lambda r: f"${r['strike']:.0f} — Last: ${r['lastPrice']:.2f} — IV: {r['impliedVolatility']:.1%} — OI: {r.get('openInterest', 'N/A')}",
                axis=1,
            )

            # Default to ATM
            atm_idx = (chain_data["strike"] - current_price).abs().idxmin()
            atm_pos = chain_data.index.get_loc(atm_idx)

            selected_label = st.selectbox(
                "Select Strike",
                options=chain_data["label"].tolist(),
                index=int(atm_pos),
                key="drill_strike",
            )

            selected_row = chain_data[chain_data["label"] == selected_label].iloc[0]
            contract_sym = selected_row["contractSymbol"]

            # Contract info cards
            ci1, ci2, ci3, ci4 = st.columns(4)
            with ci1:
                st.metric("Strike", f"${selected_row['strike']:.2f}")
            with ci2:
                st.metric("Last Price", f"${selected_row['lastPrice']:.2f}")
            with ci3:
                st.metric("IV", format_percent(selected_row["impliedVolatility"]))
            with ci4:
                oi = selected_row.get("openInterest", "N/A")
                st.metric("Open Interest", f"{int(oi):,}" if pd.notna(oi) else "N/A")

            st.markdown("---")

            # Historical price of the contract — fetch FULL available history
            with st.spinner(f"Loading history for {contract_sym}..."):
                contract_hist = load_option_contract_history(contract_sym, days=-1)

            if contract_hist is None or contract_hist.empty:
                st.info(
                    "📭 No historical price data available for this contract. "
                    "This is common for newer or illiquid contracts — Yahoo Finance "
                    "may not have historical data for all option contracts."
                )
            else:
                st.plotly_chart(
                    build_contract_chart(contract_hist, contract_sym),
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                )

                # Price change stats
                if len(contract_hist) > 1:
                    first_close = contract_hist["Close"].iloc[0]
                    last_close = contract_hist["Close"].iloc[-1]
                    change = last_close - first_close
                    change_pct = change / first_close if first_close != 0 else 0

                    sc1, sc2, sc3 = st.columns(3)
                    with sc1:
                        st.metric("Period Open", f"${first_close:.2f}")
                    with sc2:
                        st.metric("Period Close", f"${last_close:.2f}")
                    with sc3:
                        st.metric("Change", f"${change:+.2f} ({change_pct:+.1%})")

                # ── IV Trend Chart ──
                st.markdown("---")
                st.markdown("#### 📈 Implied Volatility Trend")
                st.caption(
                    "IV is back-calculated from historical option prices using Black-Scholes. "
                    "The stock's 20-day HV is shown as a dashed line for comparison."
                )

                # Load extended stock history to cover the contract's full life
                contract_start = contract_hist.index[0]
                contract_days = (datetime.now() - pd.Timestamp(contract_start).to_pydatetime().replace(tzinfo=None)).days + 30
                extended_stock_hist, _ = load_stock_data(active_ticker, max(contract_days, days + 80))
                extended_hv_20 = calc_historical_volatility(extended_stock_hist["Close"], window=20)

                opt_type = "call" if option_type == "Call" else "put"
                with st.spinner("Computing IV from historical prices..."):
                    iv_series = compute_iv_timeseries(
                        contract_hist=contract_hist,
                        stock_hist=extended_stock_hist,
                        strike=selected_row["strike"],
                        expiration_date=drill_exp,
                        option_type=opt_type,
                    )

                iv_valid = iv_series.dropna()

                if iv_valid.empty:
                    st.info(
                        "⚠️ Could not compute IV for this contract's history. "
                        "This can happen with deep ITM/OTM options or very low liquidity."
                    )
                else:
                    st.plotly_chart(
                        build_iv_trend_chart(iv_series, extended_hv_20, contract_sym),
                        use_container_width=True,
                        config=PLOTLY_CONFIG,
                    )

                    # IV summary metrics
                    iv1, iv2, iv3 = st.columns(3)
                    with iv1:
                        st.metric("Current IV", f"{iv_valid.iloc[-1]:.1%}")
                    with iv2:
                        st.metric("IV High", f"{iv_valid.max():.1%}")
                    with iv3:
                        st.metric("IV Low", f"{iv_valid.min():.1%}")

                    # ── Cheap / Expensive indicator ──
                    current_iv = iv_valid.iloc[-1]
                    iv_pct_rank = (iv_valid < current_iv).sum() / len(iv_valid) * 100

                    if iv_pct_rank <= 25:
                        label = "💚 CHEAP"
                        color = "#00d4aa"
                        desc = "IV is in the **lower quartile** of its historical range — options are relatively cheap."
                    elif iv_pct_rank <= 50:
                        label = "🟡 FAIR (Low Side)"
                        color = "#ffd700"
                        desc = "IV is **below the median** — options are reasonably priced."
                    elif iv_pct_rank <= 75:
                        label = "🟠 FAIR (High Side)"
                        color = "#ff9800"
                        desc = "IV is **above the median** — options are slightly elevated."
                    else:
                        label = "🔴 EXPENSIVE"
                        color = "#ff4757"
                        desc = "IV is in the **upper quartile** of its historical range — options are relatively expensive."

                    st.markdown("---")
                    st.markdown("#### 💰 Option Valuation")
                    st.markdown(
                        f'<div style="background: linear-gradient(135deg, #1e1e3f, #2a2a5a); '
                        f'border-left: 4px solid {color}; border-radius: 10px; '
                        f'padding: 20px; margin: 10px 0;">'
                        f'<span style="font-size: 1.4em; font-weight: 700; color: {color};">{label}</span>'
                        f'<span style="float: right; font-size: 1.2em; color: #e0e0ff;">'
                        f'IV Percentile: <strong>{iv_pct_rank:.0f}%</strong></span>'
                        f'<br><br><span style="color: #c0c0e0;">{desc}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"Based on {len(iv_valid)} data points. Current IV ({current_iv:.1%}) is higher than "
                        f"{iv_pct_rank:.0f}% of historical observations for this contract."
                    )
