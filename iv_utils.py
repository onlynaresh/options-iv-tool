"""
Utility functions for historical volatility calculation and IV computation.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq


def calc_historical_volatility(prices: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate annualized rolling historical volatility from a price series.

    Args:
        prices: Series of closing prices.
        window: Rolling window in trading days.

    Returns:
        Series of annualized rolling volatility values.
    """
    log_returns = np.log(prices / prices.shift(1))
    rolling_std = log_returns.rolling(window=window).std()
    annualized_vol = rolling_std * np.sqrt(252)
    return annualized_vol


# ──────────────────────────────────────────────
# Black-Scholes IV Solver
# ──────────────────────────────────────────────
def bs_price(S, K, T, r, sigma, option_type="call"):
    """
    Black-Scholes option price.

    Args:
        S: Spot price
        K: Strike price
        T: Time to expiry in years
        r: Risk-free rate (annualized)
        sigma: Volatility (annualized)
        option_type: 'call' or 'put'

    Returns:
        Theoretical option price.
    """
    if T <= 0 or sigma <= 0:
        return max(0, S - K) if option_type == "call" else max(0, K - S)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_volatility(price, S, K, T, r, option_type="call"):
    """
    Solve for implied volatility using Brent's method.

    Returns:
        Implied volatility, or NaN if it can't be solved.
    """
    if T <= 0 or price <= 0 or S <= 0:
        return np.nan

    intrinsic = max(0, S - K) if option_type == "call" else max(0, K - S)
    if price < intrinsic * 0.99:
        return np.nan

    try:
        iv = brentq(
            lambda sigma: bs_price(S, K, T, r, sigma, option_type) - price,
            1e-6,
            10.0,
            xtol=1e-6,
            maxiter=200,
        )
        return iv
    except (ValueError, RuntimeError):
        return np.nan


def compute_iv_timeseries(
    contract_hist: pd.DataFrame,
    stock_hist: pd.DataFrame,
    strike: float,
    expiration_date: str,
    option_type: str = "call",
    risk_free_rate: float = 0.045,
) -> pd.Series:
    """
    Compute implied volatility for each day from historical option prices.

    Args:
        contract_hist: DataFrame of option contract prices (must have 'Close' column).
        stock_hist: DataFrame of stock prices (must have 'Close' column).
        strike: Strike price of the option.
        expiration_date: Expiration date as a string (YYYY-MM-DD).
        option_type: 'call' or 'put'.
        risk_free_rate: Annualized risk-free rate.

    Returns:
        Series of daily IV values indexed by date.
    """
    exp_dt = pd.Timestamp(expiration_date)

    # Build a lookup dict of stock closing prices keyed by date string (YYYY-MM-DD)
    # This avoids timezone-aware vs naive comparison issues between
    # yf.Ticker.history() (tz-aware) and yf.download() (tz-naive)
    stock_by_date = {}
    for idx, val in stock_hist["Close"].items():
        date_str = pd.Timestamp(idx).strftime("%Y-%m-%d")
        stock_by_date[date_str] = float(val)

    iv_values = {}

    for i in range(len(contract_hist)):
        date = contract_hist.index[i]
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")

        stock_price = stock_by_date.get(date_str)
        if stock_price is None or stock_price <= 0:
            continue

        option_price = float(contract_hist["Close"].iloc[i])

        # Time to expiry in years
        days_to_exp = (exp_dt - pd.Timestamp(date_str)).days
        T = days_to_exp / 365.0

        if T <= 0 or option_price <= 0:
            continue

        iv = implied_volatility(option_price, stock_price, strike, T, risk_free_rate, option_type)
        iv_values[date] = iv

    return pd.Series(iv_values, name="IV")


def format_number(value, decimals=2):
    """Format a number for display, handling NaN gracefully."""
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


def format_percent(value, decimals=2):
    """Format a decimal as a percentage string."""
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:,.{decimals}f}%"

