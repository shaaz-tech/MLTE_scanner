"""
indicators.py
-------------
Technical indicator calculations and condition evaluation used by scanner.py.

Implements:
  - EMA9/21 crossover
  - Turtle-style breakout (20/55 period channel breakout)
  - SuperTrend + RSI filter
  - RVOL (relative volume vs its own rolling average)
  - evaluate_conditions(df, config) -> dict consumed by scanner.py
  - detect_liquidity_sweep(df, ...) -> SSL/BSL (pivot-based liquidity sweep) detector

Expected input `df` columns: "open", "high", "low", "close", "volume"
(standard OHLCV naming used by yfinance and Kraken data pulls in data_sources.py)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ==========================================
# Core indicator helpers
# ==========================================

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    """Returns a boolean Series: True = price above SuperTrend line (bullish)."""
    hl2 = (df["high"] + df["low"]) / 2
    atr_val = atr(df, period)
    upper_band = hl2 + multiplier * atr_val
    lower_band = hl2 - multiplier * atr_val

    final_upper = upper_band.copy()
    final_lower = lower_band.copy()
    trend_up = pd.Series(True, index=df.index)

    for i in range(1, len(df)):
        close_prev = df["close"].iloc[i - 1]

        if upper_band.iloc[i] < final_upper.iloc[i - 1] or close_prev > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = upper_band.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        if lower_band.iloc[i] > final_lower.iloc[i - 1] or close_prev < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = lower_band.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

        close_now = df["close"].iloc[i]
        if trend_up.iloc[i - 1] and close_now < final_lower.iloc[i]:
            trend_up.iloc[i] = False
        elif (not trend_up.iloc[i - 1]) and close_now > final_upper.iloc[i]:
            trend_up.iloc[i] = True
        else:
            trend_up.iloc[i] = trend_up.iloc[i - 1]

    return trend_up


def rvol(df: pd.DataFrame, length: int = 20) -> pd.Series:
    avg_vol = df["volume"].rolling(length).mean()
    return df["volume"] / avg_vol.replace(0, np.nan)


# ==========================================
# Main condition evaluator (used by scanner.py)
# ==========================================

def evaluate_conditions(df: pd.DataFrame, config: dict) -> dict:
    """
    Evaluates the combined strategy conditions on the LAST CLOSED bar of df.

    Returns:
        {
            "match_long": bool,
            "match_short": bool,
            "rvol": float,
            "close": float,
        }
    """
    if df is None or len(df) < max(config["s2_len"], config["ema_slow"], config["rvol_len"]) + 5:
        return {"match_long": False, "match_short": False, "rvol": 0.0, "close": None}

    df = df.copy()

    ema_fast = ema(df["close"], config["ema_fast"])
    ema_slow = ema(df["close"], config["ema_slow"])
    rsi_val = rsi(df["close"], config["rsi_len"])
    trend_up = supertrend(df, config["st_period"], config["st_mult"])
    rvol_val = rvol(df, config["rvol_len"])

    donchian_high_s1 = df["high"].rolling(config["s1_len"]).max()
    donchian_low_s1 = df["low"].rolling(config["s1_len"]).min()
    donchian_high_s2 = df["high"].rolling(config["s2_len"]).max()
    donchian_low_s2 = df["low"].rolling(config["s2_len"]).min()

    i = -1     # last closed bar
    prev = -2  # bar before that

    ema_cross_up = ema_fast.iloc[prev] <= ema_slow.iloc[prev] and ema_fast.iloc[i] > ema_slow.iloc[i]
    ema_cross_down = ema_fast.iloc[prev] >= ema_slow.iloc[prev] and ema_fast.iloc[i] < ema_slow.iloc[i]

    turtle_breakout_long = df["close"].iloc[i] > donchian_high_s1.iloc[prev] or df["close"].iloc[i] > donchian_high_s2.iloc[prev]
    turtle_breakout_short = df["close"].iloc[i] < donchian_low_s1.iloc[prev] or df["close"].iloc[i] < donchian_low_s2.iloc[prev]

    st_bull = bool(trend_up.iloc[i])
    st_bear = not st_bull

    rsi_bull = rsi_val.iloc[i] >= config["rsi_thresh"]
    rsi_bear = rsi_val.iloc[i] < config["rsi_thresh"]

    current_rvol = rvol_val.iloc[i]
    current_rvol = 0.0 if pd.isna(current_rvol) else float(current_rvol)
    vol_ok = current_rvol >= config.get("rvol_min", 1.5)

    match_long = bool((ema_cross_up or turtle_breakout_long) and st_bull and rsi_bull and vol_ok)
    match_short = bool((ema_cross_down or turtle_breakout_short) and st_bear and rsi_bear and vol_ok)

    return {
        "match_long": match_long,
        "match_short": match_short,
        "rvol": round(current_rvol, 2),
        "close": float(df["close"].iloc[i]),
    }


# ==========================================
# Liquidity sweep detector (SSL / BSL) — NEW
# ==========================================

def detect_liquidity_sweep(
    df: pd.DataFrame,
    pivot_left: int = 5,
    pivot_right: int = 5,
    min_rng_pct: float = 0.5,
) -> dict:
    """
    Detects a freshly confirmed SSL (sell-side liquidity / pivot low) or
    BSL (buy-side liquidity / pivot high), mirroring the Pine Script
    ta.pivothigh/ta.pivotlow(5,5) logic, filtered by candle range %.

    Range filter uses percentage of price (not absolute), so it behaves
    consistently across NSE stocks and crypto regardless of price scale:
        rng_pct = (high - low) / close * 100

    Returns a dict like:
        {"ssl": True/False, "bsl": True/False, "rng_pct": float, "pivot_price": float}

    A pivot only confirms `pivot_right` bars after it forms, so this checks
    the bar at position -(pivot_right+1) from the end of df.
    """
    result = {"ssl": False, "bsl": False, "rng_pct": None, "pivot_price": None}

    needed = pivot_left + pivot_right + 1
    if df is None or len(df) < needed:
        return result

    pivot_idx = len(df) - pivot_right - 1

    left_high = df["high"].iloc[pivot_idx - pivot_left: pivot_idx]
    right_high = df["high"].iloc[pivot_idx + 1: pivot_idx + 1 + pivot_right]
    pivot_high = df["high"].iloc[pivot_idx]

    left_low = df["low"].iloc[pivot_idx - pivot_left: pivot_idx]
    right_low = df["low"].iloc[pivot_idx + 1: pivot_idx + 1 + pivot_right]
    pivot_low = df["low"].iloc[pivot_idx]

    is_pivot_high = (pivot_high > left_high.max()) and (pivot_high > right_high.max())
    is_pivot_low = (pivot_low < left_low.min()) and (pivot_low < right_low.min())

    if not (is_pivot_high or is_pivot_low):
        return result

    bar_high = df["high"].iloc[pivot_idx]
    bar_low = df["low"].iloc[pivot_idx]
    bar_close = df["close"].iloc[pivot_idx]
    rng_pct = ((bar_high - bar_low) / bar_close) * 100 if bar_close else 0.0

    if rng_pct < min_rng_pct:
        return result

    result["rng_pct"] = round(float(rng_pct), 3)
    if is_pivot_high:
        result["bsl"] = True
        result["pivot_price"] = float(pivot_high)
    if is_pivot_low:
        result["ssl"] = True
        result["pivot_price"] = float(pivot_low)

    return result
