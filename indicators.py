"""
indicators.py
-------------
Pure pandas/numpy implementations of the same conditions used in the
TradingView Pine scanner: RSI, EMA, ATR, SuperTrend, Donchian/Turtle
breakout, and a relative-volume spike check. No TA-Lib dependency, so this
installs cleanly on any machine or CI runner (e.g. GitHub Actions) with
just `pip install -r requirements.txt`.

All functions expect a pandas DataFrame with columns:
    Open, High, Low, Close, Volume
indexed by datetime, most-recent row last (the normal yfinance/Binance order).
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    """Returns (supertrend_line, direction) where direction is +1 bullish, -1 bearish."""
    atr_val = atr(df, period)
    hl2 = (df["High"] + df["Low"]) / 2
    upperband = hl2 + multiplier * atr_val
    lowerband = hl2 - multiplier * atr_val

    final_upper = upperband.copy()
    final_lower = lowerband.copy()
    direction = pd.Series(1, index=df.index)
    st = pd.Series(0.0, index=df.index)

    close = df["Close"].to_numpy(copy=True)
    up = upperband.to_numpy(copy=True)
    lo = lowerband.to_numpy(copy=True)
    fu = final_upper.to_numpy(copy=True)
    fl = final_lower.to_numpy(copy=True)
    dirn = direction.to_numpy(copy=True)
    line = st.to_numpy(copy=True)

    for i in range(1, len(df)):
        fu[i] = up[i] if (up[i] < fu[i - 1] or close[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lo[i] if (lo[i] > fl[i - 1] or close[i - 1] < fl[i - 1]) else fl[i - 1]

        if dirn[i - 1] == -1 and close[i] > fu[i - 1]:
            dirn[i] = 1
        elif dirn[i - 1] == 1 and close[i] < fl[i - 1]:
            dirn[i] = -1
        else:
            dirn[i] = dirn[i - 1]

        line[i] = fl[i] if dirn[i] == 1 else fu[i]

    return pd.Series(line, index=df.index), pd.Series(dirn, index=df.index)


def donchian_breakout(df: pd.DataFrame, length: int):
    """Returns (breakout_long, breakout_short) booleans for the latest bar,
    using the classic Turtle rule: compare today's close to the highest
    high / lowest low of the PRIOR `length` bars (shifted by 1 so today's
    own bar can't set its own breakout level)."""
    prior_high = df["High"].shift(1).rolling(length).max()
    prior_low = df["Low"].shift(1).rolling(length).min()
    breakout_long = df["Close"] > prior_high
    breakout_short = df["Close"] < prior_low
    return breakout_long, breakout_short


def ema_cross(close: pd.Series, fast_len: int = 9, slow_len: int = 21):
    fast = ema(close, fast_len)
    slow = ema(close, slow_len)
    bull_cross = (fast.shift(1) <= slow.shift(1)) & (fast > slow)
    bear_cross = (fast.shift(1) >= slow.shift(1)) & (fast < slow)
    return bull_cross, bear_cross


def relative_volume(volume: pd.Series, length: int = 20) -> pd.Series:
    avg_vol = volume.rolling(length).mean()
    return (volume / avg_vol.replace(0, np.nan)).fillna(0.0)


def evaluate_conditions(df: pd.DataFrame, cfg: dict) -> dict:
    """
    Runs every condition on the latest confirmed bar of df and returns a
    dict describing what matched. cfg keys (all optional, sensible
    defaults applied): rsi_len, rsi_thresh, st_period, st_mult,
    ema_fast, ema_slow, s1_len, s2_len, rvol_len, rvol_min.
    """
    if len(df) < max(cfg.get("s2_len", 55), cfg.get("rvol_len", 20)) + 5:
        return {"match_long": False, "match_short": False, "tag": "insufficient data"}

    close = df["Close"]
    rsi_val = rsi(close, cfg.get("rsi_len", 14))
    st_line, st_dir = supertrend(df, cfg.get("st_period", 10), cfg.get("st_mult", 3.0))
    a_bullish = (close > st_line) & (rsi_val > cfg.get("rsi_thresh", 50.0))
    a_bearish = (close < st_line) & (rsi_val < (100 - cfg.get("rsi_thresh", 50.0)))

    bull_cross, bear_cross = ema_cross(close, cfg.get("ema_fast", 9), cfg.get("ema_slow", 21))

    rvol = relative_volume(df["Volume"], cfg.get("rvol_len", 20))
    vol_spike = rvol >= cfg.get("rvol_min", 1.5)

    s1_long, s1_short = donchian_breakout(df, cfg.get("s1_len", 20))
    s2_long, s2_short = donchian_breakout(df, cfg.get("s2_len", 55))
    turtle_long = s1_long | s2_long
    turtle_short = s1_short | s2_short

    i = -1  # latest bar
    match_long = bool(turtle_long.iloc[i] or bull_cross.iloc[i] or (a_bullish.iloc[i] and vol_spike.iloc[i]))
    match_short = bool(turtle_short.iloc[i] or bear_cross.iloc[i] or (a_bearish.iloc[i] and vol_spike.iloc[i]))

    if turtle_long.iloc[i] or turtle_short.iloc[i]:
        tag = "S2" if (s2_long.iloc[i] or s2_short.iloc[i]) else "S1"
    elif bull_cross.iloc[i] or bear_cross.iloc[i]:
        tag = "EMA-X"
    elif a_bullish.iloc[i] or a_bearish.iloc[i]:
        tag = "ST/RSI+VOL"
    else:
        tag = "-"

    return {
        "match_long": match_long,
        "match_short": match_short,
        "tag": tag,
        "price": float(close.iloc[i]),
        "rsi": float(rsi_val.iloc[i]),
        "rvol": float(rvol.iloc[i]),
    }
