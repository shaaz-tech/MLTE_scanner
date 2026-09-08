def detect_liquidity_sweep(df, pivot_left: int = 5, pivot_right: int = 5, min_rng_pct: float = 0.5) -> dict:
    """
    Detects a freshly confirmed SSL (sell-side liquidity / pivot low) or
    BSL (buy-side liquidity / pivot high), mirroring the Pine Script
    ta.pivothigh/ta.pivotlow(5,5) logic, filtered by candle range %.

    Returns a dict like:
        {"ssl": True/False, "bsl": True/False, "rng_pct": float, "pivot_price": float}
    A pivot only confirms `pivot_right` bars after it forms, so this checks
    the bar at position -(pivot_right+1) from the end of df.
    """
    result = {"ssl": False, "bsl": False, "rng_pct": None, "pivot_price": None}

    needed = pivot_left + pivot_right + 1
    if len(df) < needed:
        return result

    # Index of the candidate pivot bar
    pivot_idx = len(df) - pivot_right - 1
    left_slice = df["high"].iloc[pivot_idx - pivot_left: pivot_idx]
    right_slice = df["high"].iloc[pivot_idx + 1: pivot_idx + 1 + pivot_right]
    pivot_high = df["high"].iloc[pivot_idx]

    left_slice_low = df["low"].iloc[pivot_idx - pivot_left: pivot_idx]
    right_slice_low = df["low"].iloc[pivot_idx + 1: pivot_idx + 1 + pivot_right]
    pivot_low = df["low"].iloc[pivot_idx]

    is_pivot_high = (pivot_high > left_slice.max()) and (pivot_high > right_slice.max())
    is_pivot_low = (pivot_low < left_slice_low.min()) and (pivot_low < right_slice_low.min())

    if not (is_pivot_high or is_pivot_low):
        return result

    bar_high = df["high"].iloc[pivot_idx]
    bar_low = df["low"].iloc[pivot_idx]
    bar_close = df["close"].iloc[pivot_idx]
    rng_pct = ((bar_high - bar_low) / bar_close) * 100 if bar_close else 0.0

    if rng_pct < min_rng_pct:
        return result

    result["rng_pct"] = round(rng_pct, 3)
    if is_pivot_high:
        result["bsl"] = True
        result["pivot_price"] = pivot_high
    if is_pivot_low:
        result["ssl"] = True
        result["pivot_price"] = pivot_low

    return result
