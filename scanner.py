"""
scanner.py
----------
Main entry point. Scans the FULL NSE equity list + a broad crypto/meme-coin
universe for the same conditions used in the TradingView system (Turtle
breakout, EMA9/21 cross, SuperTrend+RSI with a volume spike), and sends a
Telegram message listing every symbol that matched.

Also scans both markets for freshly confirmed SSL/BSL (retail liquidity
pivot) sweeps with a meaningful candle range (Rng% >= LIQ_MIN_RNG_PCT),
and alerts separately for those regardless of the main strategy match.

Run manually:      python scanner.py
Run on a schedule:  see .github/workflows/scan.yml (or any cron/scheduler)

Configuration is in CONFIG below -- edit thresholds there, no need to touch
the indicator/data-source modules for routine tuning.
"""

from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import data_sources as ds
from indicators import evaluate_conditions, detect_liquidity_sweep
from notifier import send_telegram_message, format_match_message

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("scanner")

CONFIG = {
    "rsi_len": 14,
    "rsi_thresh": 50.0,
    "st_period": 10,
    "st_mult": 3.0,
    "ema_fast": 9,
    "ema_slow": 21,
    "s1_len": 20,
    "s2_len": 55,
    "rvol_len": 20,
    "rvol_min": 1.5,
    "liq_pivot_left": 5,
    "liq_pivot_right": 5,
    "liq_min_rng_pct": 0.5,   # SSL/BSL alert only if candle range >= 0.5% of close
}

NSE_INTERVAL = "15m"      # yfinance intraday interval
CRYPTO_INTERVAL = "15m"   # Kraken kline interval
CRYPTO_TOP_N = 200        # top coins by market cap to include
CRYPTO_MAX_MEME = 150     # cap on meme-token category coins
MIN_RVOL_ALERT = 4.0      # NSE-only: notify main-strategy matches with RVOL >= this value


def is_nse_market_hours() -> bool:
    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
    if now_ist.weekday() >= 5:  # Sat=5, Sun=6
        return False
    open_time = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    close_time = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
    return open_time <= now_ist <= close_time


def _check_liquidity_sweep(symbol: str, df) -> dict | None:
    liq = detect_liquidity_sweep(
        df,
        pivot_left=CONFIG["liq_pivot_left"],
        pivot_right=CONFIG["liq_pivot_right"],
        min_rng_pct=CONFIG["liq_min_rng_pct"],
    )
    if liq["ssl"] or liq["bsl"]:
        return {
            "symbol": symbol,
            "liquidity_alert": "SSL" if liq["ssl"] else "BSL",
            "rng_pct": liq["rng_pct"],
            "pivot_price": liq["pivot_price"],
        }
    return None


def scan_nse() -> tuple[list[dict], list[dict]]:
    logger.info("Starting NSE scan...")
    symbols = ds.get_nse_symbol_list()
    candles = ds.get_nse_candles_batch(symbols, interval=NSE_INTERVAL)

    matches = []
    liquidity_alerts = []
    for symbol, df in candles.items():
        try:
            result = evaluate_conditions(df, CONFIG)
        except Exception as e:
            logger.warning(f"Condition eval failed for {symbol}: {e}")
            continue
        if result.get("match_long") or result.get("match_short"):
            matches.append({"symbol": symbol, **result})

        try:
            liq_hit = _check_liquidity_sweep(symbol, df)
            if liq_hit:
                liquidity_alerts.append(liq_hit)
        except Exception as e:
            logger.warning(f"Liquidity sweep check failed for {symbol}: {e}")

    # NSE keeps the RVOL >= 4x filter on the main strategy matches
    matches = [m for m in matches if m["rvol"] >= MIN_RVOL_ALERT]
    logger.info(
        f"NSE scan complete: {len(matches)} strategy matches (RVOL >= {MIN_RVOL_ALERT}x), "
        f"{len(liquidity_alerts)} liquidity sweep alerts, out of {len(candles)} scanned"
    )
    return matches, liquidity_alerts


def scan_crypto() -> tuple[list[dict], list[dict]]:
    logger.info("Starting crypto scan...")
    coin_symbols = ds.get_crypto_universe(top_n=CRYPTO_TOP_N, max_meme_coins=CRYPTO_MAX_MEME)
    pairs = ds.match_to_kraken_usd_pairs(coin_symbols)

    matches = []
    liquidity_alerts = []
    for pair in pairs:
        df = ds.get_kraken_candles(pair, interval=CRYPTO_INTERVAL)
        if df is None:
            continue
        try:
            result = evaluate_conditions(df, CONFIG)
        except Exception as e:
            logger.warning(f"Condition eval failed for {pair}: {e}")
            continue
        if result.get("match_long") or result.get("match_short"):
            matches.append({"symbol": pair, **result})

        try:
            liq_hit = _check_liquidity_sweep(pair, df)
            if liq_hit:
                liquidity_alerts.append(liq_hit)
        except Exception as e:
            logger.warning(f"Liquidity sweep check failed for {pair}: {e}")

    # Crypto: RVOL >= 4x filter REMOVED (per request) — no volume filter applied here
    logger.info(
        f"Crypto scan complete: {len(matches)} strategy matches (no RVOL filter), "
        f"{len(liquidity_alerts)} liquidity sweep alerts, out of {len(pairs)} scanned"
    )
    return matches, liquidity_alerts


def main():
    all_messages = []

    if is_nse_market_hours():
        nse_matches, nse_liq = scan_nse()
        msg = format_match_message("NSE", nse_matches, liquidity_alerts=nse_liq)
        if msg:
            all_messages.append(msg)
    else:
        logger.info("Outside NSE market hours (9:15-15:30 IST, Mon-Fri) — skipping NSE scan")

    crypto_matches, crypto_liq = scan_crypto()
    msg = format_match_message("Crypto", crypto_matches, liquidity_alerts=crypto_liq)
    if msg:
        all_messages.append(msg)

    if all_messages:
        send_telegram_message("\n\n".join(all_messages))
        logger.info("Sent Telegram notification")
    else:
        logger.info("No matches this run — no notification sent")


if __name__ == "__main__":
    main()
