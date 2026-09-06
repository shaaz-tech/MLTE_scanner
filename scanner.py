"""
scanner.py
----------
Main entry point. Scans the FULL NSE equity list + a broad crypto/meme-coin
universe for the same conditions used in the TradingView system (Turtle
breakout, EMA9/21 cross, SuperTrend+RSI with a volume spike), and sends a
Telegram message listing every symbol that matched.

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
from indicators import evaluate_conditions
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
}

NSE_INTERVAL = "15m"      # yfinance intraday interval
CRYPTO_INTERVAL = "15m"   # Kraken kline interval
CRYPTO_TOP_N = 200        # top coins by market cap to include
CRYPTO_MAX_MEME = 150     # cap on meme-token category coins


def is_nse_market_hours() -> bool:
    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
    if now_ist.weekday() >= 5:  # Sat=5, Sun=6
        return False
    open_time = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    close_time = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
    return open_time <= now_ist <= close_time


def scan_nse() -> list[dict]:
    logger.info("Starting NSE scan...")
    symbols = ds.get_nse_symbol_list()
    candles = ds.get_nse_candles_batch(symbols, interval=NSE_INTERVAL)

    matches = []
    for symbol, df in candles.items():
        try:
            result = evaluate_conditions(df, CONFIG)
        except Exception as e:
            logger.warning(f"Condition eval failed for {symbol}: {e}")
            continue
        if result.get("match_long") or result.get("match_short"):
            matches.append({"symbol": symbol, **result})

    logger.info(f"NSE scan complete: {len(matches)} matches out of {len(candles)} scanned")
    return matches


def scan_crypto() -> list[dict]:
    logger.info("Starting crypto scan...")
    coin_symbols = ds.get_crypto_universe(top_n=CRYPTO_TOP_N, max_meme_coins=CRYPTO_MAX_MEME)
    pairs = ds.match_to_kraken_usd_pairs(coin_symbols)

    matches = []
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

    logger.info(f"Crypto scan complete: {len(matches)} matches out of {len(pairs)} scanned")
    return matches


def main():
    all_messages = []

    if is_nse_market_hours():
        nse_matches = scan_nse()
        msg = format_match_message("NSE", nse_matches)
        if msg:
            all_messages.append(msg)
    else:
        logger.info("Outside NSE market hours (9:15-15:30 IST, Mon-Fri) — skipping NSE scan")

    crypto_matches = scan_crypto()
    msg = format_match_message("Crypto", crypto_matches)
    if msg:
        all_messages.append(msg)

    if all_messages:
        send_telegram_message("\n\n".join(all_messages))
        logger.info("Sent Telegram notification")
    else:
        logger.info("No matches this run — no notification sent")


if __name__ == "__main__":
    main()
