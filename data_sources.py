"""
data_sources.py
----------------
Fetches the FULL NSE equity list and candle data (via yfinance, free, no
broker account needed) plus a broad crypto universe (top market-cap coins +
the full CoinGecko "meme-token" category, matched to Binance USDT pairs for
free candle data).

Honest limitations, stated up front:
  - NSE data through yfinance is delayed (typically ~15 min), not tick-level
    real-time. For genuinely real-time intraday data you need a broker API
    (Zerodha Kite Connect, Upstox, Fyers, Angel One) with your own
    credentials -- that's a straightforward swap-in later if you need it,
    but it requires a broker account + API subscription, which is why this
    starts with the free/no-account option.
  - Meme coins that aren't listed on Binance are skipped for candle data
    (logged, not silently dropped) since Binance's public API is free and
    reliable; CoinGecko's own OHLC endpoint could be added as a fallback
    for those if you want full coverage of very new/small meme coins later.
"""

from __future__ import annotations

import io
import time
import logging
import requests
import pandas as pd
import yfinance as yf

logger = logging.getLogger("scanner.data")

NSE_CSV_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
NSE_CSV_URL_FALLBACK = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"

COINGECKO_MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"
COINGECKO_MEME_CATEGORY = "meme-token"

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
BINANCE_EXCHANGE_INFO_URL = "https://api.binance.com/api/v3/exchangeInfo"


# ============================================================================
# NSE — full equity list + candles via yfinance
# ============================================================================
def get_nse_symbol_list() -> list[str]:
    """Returns every NSE equity symbol as plain tickers (no exchange suffix),
    e.g. ['RELIANCE', 'TCS', ...]. NSE blocks requests without browser-like
    headers and a warmed-up session cookie, so we visit the homepage first."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
        "Accept": "text/csv,application/csv,*/*",
    }
    session = requests.Session()
    session.headers.update(headers)

    try:
        session.get("https://www.nseindia.com/", timeout=10)  # warms up cookies
        time.sleep(1)
        resp = session.get(NSE_CSV_URL, timeout=15)
        resp.raise_for_status()
    except requests.RequestException:
        logger.warning("Primary NSE archive URL failed, trying fallback host")
        resp = session.get(NSE_CSV_URL_FALLBACK, timeout=15)
        resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text))
    symbols = df["SYMBOL"].dropna().astype(str).str.strip().tolist()
    logger.info(f"Fetched {len(symbols)} NSE equity symbols")
    return symbols


def get_nse_candles_batch(symbols: list[str], interval: str = "15m", period: str = "5d",
                            batch_size: int = 100) -> dict:
    """Downloads intraday candles for many NSE symbols at once via yfinance's
    batch downloader (much faster than one request per symbol). Returns
    {symbol: DataFrame} for symbols that returned usable data."""
    results = {}
    yf_tickers = [f"{s}.NS" for s in symbols]

    for i in range(0, len(yf_tickers), batch_size):
        chunk = yf_tickers[i:i + batch_size]
        try:
            data = yf.download(
                tickers=chunk, period=period, interval=interval,
                group_by="ticker", threads=True, progress=False,
            )
        except Exception as e:
            logger.warning(f"yfinance batch {i}-{i+batch_size} failed: {e}")
            continue

        for yf_ticker in chunk:
            plain_symbol = yf_ticker.replace(".NS", "")
            try:
                df = data[yf_ticker] if len(chunk) > 1 else data
                df = df.dropna(how="all")
                if len(df) >= 30:
                    results[plain_symbol] = df
            except (KeyError, TypeError):
                continue
    logger.info(f"Got usable candles for {len(results)}/{len(symbols)} NSE symbols")
    return results


# ============================================================================
# CRYPTO — top market cap + full meme-token category, matched to Binance
# ============================================================================
def get_crypto_universe(top_n: int = 200, include_meme_category: bool = True,
                          max_meme_coins: int = 150) -> list[str]:
    """Returns a de-duplicated list of coin symbols (e.g. 'BTC', 'DOGE', 'PEPE')
    combining the top N coins by market cap with the full meme-token category
    (capped at max_meme_coins to avoid scanning hundreds of dead/illiquid
    micro-cap tokens with no real trading data)."""
    symbols = set()

    try:
        resp = requests.get(COINGECKO_MARKETS_URL, params={
            "vs_currency": "usd", "order": "market_cap_desc",
            "per_page": min(top_n, 250), "page": 1,
        }, timeout=15)
        resp.raise_for_status()
        for coin in resp.json():
            symbols.add(coin["symbol"].upper())
    except requests.RequestException as e:
        logger.warning(f"CoinGecko top-market-cap fetch failed: {e}")

    if include_meme_category:
        try:
            resp = requests.get(COINGECKO_MARKETS_URL, params={
                "vs_currency": "usd", "category": COINGECKO_MEME_CATEGORY,
                "order": "market_cap_desc", "per_page": min(max_meme_coins, 250),
                "page": 1,
            }, timeout=15)
            resp.raise_for_status()
            for coin in resp.json():
                symbols.add(coin["symbol"].upper())
        except requests.RequestException as e:
            logger.warning(f"CoinGecko meme-token category fetch failed: {e}")

    logger.info(f"Crypto universe: {len(symbols)} unique symbols before Binance matching")
    return sorted(symbols)


def match_to_binance_usdt_pairs(coin_symbols: list[str]) -> list[str]:
    """Filters coin symbols down to ones that actually have a USDT pair on
    Binance (needed for free candle data). Returns e.g. ['BTCUSDT', 'DOGEUSDT']."""
    try:
        resp = requests.get(BINANCE_EXCHANGE_INFO_URL, timeout=15)
        resp.raise_for_status()
        listed = {s["symbol"] for s in resp.json()["symbols"] if s["quoteAsset"] == "USDT"}
    except requests.RequestException as e:
        logger.warning(f"Binance exchangeInfo fetch failed: {e}")
        return []

    matched, skipped = [], []
    for sym in coin_symbols:
        pair = f"{sym}USDT"
        if pair in listed:
            matched.append(pair)
        else:
            skipped.append(sym)

    if skipped:
        logger.info(f"{len(skipped)} coins have no Binance USDT pair, skipped: {skipped[:20]}{'...' if len(skipped) > 20 else ''}")
    logger.info(f"{len(matched)} coins matched to Binance USDT pairs")
    return matched


def get_binance_candles(pair: str, interval: str = "15m", limit: int = 200):
    """Fetches candles for one Binance pair. interval examples: 1m, 5m, 15m, 1h, 1d."""
    try:
        resp = requests.get(BINANCE_KLINES_URL, params={
            "symbol": pair, "interval": interval, "limit": limit,
        }, timeout=10)
        resp.raise_for_status()
        raw = resp.json()
    except requests.RequestException as e:
        logger.warning(f"Binance klines fetch failed for {pair}: {e}")
        return None

    if not raw:
        return None

    df = pd.DataFrame(raw, columns=[
        "OpenTime", "Open", "High", "Low", "Close", "Volume", "CloseTime",
        "QuoteVol", "Trades", "TakerBase", "TakerQuote", "Ignore",
    ])
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].astype(float)
    df.index = pd.to_datetime(df["OpenTime"], unit="ms")
    return df[["Open", "High", "Low", "Close", "Volume"]]
