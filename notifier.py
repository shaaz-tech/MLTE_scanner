"""
notifier.py
-----------
Formats scan results into Telegram messages and sends them.

Requires two environment variables:
    TELEGRAM_BOT_TOKEN  - your bot's token from BotFather
    TELEGRAM_CHAT_ID    - the chat/channel/group ID to post alerts to

Set these as env vars (or GitHub Actions secrets) — never hardcode them here.
"""

from __future__ import annotations

import logging
import os

import requests

logger = logging.getLogger("notifier")

TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"


def send_telegram_message(text: str) -> bool:
    """Sends a message to the configured Telegram chat. Returns True on success."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        logger.error("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set — cannot send message")
        return False

    url = TELEGRAM_API_URL.format(token=token)
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        resp = requests.post(url, data=payload, timeout=15)
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
        return False


def _format_strategy_match(m: dict) -> str:
    direction = "LONG" if m.get("match_long") else "SHORT"
    arrow = "🟢" if direction == "LONG" else "🔴"
    close = m.get("close")
    close_str = f"{close:.4f}" if isinstance(close, (int, float)) else "n/a"
    return f"{arrow} <b>{m['symbol']}</b> — {direction} | RVOL: {m['rvol']}x | Close: {close_str}"


def _format_liquidity_alert(a: dict) -> str:
    kind = a.get("liquidity_alert", "")
    icon = "🟥" if kind == "BSL" else "🟩"
    label = "BSL (Buy-Side Liquidity / Retail Stops Above)" if kind == "BSL" else "SSL (Sell-Side Liquidity / Retail Stops Below)"
    rng_pct = a.get("rng_pct")
    pivot_price = a.get("pivot_price")
    rng_str = f"{rng_pct}%" if rng_pct is not None else "n/a"
    price_str = f"{pivot_price:.4f}" if isinstance(pivot_price, (int, float)) else "n/a"
    return f"{icon} <b>{a['symbol']}</b> — {label}\n    Rng: {rng_str} | Pivot: {price_str}"


def format_match_message(market_label: str, matches: list[dict], liquidity_alerts: list[dict] | None = None) -> str | None:
    """
    Builds a single Telegram message block for one market (NSE or Crypto),
    combining main-strategy matches and SSL/BSL liquidity sweep alerts.

    Returns None if there is nothing to report (so scanner.py can skip sending).
    """
    liquidity_alerts = liquidity_alerts or []

    if not matches and not liquidity_alerts:
        return None

    lines = [f"<b>=== {market_label} ===</b>"]

    if matches:
        lines.append(f"\n<b>Strategy Matches ({len(matches)})</b>")
        for m in matches:
            lines.append(_format_strategy_match(m))

    if liquidity_alerts:
        lines.append(f"\n<b>Liquidity Sweep Alerts ({len(liquidity_alerts)})</b>")
        for a in liquidity_alerts:
            lines.append(_format_liquidity_alert(a))

    return "\n".join(lines)
