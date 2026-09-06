"""
notifier.py
-----------
Sends scan results to you via Telegram (free, instant, works from any
server/CI runner with just an HTTP POST -- no email server setup needed).

Setup (one-time, ~2 minutes):
  1. In Telegram, message @BotFather -> /newbot -> follow prompts -> copy
     the bot token it gives you.
  2. Message your new bot anything (e.g. "hi") so it can see your chat.
  3. Visit https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates in a browser
     and find your numeric "chat":{"id": ...} -- that's your TELEGRAM_CHAT_ID.
  4. Put both values in GitHub repo Settings -> Secrets and variables ->
     Actions -> New repository secret (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID).
"""

from __future__ import annotations

import os
import logging
import requests

logger = logging.getLogger("scanner.notifier")

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


def send_telegram_message(text: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        logger.warning("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set — printing to console instead")
        print(text)
        return False

    # Telegram caps messages at 4096 chars; split into chunks if needed.
    max_len = 4000
    chunks = [text[i:i + max_len] for i in range(0, len(text), max_len)] or [text]

    ok = True
    for chunk in chunks:
        try:
            resp = requests.post(
                TELEGRAM_API.format(token=token),
                data={"chat_id": chat_id, "text": chunk, "parse_mode": "HTML"},
                timeout=10,
            )
            if resp.status_code != 200:
                logger.warning(f"Telegram send failed: {resp.status_code} {resp.text}")
                ok = False
        except requests.RequestException as e:
            logger.warning(f"Telegram send error: {e}")
            ok = False
    return ok


def format_match_message(market_label: str, matches: list[dict]) -> str:
    if not matches:
        return ""
    lines = [f"<b>{market_label} — {len(matches)} match(es)</b>"]
    for m in matches:
        direction = "🟢 LONG" if m["match_long"] else "🔴 SHORT"
        lines.append(
            f"{m['symbol']}: {direction} | {m['tag']} | price {m['price']:.2f} | "
            f"RSI {m['rsi']:.1f} | RVOL {m['rvol']:.2f}x"
        )
    return "\n".join(lines)
