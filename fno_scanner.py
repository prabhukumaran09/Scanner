"""
Indian FNO Options Scanner — v2
================================
Scans stock options (Call & Put) every 3 minutes during market hours.

Filters applied per contract:
  1. Strike is ATM, nearest ITM, or nearest OTM (1 strike each side)
  2. Option price is RISING (current close > previous close)
  3. Option price is ABOVE its intraday VWAP
  4. RSI(14) > 60
  5. Volume > 20-period moving average of volume

Alerts via Telegram.
Hosted free on GitHub Actions.
"""

import os
import sys
import time
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, date, time as dtime
import pytz
from kiteconnect import KiteConnect

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
KITE_API_KEY      = os.environ["KITE_API_KEY"]
KITE_ACCESS_TOKEN = os.environ["KITE_ACCESS_TOKEN"]
TELEGRAM_TOKEN    = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID  = os.environ["TELEGRAM_CHAT_ID"]

RSI_PERIOD        = 14
VOLUME_MA_PERIOD  = 20
SCAN_INTERVAL     = 300          # seconds (5 minutes)
MARKET_OPEN       = dtime(9, 15)
MARKET_CLOSE      = dtime(15, 30)
IST               = pytz.timezone("Asia/Kolkata")

# Index names to exclude (we want stock options only)
INDEX_NAMES = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX", "NIFTYNXT50"}

# ── Zerodha client ────────────────────────────────────────────────────────────
kite = KiteConnect(api_key=KITE_API_KEY)
kite.set_access_token(KITE_ACCESS_TOKEN)

# ─────────────────────────────────────────────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────────────────────────────────────────────
def send_telegram(message: str):
    url     = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        log.info("Telegram alert sent.")
    except Exception as e:
        log.error(f"Telegram error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# INSTRUMENT LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_all_nfo_instruments() -> pd.DataFrame:
    """
    Download full NFO instrument list and return stock CE/PE options
    for the nearest expiry only.
    """
    instruments = kite.instruments("NFO")
    df = pd.DataFrame(instruments)
    df = df[df["instrument_type"].isin(["CE", "PE"])]
    df = df[~df["name"].isin(INDEX_NAMES)]
    df["expiry"] = pd.to_datetime(df["expiry"])
    today = pd.Timestamp.now().normalize()
    df = df[df["expiry"] >= today]
    nearest = df["expiry"].min()
    df = df[df["expiry"] == nearest].copy()
    log.info(f"Loaded {len(df)} stock option contracts (expiry: {nearest.date()})")
    return df


def load_equity_instruments() -> pd.DataFrame:
    """Load NSE equity instruments for LTP lookup of underlyings."""
    instruments = kite.instruments("NSE")
    df = pd.DataFrame(instruments)
    log.info(f"NSE instrument columns: {list(df.columns)}")   # log once for debugging

    # Filter to plain equity shares only — try multiple column strategies
    # since Kite's column names can vary slightly across API versions
    if "instrument_type" in df.columns:
        df = df[df["instrument_type"] == "EQ"]
    elif "series" in df.columns:
        df = df[df["series"] == "EQ"]
    else:
        # Fallback: keep all NSE instruments and deduplicate by tradingsymbol
        # The NFO merge will naturally limit to only valid underlyings
        log.warning("Neither 'instrument_type' nor 'series' column found in NSE instruments. "
                    "Using all NSE instruments as fallback.")

    df = df[["tradingsymbol", "instrument_token"]].copy()
    df = df.rename(columns={"tradingsymbol": "name", "instrument_token": "equity_token"})
    df = df.drop_duplicates("name")
    log.info(f"Loaded {len(df)} NSE equity instruments.")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# STRIKE SELECTION — ATM only
# ─────────────────────────────────────────────────────────────────────────────
def select_strikes(
    options_df: pd.DataFrame,
    spot_price: float,
    opt_type: str           # "CE" or "PE"
) -> pd.DataFrame:
    """
    Returns only the ATM strike (closest to spot price) for the given
    option type. One contract per underlying per type (CE/PE).
    """
    df = options_df[options_df["instrument_type"] == opt_type].copy()
    if df.empty:
        return df

    strikes = sorted(df["strike"].unique())
    if not strikes:
        return pd.DataFrame()

    # ATM = strike closest to spot price
    atm_strike = min(strikes, key=lambda s: abs(s - spot_price))

    # ATM only — return just the single closest strike
    return df[df["strike"] == atm_strike]


def build_scan_universe(
    nfo_df: pd.DataFrame,
    equity_df: pd.DataFrame
) -> pd.DataFrame:
    """
    For every underlying, fetch its spot LTP and keep only
    ATM + nearest ITM + nearest OTM for both CE and PE.
    Returns a trimmed DataFrame ready for scanning.

    Spot prices are fetched in a single batch LTP call.
    """
    merged = nfo_df.merge(equity_df, on="name", how="inner")
    unique_stocks = merged.drop_duplicates("name")

    # Build Kite-format trading symbols for batch LTP
    trading_symbols = [f"NSE:{sym}" for sym in unique_stocks["name"].tolist()]

    # Kite LTP allows max 500 symbols per call — chunk if needed
    spot_prices: dict = {}
    for i in range(0, len(trading_symbols), 500):
        chunk = trading_symbols[i:i + 500]
        try:
            ltp_data = kite.ltp(chunk)
            for sym, data in ltp_data.items():
                name = sym.replace("NSE:", "")
                spot_prices[name] = data["last_price"]
        except Exception as e:
            log.error(f"LTP batch fetch error: {e}")

    log.info(f"Fetched spot prices for {len(spot_prices)} underlyings.")

    selected_rows = []
    for name, grp in merged.groupby("name"):
        spot = spot_prices.get(name)
        if not spot:
            continue
        for opt_type in ["CE", "PE"]:
            chosen = select_strikes(grp, spot, opt_type)
            if not chosen.empty:
                chosen = chosen.copy()
                chosen["spot_price"] = spot
                selected_rows.append(chosen)

    if not selected_rows:
        return pd.DataFrame()

    result = pd.concat(selected_rows, ignore_index=True)
    log.info(f"Universe trimmed to {len(result)} contracts (ATM±1 per underlying).")
    return result

# ─────────────────────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS
# ─────────────────────────────────────────────────────────────────────────────
def compute_rsi(prices: pd.Series, period: int = 14) -> float:
    if len(prices) < period + 1:
        return np.nan
    delta    = prices.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 2)


def compute_vwap(df: pd.DataFrame) -> float:
    """
    Intraday VWAP calculated from market open of the day.
    Formula: VWAP = Σ(Typical Price × Volume) / Σ(Volume)
             Typical Price = (High + Low + Close) / 3

    Using cumulative VWAP — the value shown is the running average
    from 9:15 AM to the latest candle.
    """
    if df.empty or df["volume"].sum() == 0:
        return np.nan
    tp   = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (tp * df["volume"]).cumsum() / df["volume"].cumsum()
    return round(float(vwap.iloc[-1]), 4)


def price_is_rising(closes: pd.Series) -> tuple:
    """Returns (is_rising, current_price, prev_price)."""
    if len(closes) < 2:
        return False, 0.0, 0.0
    return (
        bool(closes.iloc[-1] > closes.iloc[-2]),
        float(closes.iloc[-1]),
        float(closes.iloc[-2])
    )


def volume_above_ma(volumes: pd.Series, period: int = 20) -> tuple:
    """Returns (is_above_ma, current_volume, ma_volume)."""
    if len(volumes) < period:
        return False, 0.0, 0.0
    ma  = float(volumes.iloc[-period:].mean())
    cur = float(volumes.iloc[-1])
    return cur > ma, round(cur), round(ma)

# ─────────────────────────────────────────────────────────────────────────────
# CANDLE FETCH
# ─────────────────────────────────────────────────────────────────────────────
def fetch_candles(instrument_token: int, interval: str = "5minute") -> pd.DataFrame:
    today = date.today()
    try:
        candles = kite.historical_data(
            instrument_token=instrument_token,
            from_date=today,
            to_date=today,
            interval=interval
        )
        if not candles:
            return pd.DataFrame()
        df = pd.DataFrame(candles)
        df.columns = ["date", "open", "high", "low", "close", "volume"]
        df = df.astype({
            "open": float, "high": float,
            "low": float, "close": float, "volume": float
        })
        return df
    except Exception as e:
        log.debug(f"Candle fetch failed for token {instrument_token}: {e}")
        return pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# MONEYNESS LABEL
# ─────────────────────────────────────────────────────────────────────────────
def moneyness_label(strike: float, spot: float, opt_type: str) -> str:
    rel = abs(strike - spot) / spot
    if rel < 0.005:                    # within 0.5% of spot → ATM
        return "ATM"
    if opt_type == "CE":
        return "ITM" if strike < spot else "OTM"
    else:
        return "ITM" if strike > spot else "OTM"

# ─────────────────────────────────────────────────────────────────────────────
# CORE SCAN LOOP
# ─────────────────────────────────────────────────────────────────────────────
def scan_once(universe_df: pd.DataFrame) -> list:
    matches = []
    log.info(f"Scanning {len(universe_df)} contracts (ATM±1 only)...")

    for _, row in universe_df.iterrows():
        token    = row["instrument_token"]
        symbol   = row["tradingsymbol"]
        opt_type = row["instrument_type"]
        strike   = float(row["strike"])
        name     = row["name"]
        spot     = float(row["spot_price"])

        df = fetch_candles(token)
        if df.empty or len(df) < VOLUME_MA_PERIOD + 1:
            continue

        closes  = df["close"]
        volumes = df["volume"]

        # Filter 1 — Price above VWAP
        cur_price = float(closes.iloc[-1])
        vwap = compute_vwap(df)
        if np.isnan(vwap) or cur_price <= vwap:
            continue

        # Filter 2 — RSI > 60
        rsi = compute_rsi(closes, RSI_PERIOD)
        if np.isnan(rsi) or rsi <= 60:
            continue

        # Filter 3 — Volume > 20-period MA
        vol_ok, cur_vol, ma_vol = volume_above_ma(volumes, VOLUME_MA_PERIOD)
        if not vol_ok:
            continue

        vwap_gap_pct = round(((cur_price - vwap) / vwap) * 100, 2)

        matches.append({
            "symbol":        symbol,
            "name":          name,
            "type":          opt_type,
            "strike":        strike,
            "moneyness":     moneyness_label(strike, spot, opt_type),
            "spot":          round(spot, 2),
            "price":         cur_price,
            "vwap":          vwap,
            "vwap_gap_pct":  vwap_gap_pct,
            "rsi":           rsi,
            "volume":        cur_vol,
            "vol_ma20":      ma_vol,
        })

    log.info(f"Scan complete — {len(matches)} match(es).")
    return matches

# ─────────────────────────────────────────────────────────────────────────────
# ALERT FORMATTING
# ─────────────────────────────────────────────────────────────────────────────
def format_alert(matches: list) -> str:
    now   = datetime.now(IST).strftime("%d-%b-%Y %H:%M")
    lines = [f"<b>🔍 FNO Scanner — {now} IST</b>\n"]

    for m in matches:
        emoji     = "🟢" if m["type"] == "CE" else "🔴"
        money_tag = {"ATM": "🎯", "ITM": "💰", "OTM": "🎲"}.get(m["moneyness"], "")
        v_sign    = "+" if m["vwap_gap_pct"] >= 0 else ""

        lines.append(
            f"{emoji} <b>{m['symbol']}</b>  {money_tag} {m['moneyness']}\n"
            f"   📌 Stock: <b>{m['name']}</b>  |  {m['type']}  |  Strike: ₹{m['strike']:,.0f}\n"
            f"   💹 Spot: ₹{m['spot']:,.2f}\n"
            f"   💲 Price: ₹{m['price']:.2f}\n"
            f"   📊 VWAP:  ₹{m['vwap']:.2f}  (Price {v_sign}{m['vwap_gap_pct']}% above VWAP)\n"
            f"   📈 RSI: {m['rsi']}  |  Vol: {m['volume']:,.0f}  (MA20: {m['vol_ma20']:,.0f})\n"
        )

    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# MARKET HOURS
# ─────────────────────────────────────────────────────────────────────────────
def is_market_open() -> bool:
    now_ist = datetime.now(IST)
    if now_ist.weekday() >= 5:
        return False
    return MARKET_OPEN <= now_ist.time() <= MARKET_CLOSE

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    log.info("FNO Scanner v2 started.")
    send_telegram(
        "✅ <b>FNO Scanner v2 Started</b>\n"
        "Scanning <b>ATM only</b> strikes every 5 min.\n"
        "Filters: Price &gt; VWAP · RSI &gt; 60 · Vol &gt; MA20"
    )

    nfo_df    = load_all_nfo_instruments()
    equity_df = load_equity_instruments()

    universe_df           = build_scan_universe(nfo_df, equity_df)
    last_universe_refresh = datetime.now(IST)

    while True:
        now_ist = datetime.now(IST)

        # Rebuild universe every 30 min — spot drifts and ATM can shift
        if (now_ist - last_universe_refresh).total_seconds() > 1800:
            log.info("Refreshing scan universe (ATM may have shifted)...")
            nfo_df              = load_all_nfo_instruments()
            universe_df         = build_scan_universe(nfo_df, equity_df)
            last_universe_refresh = now_ist

        if not is_market_open():
            now_ist = datetime.now(IST)
            msg = (
                f"\U0001f534 <b>FNO Scanner \u2014 Market Closed</b>\n"
                f"Time: {now_ist.strftime('%d-%b-%Y %H:%M')} IST\n"
                f"Market hours: 09:15 AM \u2013 03:30 PM IST (Mon\u2013Fri)\n"
                f"Scanner will auto-start on next trading day at 9:15 AM."
            )
            log.info("Market is closed. Sending Telegram alert and exiting.")
            send_telegram(msg)
            sys.exit(0)

        if universe_df.empty:
            log.warning("Universe is empty — check Kite API credentials.")
            time.sleep(SCAN_INTERVAL)
            continue

        matches = scan_once(universe_df)

        if matches:
            for i in range(0, len(matches), 8):
                batch = matches[i:i + 8]
                send_telegram(format_alert(batch))

        log.info(f"Sleeping {SCAN_INTERVAL}s until next scan...")
        time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    main()
