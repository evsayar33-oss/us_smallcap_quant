"""Free market & fundamental data — US small-cap port (NYSE/NASDAQ/AMEX, $250M-$6B).

* TradingView public scanner: end-of-day snapshot + current fundamentals.
  Field names are DISCOVERED (each optional field is probed once a week), so a
  renamed/removed TradingView field can never break the daily run.
* Yahoo Finance (yfinance): split/dividend-adjusted daily history used for the
  price factors, the 12-month labels and the index (IWM). The same function
  feeds the live engine and the backtest.
"""
from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

import config as C

TRADINGVIEW_URL = f"https://scanner.tradingview.com/{C.MARKET}/scan"
UNIVERSE_FILTER = [
    {"left": "exchange", "operation": "in_range", "right": C.EXCHANGES},
    {"left": "market_cap_basic", "operation": "in_range", "right": [C.MCAP_MIN_USD, C.MCAP_MAX_USD]},
    {"left": "close", "operation": "egreater", "right": C.MIN_PRICE_USD},
]
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

CORE = {
    "name": "ticker", "open": "open", "high": "high", "low": "low", "close": "close",
    "volume": "volume", "change": "change_pct", "Value.Traded": "value_traded",
}
# concept -> candidate TradingView fields (first valid one is used)
OPTIONAL = {
    "market_cap": ["market_cap_basic"],
    "sector": ["sector"],
    "roe": ["return_on_equity_fq", "return_on_equity"],
    "pe": ["price_earnings_ttm"],
    "pb": ["price_book_fq", "price_book_ratio"],
    "ps": ["price_sales_current", "price_sales_ratio"],
    "net_income": ["net_income_ttm", "net_income"],
    "revenue": ["total_revenue_ttm", "total_revenue"],
    "rev_growth": ["total_revenue_yoy_growth_ttm", "total_revenue_yoy_growth_fy"],
    "op_margin": ["operating_margin_ttm", "operating_margin"],
    "debt_to_equity": ["debt_to_equity_fq", "debt_to_equity"],
    "div_yield": ["dividends_yield_current", "dividend_yield_recent", "dividends_yield"],
    "shares": ["total_shares_outstanding_fundamental", "total_shares_outstanding"],
}
TEXT_FIELDS = {"ticker", "sector"}


def _num(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def _scan(columns: List[str], limit: int, extra_filter: Optional[list] = None) -> list:
    payload = {
        "filter": [{"left": "type", "operation": "equal", "right": "stock"}] + UNIVERSE_FILTER + (extra_filter or []),
        "columns": columns,
        "sort": {"sortBy": "Value.Traded", "sortOrder": "desc"},
        "range": [0, limit],
    }
    r = requests.post(TRADINGVIEW_URL, json=payload, headers=HEADERS, timeout=25)
    r.raise_for_status()
    return r.json().get("data", [])


def discover_fields(state: Dict, force: bool = False) -> Dict[str, str]:
    cache = state.setdefault("tv_fields", {})
    ts = cache.get("checked")
    if not force and ts and (pd.Timestamp.now() - pd.Timestamp(ts)).days < 7 and cache.get("map"):
        return cache["map"]
    found = {}
    for concept, cands in OPTIONAL.items():
        for f in cands:
            try:
                rows = _scan(["name", f], 3)
                if rows:
                    found[concept] = f
                    break
            except Exception:
                continue
            finally:
                time.sleep(0.15)
    cache["map"] = found
    cache["checked"] = datetime.utcnow().isoformat()
    return found


def list_all_tickers(limit: int = 1000) -> List[str]:
    """Every US small cap ($250M-$6B, NYSE/NASDAQ/AMEX, price >= $3) TradingView lists today."""
    rows = _scan(["name", "Value.Traded"], limit)
    out = []
    for it in rows:
        d = it.get("d", [])
        if d and isinstance(d[0], str) and re.fullmatch(r"[A-Z][A-Z0-9.]{0,6}", d[0].upper()):
            out.append(d[0].upper())
    return list(dict.fromkeys(out))


def fetch_snapshot(state: Dict, limit: int = C.SCAN_LIMIT) -> Tuple[pd.DataFrame, Dict]:
    meta = {"source": "tradingview"}
    try:
        fmap = discover_fields(state)
    except Exception as exc:
        fmap = {}
        meta["discover_error"] = str(exc)[:150]
    cols = list(CORE.keys()) + list(fmap.values())
    try:
        rows = _scan(cols, limit, [{"left": "Value.Traded", "operation": "greater", "right": 1_000_000}])
    except Exception as exc:
        try:   # field set may have changed since discovery -> core only, re-discover next run
            rows = _scan(list(CORE.keys()), limit, [{"left": "Value.Traded", "operation": "greater", "right": 1_000_000}])
            cols = list(CORE.keys())
            state.get("tv_fields", {}).pop("checked", None)
            meta["fallback"] = f"core_only: {str(exc)[:120]}"
        except Exception as exc2:
            meta["error"] = str(exc2)[:200]
            return pd.DataFrame(), meta
    inv = {**CORE, **{v: k for k, v in fmap.items()}}
    recs = []
    for it in rows:
        d = it.get("d", [])
        if len(d) != len(cols):
            continue
        rec = {}
        for c, v in zip(cols, d):
            key = inv.get(c, c)
            rec[key] = v if key in TEXT_FIELDS else _num(v)
        recs.append(rec)
    df = pd.DataFrame(recs)
    if df.empty:
        return df, meta
    for concept in OPTIONAL:
        if concept not in df.columns:
            df[concept] = None if concept in TEXT_FIELDS else np.nan
    df["ticker"] = df["ticker"].astype(str)
    df["tarih"] = pd.Timestamp.now(tz=C.MARKET_TZ).normalize().tz_localize(None)
    meta["fields"] = fmap
    meta["rows"] = int(len(df))
    return df, meta


# ------------------------------------------------------------------ yfinance history
def yf_symbol(t: str) -> str:
    return str(t).upper().replace(".", "-")          # BRK.B -> BRK-B


def download_history(tickers: List[str], start: str, end: Optional[str] = None,
                     min_rows: int = 60, chunk_size: int = 50) -> Dict[str, pd.DataFrame]:
    """Adjusted daily OHLCV per ticker. Yahoo has rate-limited aggressively since 2025, so:
    small chunks, pauses, exponential backoff, and a final one-by-one retry for misses."""
    import yfinance as yf
    out: Dict[str, pd.DataFrame] = {}
    back = {yf_symbol(t): t for t in tickers}
    syms = list(back)

    def _take(raw, chunk):
        for s_ in chunk:
            try:
                g = raw[s_] if isinstance(raw.columns, pd.MultiIndex) else raw
                g = g.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
                g = g[g["close"] > 0]
                g.index = pd.to_datetime(g.index).tz_localize(None).normalize()
                if len(g) >= min_rows:
                    out[back[s_]] = g
            except Exception:
                continue

    for i in range(0, len(syms), chunk_size):
        chunk = syms[i:i + chunk_size]
        for attempt in range(4):
            try:
                raw = yf.download(chunk, start=start, end=end, interval="1d", auto_adjust=True,
                                  group_by="ticker", progress=False, threads=False)
                if raw is not None and not raw.empty:
                    _take(raw, chunk)
                break
            except Exception as exc:
                wait = 60 * (2 ** attempt) if "rate" in str(exc).lower() or "too many" in str(exc).lower() else 5 * (attempt + 1)
                print(f"⚠️ yfinance hata (deneme {attempt + 1}, {wait}s bekleme): {str(exc)[:100]}")
                time.sleep(wait)
        time.sleep(2.0)
    missing = [s_ for s_ in syms if back[s_] not in out]
    if missing and len(missing) <= 300:
        for s_ in missing:
            try:
                raw = yf.download(s_, start=start, end=end, interval="1d", auto_adjust=True, progress=False, threads=False)
                if raw is not None and not raw.empty:
                    if isinstance(raw.columns, pd.MultiIndex):
                        raw.columns = [c[0] for c in raw.columns]
                    g = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
                    g = g[g["close"] > 0]
                    g.index = pd.to_datetime(g.index).tz_localize(None).normalize()
                    if len(g) >= min_rows:
                        out[back[s_]] = g
            except Exception:
                pass
            time.sleep(0.8)
    print(f"📥 fiyat geçmişi: {len(out)}/{len(tickers)} hisse")
    return out


def download_index(start: str, end: Optional[str] = None) -> pd.Series:
    import yfinance as yf
    raw = yf.download(C.REGIME_TICKER_INDEX, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        return pd.Series(dtype=float)
    s = raw["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s.dropna()
