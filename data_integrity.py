"""Data integrity: validation, stale-data detection and corporate-action adjustment.

BIST has frequent bonus issues (bedelsiz) and rights issues. Raw snapshot
prices therefore contain fake -30%..-80% jumps. Because the BIST daily price
limit is +-10%, a raw move beyond the limit between two *adjacent* sessions is
either a corporate action or a bad print. TradingView's `change` field is
computed against the adjusted previous close, so the adjustment ratio is:

    ratio = (close_t / (1 + change_t/100)) / close_{t-1}

All earlier prices of that ticker are multiplied by `ratio`. When the two
snapshots are not adjacent sessions (a run was missed) the ratio cannot be
verified; such breaks invalidate any label that spans them instead of guessing.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

import calendar_tr as cal
import config as C

REQUIRED = ["ticker", "close", "open", "high", "low", "volume", "change_pct", "value_traded"]
RECOMMENDED = ["market_cap", "sector", "roe", "pb", "pe"]


def validate_market_frame(df: pd.DataFrame, min_rows: int = C.MIN_CROSS_SECTION) -> Tuple[pd.DataFrame, Dict]:
    if df is None or df.empty:
        return pd.DataFrame(), {"ok": False, "reason": "empty_market_frame", "score": 0.0}
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return pd.DataFrame(), {"ok": False, "reason": "missing_required:" + ",".join(missing), "score": 0.0}

    out = df.copy()
    for c in REQUIRED + RECOMMENDED:
        if c in out.columns and c not in ("ticker", "sector"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.drop_duplicates(subset=["ticker"], keep="last")

    o, h, l, c = (out[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    bad = ~np.isfinite(o) | ~np.isfinite(h) | ~np.isfinite(l) | ~np.isfinite(c)
    with np.errstate(invalid="ignore"):
        bad |= (o <= 0) | (h <= 0) | (l <= 0) | (c <= 0) | (h < l)
        bad |= (h < np.maximum(o, c) * 0.999) | (l > np.minimum(o, c) * 1.001)
        bad |= ~np.isfinite(out["change_pct"].to_numpy(dtype=float))
        bad |= out["value_traded"].to_numpy(dtype=float) < 0
    invalid = int(bad.sum())
    out = out.loc[~bad].copy()
    if out.empty:
        return pd.DataFrame(), {"ok": False, "reason": "all_rows_invalid", "score": 0.0, "invalid_rows": invalid}

    rec_present = [k for k in RECOMMENDED if k in out.columns]
    rec_missing_rate = float(np.mean([out[k].isna().mean() for k in rec_present])) if rec_present else 1.0
    penalty = 0.0
    if len(out) < min_rows:
        penalty += 40.0
    penalty += min(30.0, rec_missing_rate * 60.0)
    penalty += min(15.0, invalid / max(len(out) + invalid, 1) * 100.0)
    score = round(max(0.0, 100.0 - penalty), 1)
    summary = {
        "ok": bool(len(out) >= min_rows and score >= 60.0),
        "score": score,
        "rows": int(len(out)),
        "invalid_rows": invalid,
        "recommended_missing_rate": round(rec_missing_rate, 4),
    }
    if not summary["ok"]:
        summary["reason"] = "quality_below_threshold"
    return out.reset_index(drop=True), summary


def is_stale(today: pd.DataFrame, last: pd.DataFrame) -> bool:
    """True when today's snapshot is a copy of the previous one (holiday / feed freeze)."""
    if today is None or today.empty or last is None or last.empty:
        return False
    m = today[["ticker", "close", "volume"]].merge(
        last[["ticker", "close", "volume"]], on="ticker", suffixes=("", "_prev"))
    if len(m) < C.MIN_CROSS_SECTION:
        return False
    same = (np.isclose(m["close"], m["close_prev"]) & np.isclose(m["volume"], m["volume_prev"])).mean()
    return bool(same >= 0.90)


def build_price_panel(snapshots: pd.DataFrame) -> Dict:
    """Wide, corporate-action-adjusted OHLC panel (dates x tickers) from EOD snapshots."""
    if snapshots is None or snapshots.empty:
        return {}
    s = snapshots.copy()
    s["tarih"] = pd.to_datetime(s["tarih"]).dt.normalize()
    s = s.drop_duplicates(subset=["tarih", "ticker"], keep="last")

    def wide(col):
        return s.pivot(index="tarih", columns="ticker", values=col).sort_index().astype(float)

    O, H, L, Cl = wide("open"), wide("high"), wide("low"), wide("close")
    CH = wide("change_pct")
    ATR = wide("atr") if "atr" in s.columns else Cl * np.nan
    dates = list(Cl.index)

    adjacent = np.array([False] + [cal.sessions_after_count(dates[i - 1], dates[i]) == 1
                                   for i in range(1, len(dates))])
    prev_close = Cl.shift(1)
    raw_move = (Cl / prev_close - 1.0) * 100.0
    implied_prev = Cl / (1.0 + CH / 100.0)
    ratio = implied_prev / prev_close

    beyond = raw_move.abs() > C.CORP_ACTION_MOVE_PCT
    adj_mask = np.repeat(adjacent[:, None], Cl.shape[1], axis=1)
    verified_ca = beyond & adj_mask & ((ratio - 1.0).abs() > 0.03) & np.isfinite(ratio) & (ratio > 0)
    # unverifiable big jumps across missing sessions -> break (labels invalidated)
    gap_sessions = np.array([1] + [max(1, cal.sessions_after_count(dates[i - 1], dates[i]))
                                   for i in range(1, len(dates))], dtype=float)
    lim = ((1 + C.CORP_ACTION_MOVE_PCT / 100.0) ** gap_sessions - 1.0) * 100.0
    unverified = (~adj_mask) & (raw_move.abs().to_numpy() > lim[:, None]) & np.isfinite(raw_move.to_numpy())
    bad_print = beyond & adj_mask & ~verified_ca

    step = ratio.where(verified_ca, 1.0).fillna(1.0)
    # factor for date t = product of step ratios at dates > t
    rev_cum = step.iloc[::-1].cumprod().iloc[::-1]
    factor = rev_cum.shift(-1).fillna(1.0)

    breaks = pd.DataFrame(unverified, index=Cl.index, columns=Cl.columns) | bad_print

    events = []
    for (d, t) in zip(*np.where(verified_ca.to_numpy())):
        events.append({"tarih": str(dates[d].date()), "ticker": Cl.columns[t],
                       "ratio": round(float(ratio.iat[d, t]), 6)})

    return {
        "dates": dates,
        "open": O * factor,
        "high": H * factor,
        "low": L * factor,
        "close": Cl * factor,
        "atr_pct": (ATR / Cl) * 100.0,
        "breaks": breaks.fillna(False).astype(bool),
        "ca_events": events,
        "adjacent": adjacent,
    }


def today_ca_ratio(today: pd.DataFrame, last: pd.DataFrame, last_date, today_date) -> Dict[str, float]:
    """Corporate-action ratios detected today (used to rescale open ledger positions)."""
    if today is None or today.empty or last is None or last.empty:
        return {}
    if cal.sessions_after_count(last_date, today_date) != 1:
        return {}
    m = today[["ticker", "close", "change_pct"]].merge(last[["ticker", "close"]], on="ticker",
                                                      suffixes=("", "_prev"))
    raw = (m["close"] / m["close_prev"] - 1.0) * 100.0
    r = (m["close"] / (1.0 + m["change_pct"] / 100.0)) / m["close_prev"]
    sel = (raw.abs() > C.CORP_ACTION_MOVE_PCT) & ((r - 1.0).abs() > 0.03) & np.isfinite(r) & (r > 0)
    return {str(t): float(x) for t, x in zip(m.loc[sel, "ticker"], r[sel])}
