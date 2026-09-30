"""Benchmarks for a USD investor (US small-cap port). Internal column names are kept from the
BIST engine so every downstream module works unchanged:
  usdtry   -> constant 1.0 (no FX leg for a USD investor; the "usd" component is disabled)
  gold_try -> gold in USD per gram (GC=F / 31.1035; the scale does not affect returns)
  xu100    -> IWM (Russell 2000 ETF, total-return adjusted close)
Hurdle components: US CPI (CPIAUCSL), gold (USD), 3M T-bill (DTB3). Sum (+margin) is the displayed
target, the strongest single one is the floor. Forward-looking: CPI = trailing blend, gold = CPI
(no own-drift forecast), T-bill = current yield.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

import config as C

GRAM_PER_OZ = 31.1035
NAMES_TR = {"cpi": "ABD TÜFE", "usd": "Dolar", "gold": "Altın", "deposit": "Hazine bonosu", "xu100": "Russell 2000"}


def download_benchmarks(start: str = "2011-01-01", end: Optional[str] = None) -> pd.DataFrame:
    import yfinance as yf
    raw = yf.download(["GC=F", C.REGIME_TICKER_INDEX], start=start, end=end, interval="1d",
                      auto_adjust=True, progress=False, threads=False, group_by="ticker")
    if raw is None or raw.empty:
        raise RuntimeError("benchmark download empty")

    def col(sym):
        try:
            s = raw[sym]["Close"].dropna()
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            return s
        except Exception:
            return pd.Series(dtype=float)
    idx = col(C.REGIME_TICKER_INDEX)
    one = pd.Series(1.0, index=idx.index.union(col("GC=F").index))
    return assemble(one, col("GC=F"), idx)


def assemble(usdtry: pd.Series, gold_usd: Optional[pd.Series], xu100: Optional[pd.Series]) -> pd.DataFrame:
    idx = usdtry.index
    for s in (gold_usd, xu100):
        if s is not None and len(s):
            idx = idx.union(s.index)
    df = pd.DataFrame(index=idx.sort_values())
    df["usdtry"] = usdtry.reindex(df.index).ffill()
    df["gold_try"] = (gold_usd.reindex(df.index).ffill() * df["usdtry"] / GRAM_PER_OZ) if gold_usd is not None and len(gold_usd) else np.nan
    df["xu100"] = xu100.reindex(df.index).ffill() if xu100 is not None and len(xu100) else np.nan
    return df.dropna(subset=["usdtry"])


def _at(s: pd.Series, d) -> float:
    if s is None or not len(s):
        return np.nan
    v = s.asof(pd.Timestamp(d)) if pd.Timestamp(d) >= s.index[0] else np.nan
    return float(v) if v is not None and np.isfinite(v) else np.nan


def _deposit_growth(crate: Optional[pd.Series], a, b) -> float:
    """Compounded net money-market growth between dates a and b (monthly net rates)."""
    from inflation import cash_yield_at
    if crate is None or not len(crate):
        return np.nan
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    g = 1.0
    cur = a
    while cur < b:
        nxt = min(cur + pd.offsets.MonthBegin(1), b)
        y = cash_yield_at(crate, cur)
        g *= (1 + y / 100.0) ** ((nxt - cur).days / 365.25)
        cur = nxt
    return g


def window_returns(bm: Optional[pd.DataFrame], cpi: Optional[pd.Series], crate: Optional[pd.Series], a, b,
                   approx_cpi: bool = False) -> Dict[str, float]:
    """Realised % returns of every benchmark between dates a and b.
    approx_cpi=True (reporting only): if the end month's CPI is not yet published, the
    latest published month is used."""
    from inflation import cpi_ratio
    out = {k: np.nan for k in ("cpi", "usd", "gold", "deposit", "xu100")}
    years = max((pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25, 0.0)
    c = cpi_ratio(cpi, a, b) if cpi is not None and len(cpi) else np.nan
    if approx_cpi and not np.isfinite(c) and cpi is not None and len(cpi) and cpi.index[-1] > pd.Timestamp(a):
        c = cpi_ratio(cpi, a, cpi.index[-1])
    out["cpi"] = (c - 1) * 100 if np.isfinite(c) else np.nan
    if bm is not None and len(bm):
        u0, u1 = _at(bm["usdtry"], a), _at(bm["usdtry"], b)
        if np.isfinite(u0) and np.isfinite(u1) and u0 > 0:
            out["usd"] = np.nan if "usd" not in C.HURDLE_COMPONENTS else \
                ((u1 / u0) * (1 + C.US_INFLATION_PCT / 100.0) ** years - 1) * 100
        for k in ("gold_try", "xu100"):
            if k in bm:
                v0, v1 = _at(bm[k], a), _at(bm[k], b)
                if np.isfinite(v0) and np.isfinite(v1) and v0 > 0:
                    out["gold" if k == "gold_try" else "xu100"] = (v1 / v0 - 1) * 100
    g = _deposit_growth(crate, a, b)
    out["deposit"] = (g - 1) * 100 if np.isfinite(g) else np.nan
    comps = [out[k] for k in C.HURDLE_COMPONENTS if np.isfinite(out.get(k, np.nan))]
    # the hurdle is only defined when CPI is known (primary objective)
    ok = bool(comps) and np.isfinite(out["cpi"])
    out["floor"] = max(comps) if ok else np.nan
    out["hurdle"] = combine(comps) if ok else np.nan
    return out


def combine(comps) -> float:
    """Target from the components: sum (+ margin) in 'sum' mode, max otherwise."""
    comps = [float(c) for c in comps]
    if getattr(C, "HURDLE_MODE", "max") == "sum":
        return sum(comps) + C.MIN_EDGE_OVER_HURDLE_PCT
    return max(comps)


def expected_hurdles(bm: Optional[pd.DataFrame], cpi_stats: Dict, cash_yield_pct: Optional[float], as_of=None) -> Dict:
    """Forward-looking 12m hurdle at decision time (uses only data <= as_of)."""
    exp = {"cpi": cpi_stats.get("expected_12m_pct")}
    usd = None
    if bm is not None and len(bm):
        s = bm["usdtry"].dropna()
        if as_of is not None:
            s = s[s.index <= pd.Timestamp(as_of)]
        if len(s) > 260:
            y12 = s.iloc[-1] / s.iloc[-253] - 1
            a6 = (s.iloc[-1] / s.iloc[-127]) ** 2 - 1
            usd = ((1 + 0.5 * y12 + 0.5 * a6) * (1 + C.US_INFLATION_PCT / 100.0) - 1) * 100
    exp["usd"] = None                 # USD investor: no FX leg
    exp["gold"] = exp["cpi"]          # gold: preserves real value, no own-drift forecast
    exp["deposit"] = None if cash_yield_pct is None else round(float(cash_yield_pct), 2)
    comps = {k: v for k, v in exp.items() if k in C.HURDLE_COMPONENTS and v is not None}
    if exp["cpi"] is None or not comps:
        return {**exp, "hurdle": None, "binding": None}
    binding = max(comps, key=comps.get)
    return {**exp, "hurdle": round(float(combine(comps.values())), 2), "floor": round(float(comps[binding]), 2),
            "binding": binding, "mode": getattr(C, "HURDLE_MODE", "max"),
            "edge": C.MIN_EDGE_OVER_HURDLE_PCT}
