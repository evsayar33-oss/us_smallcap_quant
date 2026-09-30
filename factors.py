"""Long-horizon factor engine — ONE code path for live and backtest.

Price factors are computed from adjusted daily history (yfinance) with numpy
windows at the decision date only (no look-ahead). Fundamental factors come
from TradingView (live) or point-in-time statements (backtest) mapped to the
same definitions. Cross-sectional transform: rank -> Gaussian -> sector-neutral
-> z, winsorised at +-3; missing = neutral 0 with coverage reported.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

import config as C


def wide_from_history(hist: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    close = pd.DataFrame({t: g["close"] for t, g in hist.items()}).sort_index()
    idx = close.index
    return {
        "open": pd.DataFrame({t: g["open"] for t, g in hist.items()}).reindex(idx),
        "close": close,
        "volume": pd.DataFrame({t: g["volume"] for t, g in hist.items()}).reindex(idx),
    }


def price_factors_at(wide: Dict[str, pd.DataFrame], index_close: Optional[pd.Series], date) -> pd.DataFrame:
    """Raw price factors for every ticker using data up to and including `date`."""
    close = wide["close"]
    pos = close.index.searchsorted(pd.Timestamp(date), side="right") - 1
    if pos < C.MIN_HISTORY_SESSIONS:
        return pd.DataFrame()
    W = close.iloc[pos - 252: pos + 1].to_numpy(float)            # 253 x N
    V = wide["volume"].iloc[pos - 62: pos + 1].to_numpy(float)
    Cv = close.iloc[pos - 62: pos + 1].to_numpy(float)
    tick = list(close.columns)
    valid = np.isfinite(W).sum(axis=0) >= 240
    with np.errstate(invalid="ignore", divide="ignore"):
        last = W[-1]
        mom = W[-22] / W[0] - 1.0
        hi = np.nanmax(W, axis=0)
        high_52w = last / hi
        months = np.array([W[-1 - 21 * k] / W[-1 - 21 * (k + 1)] - 1.0 for k in range(12)])
        trend = np.nanmean(months > 0, axis=0)
        lr = np.diff(np.log(W), axis=0)
        vol6 = np.nanstd(lr[-126:], axis=0) * np.sqrt(252)
        peak = np.fmax.accumulate(np.where(np.isfinite(W), W, -np.inf), axis=0)
        dd = np.nanmin(W / peak - 1.0, axis=0)
        med_val = np.nanmedian(V * Cv, axis=0)
        max_1m = np.nanmax(W[-21:] / W[-22:-1] - 1.0, axis=0) * 100    # lottery (MAX) effect: biggest daily jump, 1m
        ret_3m = (W[-1] / W[-64] - 1.0) * 100                            # V3.10 "value trap" check: started to rise?
        from_low = (W[-1] / np.nanmin(W, axis=0) - 1.0) * 100
        beta = np.full(len(tick), np.nan)
        if index_close is not None and len(index_close):
            ic = index_close.reindex(close.index).ffill().iloc[pos - 252: pos + 1].to_numpy(float)
            rm = np.diff(np.log(ic))
            if np.isfinite(rm).sum() > 200:
                m = np.isfinite(rm)
                rmv = rm[m] - np.nanmean(rm[m])
                X = lr[m]
                Xc = X - np.nanmean(X, axis=0)
                cov = np.nanmean(Xc * rmv[:, None], axis=0)
                beta = cov / np.nanvar(rm[m])
    df = pd.DataFrame({
        "ticker": tick, "close_adj": last, "mom_12_1": mom * 100, "high_52w": high_52w,
        "trend_consistency": trend, "low_vol": -vol6 * 100, "vol_ann_pct": vol6 * 100, "low_beta": -beta,
        "beta": beta, "dd_resilience": dd * 100, "liquidity": np.log1p(med_val), "med_value_traded": med_val,
        "max_1m": max_1m, "ret_3m": ret_3m, "from_52w_low": from_low,
    })
    df = df[valid & np.isfinite(last)]
    df["tarih"] = pd.Timestamp(close.index[pos]).normalize()
    return df.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)


def fundamental_factors(df: pd.DataFrame, cpi_yoy_pct: Optional[float]) -> pd.DataFrame:
    """Maps standardised fundamental inputs to factor values. Expected (optional) columns:
    roe, net_income, revenue, market_cap, pe, pb, ps, op_margin, debt_to_equity, div_yield, rev_growth, equity."""
    def col(n):
        return pd.to_numeric(df[n], errors="coerce") if n in df.columns else pd.Series(np.nan, index=df.index)
    f = pd.DataFrame(index=df.index)
    mcap = col("market_cap").where(lambda x: x > 0)
    f["roe"] = col("roe")
    ey = col("net_income") / mcap * 100.0
    pe = col("pe")
    f["earnings_yield"] = ey.fillna((100.0 / pe).where(pe.abs() > 1e-9))
    by = col("equity") / mcap * 100.0
    pb = col("pb")
    f["book_yield"] = by.fillna((100.0 / pb).where(pb.abs() > 1e-9))
    sy = col("revenue") / mcap * 100.0
    ps = col("ps")
    f["sales_yield"] = sy.fillna((100.0 / ps).where(ps > 1e-9))
    f["op_margin"] = col("op_margin")
    f["low_leverage"] = -col("debt_to_equity")
    f["div_yield"] = col("div_yield")
    g = col("rev_growth")
    f["real_growth"] = g - (cpi_yoy_pct if cpi_yoy_pct is not None else 0.0)
    return f.replace([np.inf, -np.inf], np.nan)


def _gauss_rank(x: pd.Series) -> pd.Series:
    n = x.notna().sum()
    if n < 5:
        return pd.Series(np.nan, index=x.index)
    r = x.rank(method="average")
    return pd.Series(norm.ppf((r - 0.5) / n), index=x.index)


def cross_sectional_z(raw: pd.DataFrame, sector: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Dict[str, float]]:
    z = pd.DataFrame(index=raw.index)
    cov = {}
    for k in C.FACTORS:
        x = raw[k] if k in raw else pd.Series(np.nan, index=raw.index)
        cov[k] = float(x.notna().mean()) if len(x) else 0.0
        if cov[k] < 0.5:
            z[k] = 0.0
            continue
        g = _gauss_rank(x)
        if sector is not None:
            sec = sector.fillna("NA").astype(str)
            sizes = sec.map(sec.value_counts())
            g = g - g.groupby(sec).transform("mean").where(sizes >= 5, 0.0)
        sd = g.std(ddof=0)
        g = (g - g.mean()) / sd if sd and np.isfinite(sd) and sd > 0 else g * 0.0
        z[k] = g.clip(-3, 3).fillna(0.0)
    return z, cov


def build_frame(price_f: pd.DataFrame, fund_inputs: Optional[pd.DataFrame], cpi_yoy_pct: Optional[float],
                sector_map: Optional[Dict[str, str]] = None) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Merge price factors + fundamentals for one date, then z-score cross-sectionally."""
    if price_f is None or price_f.empty:
        return pd.DataFrame(), {}
    fr = price_f.copy()
    if fund_inputs is not None and not fund_inputs.empty:
        keep = [c for c in fund_inputs.columns if c != "tarih"]
        fr = fr.merge(fund_inputs[keep].drop_duplicates("ticker"), on="ticker", how="left")
    ff = fundamental_factors(fr, cpi_yoy_pct)
    for k in C.FUNDAMENTAL_FACTORS:
        fr[k] = ff[k]
    if "sector" not in fr.columns or fr["sector"].isna().all():
        fr["sector"] = fr["ticker"].map(sector_map or {})
    sector = fr["sector"] if fr["sector"].notna().mean() > 0.5 else None
    z, cov = cross_sectional_z(fr, sector)
    for k in C.FACTORS:
        fr[f"f_{k}"] = fr[k]
        fr[f"z_{k}"] = z[k]
    fr = fr.drop(columns=[k for k in C.FACTORS if k in fr.columns])
    return fr, cov


def composite(frame: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    s = pd.Series(0.0, index=frame.index)
    for k, w in weights.items():
        c = f"z_{k}"
        if c in frame:
            s = s + float(w) * frame[c].fillna(0.0)
    return s


def fund_break(frame: pd.DataFrame) -> pd.Series:
    """Thesis break: loss-making AND negative ROE (both must be observed)."""
    ey = pd.to_numeric(frame.get("f_earnings_yield"), errors="coerce") if "f_earnings_yield" in frame else pd.Series(np.nan, index=frame.index)
    roe = pd.to_numeric(frame.get("f_roe"), errors="coerce") if "f_roe" in frame else pd.Series(np.nan, index=frame.index)
    return (ey < 0) & (roe < 0)
