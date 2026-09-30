"""Performance measurement in the engine's own yardstick: REAL (CPI) return,
with XU100 as the secondary benchmark. Used by the live report, the weekly
audit and the backtest, so all three speak the same language."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

import config as C
from inflation import cpi_ratio, cpi_ratio_vec


def enrich_lots(lots: pd.DataFrame, cpi: Optional[pd.Series], index_close: Optional[pd.Series],
                bench: Optional[Dict] = None) -> pd.DataFrame:
    if lots is None or lots.empty:
        return pd.DataFrame()
    L = lots.copy()
    L["entry_date"] = pd.to_datetime(L["entry_date"])
    L["exit_date"] = pd.to_datetime(L["exit_date"])
    L["months_held"] = ((L["exit_date"] - L["entry_date"]).dt.days / 30.44).round(1)
    cr = cpi_ratio_vec(cpi, L["entry_date"], L["exit_date"]) if cpi is not None and len(cpi) else np.full(len(L), np.nan)
    L["cpi_ret_pct"] = (cr - 1) * 100
    L["real_ret_pct"] = ((1 + L["nominal_ret_pct"] / 100) / cr - 1) * 100
    if index_close is not None and len(index_close):
        ic = index_close.sort_index()
        a = ic.reindex(L["entry_date"], method="ffill").to_numpy()
        b = ic.reindex(L["exit_date"], method="ffill").to_numpy()
        L["xu100_ret_pct"] = (b / a - 1) * 100
        L["xu_excess_pct"] = L["nominal_ret_pct"] - L["xu100_ret_pct"]
    if bench:
        from benchmarks import window_returns
        rows = [window_returns(bench.get("bm"), cpi, bench.get("crate"), a, b) for a, b in zip(L["entry_date"], L["exit_date"])]
        for k in ("usd", "gold", "deposit", "hurdle"):
            L[f"{k}_ret_pct"] = [r[k] for r in rows]
        L["beat_all"] = np.where(np.isfinite(L["hurdle_ret_pct"]), (L["nominal_ret_pct"] > L["hurdle_ret_pct"]).astype(float), np.nan)
    return L


def lot_metrics(L: pd.DataFrame) -> Dict:
    if L is None or L.empty:
        return {"closed_lots": 0}
    r = pd.to_numeric(L["nominal_ret_pct"], errors="coerce")
    real = pd.to_numeric(L.get("real_ret_pct"), errors="coerce") if "real_ret_pct" in L else pd.Series(dtype=float)
    out = {
        "closed_lots": int(len(L)),
        "avg_months_held": round(float(L["months_held"].mean()), 1) if "months_held" in L else None,
        "hit_nominal_pct": round(float((r > 0).mean() * 100), 1),
        "avg_nominal_pct": round(float(r.mean()), 2),
        "exit_mix": L["reason"].value_counts().to_dict() if "reason" in L else {},
    }
    if real.notna().any():
        rr = real.dropna()
        out.update({"hit_beat_cpi_pct": round(float((rr > 0).mean() * 100), 1),
                    "avg_real_pct": round(float(rr.mean()), 2), "median_real_pct": round(float(rr.median()), 2),
                    "n_real": int(len(rr))})
    for k, name in (("usd", "usd"), ("gold", "gold"), ("deposit", "deposit")):
        col = f"{k}_ret_pct"
        if col in L and L[col].notna().any():
            m = L[col].notna()
            out[f"hit_beat_{name}_pct"] = round(float((L.loc[m, "nominal_ret_pct"] > L.loc[m, col]).mean() * 100), 1)
    if "beat_all" in L and L["beat_all"].notna().any():
        out["hit_beat_all_pct"] = round(float(L["beat_all"].dropna().mean() * 100), 1)
    if "xu_excess_pct" in L and L["xu_excess_pct"].notna().any():
        x = L["xu_excess_pct"].dropna()
        out.update({"hit_beat_xu100_pct": round(float((x > 0).mean() * 100), 1),
                    "avg_xu_excess_pct": round(float(x.mean()), 2)})
    return out


def nav_metrics(nav_df: pd.DataFrame, cpi: Optional[pd.Series], bench: Optional[Dict] = None) -> Dict:
    if nav_df is None or len(nav_df) < 2:
        return {"days": 0}
    d = nav_df.copy()
    d["tarih"] = pd.to_datetime(d["tarih"])
    d = d.sort_values("tarih")
    nv = pd.to_numeric(d["nav"], errors="coerce").ffill()
    r = nv.pct_change().fillna(0.0)
    years = max((d["tarih"].iloc[-1] - d["tarih"].iloc[0]).days / 365.25, 1e-9)
    total = nv.iloc[-1] / nv.iloc[0]
    out = {"start": str(d["tarih"].iloc[0].date()), "end": str(d["tarih"].iloc[-1].date()),
           "days": int(len(d)), "total_return_pct": round((total - 1) * 100, 2),
           "cagr_pct": round((total ** (1 / years) - 1) * 100, 2) if total > 0 else -100.0,
           "max_drawdown_pct": round(float((nv / nv.cummax() - 1).min() * 100), 2),
           "ann_vol_pct": round(float(r.std(ddof=0) * np.sqrt(252) * 100), 2),
           "avg_exposure_pct": round(float(pd.to_numeric(d.get("exposure"), errors="coerce").mean() * 100), 1)
           if "exposure" in d else None}
    if "xu100" in d and d["xu100"].notna().sum() > 2:
        x = pd.to_numeric(d["xu100"], errors="coerce").ffill().bfill()
        xt = x.iloc[-1] / x.iloc[0]
        out["xu100_total_pct"] = round((xt - 1) * 100, 2)
        out["xu100_cagr_pct"] = round((xt ** (1 / years) - 1) * 100, 2)
        out["excess_vs_xu100_cagr_pp"] = round(out["cagr_pct"] - out["xu100_cagr_pct"], 2)
    if cpi is not None and len(cpi) > 13:
        cr = cpi_ratio(cpi, d["tarih"].iloc[0], min(d["tarih"].iloc[-1], cpi.index[-1] + pd.offsets.MonthEnd(0)))
        if np.isfinite(cr):
            real_total = total / cr
            out["cpi_total_pct"] = round((cr - 1) * 100, 2)
            out["real_total_pct"] = round((real_total - 1) * 100, 2)
            out["real_cagr_pct"] = round((real_total ** (1 / years) - 1) * 100, 2) if real_total > 0 else -100.0
        # rolling 12-month windows: share that beat CPI (the user's "win rate" for the portfolio)
        m = d.set_index("tarih")["nav"].resample("ME").last().dropna()
        if len(m) >= 13:
            wins, beats_xu, n = 0, 0, 0
            xm = d.set_index("tarih")["xu100"].resample("ME").last() if "xu100" in d else None
            for i in range(12, len(m)):
                a, b = m.index[i - 12], m.index[i]
                c = cpi_ratio(cpi, a, b)
                if not np.isfinite(c):
                    continue
                n += 1
                wins += int(m.iloc[i] / m.iloc[i - 12] > c)
                if xm is not None and np.isfinite(xm.get(a, np.nan)) and np.isfinite(xm.get(b, np.nan)):
                    beats_xu += int(m.iloc[i] / m.iloc[i - 12] > xm[b] / xm[a])
            if n:
                out["rolling12m_windows"] = n
                out["rolling12m_beat_cpi_pct"] = round(wins / n * 100, 1)
                out["rolling12m_beat_xu100_pct"] = round(beats_xu / n * 100, 1)
    if bench:
        from benchmarks import window_returns
        a0, a1 = d["tarih"].iloc[0], d["tarih"].iloc[-1]
        tot = window_returns(bench.get("bm"), cpi, bench.get("crate"), a0, a1, approx_cpi=True)
        out["benchmarks_total_pct"] = {k: (round(float(v), 2) if np.isfinite(v) else None) for k, v in tot.items()}
        m = d.set_index("tarih")["nav"].resample("ME").last().dropna()
        if len(m) >= 13:
            wins = {k: 0 for k in ("cpi", *[c for c in ("usd", "gold", "deposit") if c in C.HURDLE_COMPONENTS], "floor", "hurdle")}
            n, gaps = 0, []
            for i in range(12, len(m)):
                a, b = m.index[i - 12], m.index[i]
                w = window_returns(bench.get("bm"), cpi, bench.get("crate"), a, b)
                if not np.isfinite(w["hurdle"]):
                    continue
                n += 1
                r = (m.iloc[i] / m.iloc[i - 12] - 1) * 100
                gaps.append(r - w["hurdle"])
                for k in wins:
                    wins[k] += int(np.isfinite(w.get(k, np.nan)) and r > w[k])
            if n:
                out["rolling12m_beat"] = {({"hurdle": "all", "floor": "each"}.get(k, k)): round(v / n * 100, 1)
                                          for k, v in wins.items()}
                out["rolling12m_median_gap_pp"] = round(float(np.median(gaps)), 1)
                out["rolling12m_beat_all_pct"] = round(wins["hurdle"] / n * 100, 1)
    return out
