"""Forward labels for the long-horizon engine (same for live learning and backtest).

For a decision taken after the close of date t:
  entry  = open of the next session
  fwd_ret  = close 12 months (252 sessions) later / entry - 1        (nominal, %)
  real_ret = (1 + fwd_ret) / (CPI[exit month] / CPI[entry month]) - 1  (CPI-deflated, %)
  xu_excess = fwd_ret - XU100 return over the same window            (%)
  fwd_1m / fwd_3m: shorter horizons used only for early monitoring (guard).
Labels exist only when the full window is in the data -> no look-ahead.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config as C
from benchmarks import window_returns
from inflation import cpi_ratio_vec

H12 = 252


def forward_labels(wide: Dict[str, pd.DataFrame], dates: List, cpi: Optional[pd.Series],
                   index_close: Optional[pd.Series], bm: Optional[pd.DataFrame] = None,
                   crate: Optional[pd.Series] = None) -> pd.DataFrame:
    O, Cl = wide["open"], wide["close"]
    idx = Cl.index
    Ov, Cv = O.to_numpy(float), Cl.to_numpy(float)
    tick = list(Cl.columns)
    ic = index_close.reindex(idx).ffill().to_numpy(float) if index_close is not None and len(index_close) else None
    out = []
    for d in dates:
        pos = idx.searchsorted(pd.Timestamp(d), side="right") - 1
        if pos < 0 or pos + 1 >= len(idx):
            continue
        entry = Ov[pos + 1]
        entry = np.where(np.isfinite(entry), entry, Cv[pos])
        rec = {"tarih": pd.Timestamp(idx[pos]).normalize(), "ticker": tick}
        for name, h in (("fwd_1m", 21), ("fwd_3m", 63), ("fwd_ret", H12)):
            e = pos + h
            rec[name] = (Cv[e] / entry - 1.0) * 100.0 if e < len(idx) else np.full(len(tick), np.nan)
        e = pos + H12
        if e < len(idx):
            cr = cpi_ratio_vec(cpi, [idx[pos + 1]] * 1, [idx[e]])[0] if cpi is not None else np.nan
            rec["real_ret"] = ((1 + rec["fwd_ret"] / 100.0) / cr - 1.0) * 100.0 if np.isfinite(cr) else np.full(len(tick), np.nan)
            rec["cpi_12m_pct"] = (cr - 1.0) * 100.0 if np.isfinite(cr) else np.nan
            if ic is not None and np.isfinite(ic[e]) and np.isfinite(ic[pos]):
                rec["xu_excess"] = rec["fwd_ret"] - (ic[e] / ic[pos] - 1.0) * 100.0
            else:
                rec["xu_excess"] = np.full(len(tick), np.nan)
            w = window_returns(bm, cpi, crate, idx[pos + 1], idx[e])
            rec["hurdle_ret"] = w["hurdle"]
            rec["b_usd"], rec["b_gold"], rec["b_deposit"] = w["usd"], w["gold"], w["deposit"]
            rec["beat_all"] = np.where(np.isfinite(w["hurdle"]), (rec["fwd_ret"] > w["hurdle"]).astype(float), np.nan) \
                if np.isfinite(w["hurdle"]) else np.full(len(tick), np.nan)
        else:
            rec["real_ret"] = rec["xu_excess"] = np.full(len(tick), np.nan)
            rec["cpi_12m_pct"] = np.nan
            rec["hurdle_ret"] = rec["b_usd"] = rec["b_gold"] = rec["b_deposit"] = np.nan
            rec["beat_all"] = np.full(len(tick), np.nan)
        df = pd.DataFrame(rec)
        df = df[np.isfinite(df["fwd_1m"]) | np.isfinite(df["fwd_ret"])]
        out.append(df)
    if not out:
        return pd.DataFrame(columns=["tarih", "ticker", "fwd_1m", "fwd_3m", "fwd_ret", "real_ret", "xu_excess", "cpi_12m_pct",
                                     "hurdle_ret", "b_usd", "b_gold", "b_deposit", "beat_all"])
    return pd.concat(out, ignore_index=True)


def month_start_sessions(idx: pd.DatetimeIndex, start=None, end=None) -> List[pd.Timestamp]:
    """First trading session of each month present in `idx`."""
    s = pd.Series(idx, index=idx)
    if start is not None:
        s = s[s >= pd.Timestamp(start)]
    if end is not None:
        s = s[s <= pd.Timestamp(end)]
    return list(s.groupby([s.index.year, s.index.month]).min())
