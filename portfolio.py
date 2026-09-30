"""Portfolio accounting shared by the live engine and the backtest (single code path).

Daily cycle (after the close of day t):
  1. pending orders from the previous decision execute at day t's OPEN
     (sells first, then weight reductions, then buys), with one-way costs;
  2. positions are marked to day t's CLOSE using the split/dividend-adjusted
     daily change (TradingView `change` live, adjusted closes in the backtest),
     so bonus issues never create fake losses;
  3. a catastrophe stop (35% below the highest close since entry, or 30% below
     entry) queues a SELL for the next open. There is NO short-term ATR stop:
     this is a long-horizon engine; the thesis exit happens at the monthly review.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import config as C


def new_portfolio(start_date) -> Dict:
    return {"cash": 1.0, "positions": {}, "pending": [], "last_rebalance_month": None,
            "start_date": str(pd.Timestamp(start_date).date()), "nav": 1.0}


def nav(pf: Dict) -> float:
    return float(pf["cash"] + sum(p["value"] for p in pf["positions"].values()))


def weights(pf: Dict) -> Dict[str, float]:
    n = nav(pf)
    return {t: p["value"] / n for t, p in pf["positions"].items()} if n > 0 else {}


def _queue(pf: Dict, ticker: str, action: str, reason: str, target_w: float = 0.0, meta: Dict = None):
    for o in pf["pending"]:
        if o["ticker"] == ticker and o["action"] == action:
            return
    pf["pending"].append({"ticker": ticker, "action": action, "reason": reason, "target_w": float(target_w),
                          **(meta or {})})


def apply_day(pf: Dict, bars: pd.DataFrame, date, cash_yield_pct: float = None) -> Tuple[List[Dict], List[Dict]]:
    """bars: index=ticker, columns open, close, chg_pct (adjusted close-to-close %)."""
    date = pd.Timestamp(date).normalize()
    ds = str(date.date())
    events, lots = [], []
    cost = C.COST_ONE_WAY
    has = set(bars.index)
    b_open = bars["open"].to_dict()
    b_close = bars["close"].to_dict()
    b_chg = bars["chg_pct"].to_dict()

    # 1) mark to OPEN
    gap = {}
    for t, p in pf["positions"].items():
        if t in has and np.isfinite(b_open.get(t, np.nan)) and np.isfinite(b_close.get(t, np.nan)) \
                and np.isfinite(b_chg.get(t, np.nan)) and b_close[t] > 0:
            prev_close = b_close[t] / (1.0 + b_chg[t] / 100.0)
            gap[t] = b_open[t] / prev_close
            p["value"] *= gap[t]

    # 2) execute orders at OPEN
    remaining = []
    # sells -> trims -> new buys -> top-ups (new names get funded before existing names are topped up)
    def _rank(o):
        if o["action"] == "SELL":
            return 0
        if o["action"] == "REBAL":
            p0 = pf["positions"].get(o["ticker"])
            return 1 if p0 is not None and o["target_w"] * nav(pf) < p0["value"] else 3
        return 2 if o["action"] == "BUY" else 4
    for o in sorted(pf["pending"], key=_rank):
        t = o["ticker"]
        if t not in has or not np.isfinite(b_open.get(t, np.nan)):
            o["age"] = int(o.get("age", 0)) + 1
            if o["age"] <= 10:
                remaining.append(o)            # suspended / no data: retry next session
            else:
                events.append({"ticker": t, "type": f"ORDER_EXPIRED_{o['action']}"})
            continue
        if o["action"] == "SELL" and t in pf["positions"]:
            p = pf["positions"].pop(t)
            proceeds = p["value"] * (1 - cost)
            pf["cash"] += proceeds
            lots.append({"ticker": t, "entry_date": p["entry_date"], "exit_date": ds, "reason": o["reason"],
                         "nominal_ret_pct": round((proceeds / p["cost_basis"] - 1) * 100, 4),
                         "entry_pct": p.get("entry_pct"), "entry_exp_real": p.get("entry_exp_real"),
                         "entry_conf": p.get("entry_conf"),
                         "peak_level": round(p["peak"], 4)})
            events.append({"ticker": t, "type": f"SELL_{o['reason']}", "ret": lots[-1]["nominal_ret_pct"]})
        elif o["action"] == "REBAL" and t in pf["positions"]:
            p = pf["positions"][t]
            target = o["target_w"] * nav(pf)
            diff = target - p["value"]
            if diff < 0:
                sell = -diff
                pf["cash"] += sell * (1 - cost)
                p["cost_basis"] *= (p["value"] - sell) / p["value"]
                p["value"] -= sell
            elif diff > 0:
                amt = min(diff, pf["cash"])
                pf["cash"] -= amt
                p["value"] += amt * (1 - cost)
                p["cost_basis"] += amt
            events.append({"ticker": t, "type": "REBAL", "target_w": round(o["target_w"], 4)})
        elif o["action"] == "BUY" and t not in pf["positions"]:
            amt = min(o["target_w"] * nav(pf), pf["cash"])
            if amt <= 0.002 * nav(pf):
                continue
            pf["cash"] -= amt
            pf["positions"][t] = {"value": amt * (1 - cost), "cost_basis": amt, "entry_date": ds,
                                  "entry_px": float(b_open[t]), "level": 1.0, "peak": 1.0, "missing": 0,
                                  "entry_pct": o.get("entry_pct"), "entry_exp_real": o.get("entry_exp_real"),
                                  "entry_exp_nominal": o.get("entry_exp_nominal"), "entry_hurdle": o.get("entry_hurdle"),
                                  "entry_p_beat_all": o.get("entry_p_beat_all"), "entry_conf": o.get("entry_conf"),
                                  "stop_level": (1.0 - o["stop_pct"]) if o.get("stop_pct") else None,
                                  "bought_today": True}
            events.append({"ticker": t, "type": "BUY", "w": round(o["target_w"], 4)})
    pf["pending"] = remaining

    # 3) mark to CLOSE, level/peak, catastrophe stop
    for t, p in pf["positions"].items():
        if t in has and np.isfinite(b_close.get(t, np.nan)) and np.isfinite(b_open.get(t, np.nan)) and b_open[t] > 0:
            r_oc = b_close[t] / b_open[t]
            p["value"] *= r_oc
            p["level"] = (r_oc if p.pop("bought_today", False) else p["level"] * gap.get(t, 1.0) * r_oc)
            p["peak"] = max(p["peak"], p["level"])
            p["missing"] = 0
            p["last_close"] = float(b_close[t])
            if p.get("stop_level") is not None:
                if p["level"] >= 1 + C.BREAKEVEN_TRIGGER_PCT / 100:
                    p["stop_level"] = max(p["stop_level"], 1.0)          # breakeven stop
                if p["level"] <= p["stop_level"] and not any(o["ticker"] == t and o["action"] == "SELL" for o in pf["pending"]):
                    _queue(pf, t, "SELL", "STOP")
                    pf.setdefault("stopped", {})[t] = date.strftime("%Y-%m")
                    events.append({"ticker": t, "type": "STOP_QUEUED", "level": round((p["level"] - 1) * 100, 2)})
            dd_peak = (p["level"] / p["peak"] - 1) * 100
            dd_entry = (p["level"] - 1) * 100
            if dd_entry <= -C.HARD_STOP_FROM_ENTRY_PCT:
                _queue(pf, t, "SELL", "CATASTROPHE_STOP")
                events.append({"ticker": t, "type": "HARD_STOP_QUEUED", "dd_entry": round(dd_entry, 2)})
            elif (dd_peak <= -C.CATASTROPHE_FROM_PEAK_PCT or dd_entry <= -C.CATASTROPHE_FROM_ENTRY_PCT) \
                    and not p.get("dd_flag"):
                p["dd_flag"] = True      # reviewed at the next monthly review (sold only if thesis failed)
                events.append({"ticker": t, "type": "DRAWDOWN_FLAG", "dd_peak": round(dd_peak, 2)})
        else:
            p["missing"] = int(p.get("missing", 0)) + 1
            p.pop("bought_today", None)
            if p["missing"] == 15:
                events.append({"ticker": t, "type": "NO_DATA_15_SESSIONS"})
    y = C.CASH_YIELD_ANNUAL_PCT if cash_yield_pct is None else cash_yield_pct
    if y and pf["cash"] > 0:
        pf["cash"] *= (1 + y / 100.0) ** (1 / 252)
    pf["nav"] = nav(pf)
    pf["last_date"] = ds
    return events, lots
