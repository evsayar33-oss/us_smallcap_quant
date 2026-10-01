"""Scoring and monthly portfolio construction (live AND backtest use these functions).

V3.8 (live): plan_tranche — 100% stocks, monthly cohorts:
* every month the TRANCHE_N highest-scored liquid stocks form a new cohort
  (max TRANCHE_SECTOR_CAP per sector, no limit-up buys),
* every cohort is held TRANCHE_MONTHS, the portfolio is the union of the active cohorts
  (a name picked in several months naturally carries more weight, capped at MAX_NAME_W),
* every pick carries a calibrated confidence (confidence.py).
The older rules below (plan_rebalance) are kept only for reference/back-compat.

Legacy (V3.0-3.7):

Rules (long-horizon, low turnover):
* BUY  : composite percentile >= calibrated cut-off, expected 12m REAL return
         after costs >= MIN_EXPECTED_REAL_PCT, liquid, no fundamental break,
         max MAX_PER_SECTOR names per sector, up to TARGET_POSITIONS x exposure.
* HOLD : keep while percentile >= HOLD_PCT and no fundamental break
         (hysteresis 85 -> 60 keeps winners for years).
* SELL : percentile < HOLD_PCT (thesis weakened) or fundamental break;
         catastrophe stop is handled daily in portfolio.apply_day.
* Weights: inverse volatility, capped at MAX_POSITION_W, total = exposure
  (autonomy guard x regime). Idle weight stays in cash. Holdings are only
  re-weighted when they drift more than REBALANCE_BAND.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config as C
from calibration import bucket_lookup, market_expectation
from factors import build_frame, composite, fund_break
from portfolio import weights as pf_weights


def active_weights(state: Dict) -> Tuple[Dict[str, float], str]:
    model = state.get("model", {})
    champ = model.get("champion_weights") or dict(C.PRIOR_IC)
    reg = state.get("regime", {})
    label = reg.get("label")
    p = float((reg.get("probs") or {}).get(label, 0.0) or 0.0)
    rw = (model.get("regime_weights") or {}).get(label)
    if rw and p >= 0.6 and not reg.get("degraded", True):
        w = {k: p * rw.get(k, 0.0) + (1 - p) * champ.get(k, 0.0) for k in C.FACTORS}
        version = f"{model.get('champion_version', 'prior')}+{label}"
    else:
        w, version = {k: champ.get(k, 0.0) for k in C.FACTORS}, model.get("champion_version", "prior")
    tot = sum(abs(v) for v in w.values()) or 1.0
    return {k: v / tot for k, v in w.items()}, version


def score_universe(price_f: pd.DataFrame, fund_inputs: Optional[pd.DataFrame], state: Dict,
                   cpi_stats: Dict, sector_map: Optional[Dict[str, str]] = None) -> Tuple[pd.DataFrame, Dict]:
    frame, cov = build_frame(price_f, fund_inputs, cpi_stats.get("yoy_pct"), sector_map)
    if frame.empty:
        return frame, {"error": "no_price_factors"}
    w, version = active_weights(state)
    frame["composite"] = composite(frame, w)
    frame["composite_pct"] = frame["composite"].rank(pct=True) * 100.0
    frame["model_version"] = version
    cal = state.get("calibration", {})
    reg = state.get("regime", {})
    calibrated = cal.get("status") == "CALIBRATED"
    table = cal.get("bucket_table", [])
    exc = bucket_lookup(frame["composite_pct"], table, "mean_excess") if calibrated else np.zeros(len(frame))
    frame["exp_excess_12m"] = np.nan_to_num(exc, nan=0.0)
    frame["p_beat_cpi"] = bucket_lookup(frame["composite_pct"], table, "p_beat_cpi") if calibrated else np.nan
    mk = market_expectation(cal, reg.get("label"), reg.get("exp_mkt_12m_pct"))
    infl = cpi_stats.get("expected_12m_pct")
    frame["exp_real_12m"] = np.nan
    frame["exp_nominal_12m"] = np.nan
    if infl is not None and mk["mkt_12m_pct"] is not None:
        if mk.get("basis") == "real":
            # E[real] = E[universe real | regime] + E[excess | score] - costs
            real = mk["mkt_12m_pct"] + frame["exp_excess_12m"]
            frame["exp_nominal_12m"] = ((1 + real / 100.0) * (1 + infl / 100.0) - 1.0) * 100.0 - C.COST_ROUND_TRIP_PCT
        else:
            frame["exp_nominal_12m"] = mk["mkt_12m_pct"] + frame["exp_excess_12m"] - C.COST_ROUND_TRIP_PCT
        frame["exp_real_12m"] = ((1 + frame["exp_nominal_12m"] / 100.0) / (1 + infl / 100.0) - 1.0) * 100.0
    # multi-benchmark hurdle (CPI, USD, gold, deposit): expected margin over the highest one
    hz = state.get("hurdles", {}) or {}
    hurdle = hz.get("hurdle")
    frame["hurdle_12m"] = hurdle if hurdle is not None else np.nan
    frame["exp_over_hurdle"] = frame["exp_nominal_12m"] - hurdle if hurdle is not None else np.nan
    floor = hz.get("floor", hurdle if hz.get("mode") != "sum" else None)
    frame["exp_over_floor"] = frame["exp_nominal_12m"] - floor if floor is not None else np.nan
    frame["p_beat_all"] = bucket_lookup(frame["composite_pct"], table, "p_beat_all") if calibrated else np.nan
    frame["fund_break"] = fund_break(frame)
    frame["regime_label"] = reg.get("label", "UNKNOWN")
    info = {"weights": {k: round(v, 4) for k, v in w.items()}, "model_version": version,
            "coverage": {k: round(v, 3) for k, v in cov.items()}, "calibrated": calibrated,
            "market_12m": mk, "expected_inflation_12m": infl, "n_scored": int(len(frame)),
            "hurdles": hz,
            "gate": ("multi_hurdle" if hurdle is not None else "real_return") if calibrated and infl is not None else ("rank_only_uncalibrated" if infl is not None else "blocked_no_inflation")}
    return frame, info


GOLD_TICKER = "ALTIN"      # gold sleeve (live: gram altın / ALTINS1 sertifikası)


def default_strategy() -> Dict:
    return {"name": "default", "n_positions": C.TARGET_POSITIONS, "buy_pct": None, "hold_pct": C.HOLD_PCT,
            "weighting": "inv_vol", "gate": "hurdle", "overlay": "none"}


def plan_rebalance(pf: Dict, frame: pd.DataFrame, state: Dict, exposure: float,
                   block: bool = False, today_change: Optional[pd.Series] = None,
                   params: Optional[Dict] = None, equity_frac: float = 1.0, gold_w: float = 0.0) -> Tuple[List[Dict], Dict]:
    """Orders for the next open + summary. `params` = strategy variant (see strategy_lab.py);
    `equity_frac`/`gold_w` come from the allocation overlay. Low exposure never forces
    selling on its own; only an overlay rotation (equity_frac == 0) exits all stocks."""
    P = {**default_strategy(), **(params or {})}
    cal = state.get("calibration", {})
    cutoff = float(P["buy_pct"]) if P.get("buy_pct") is not None else float(cal.get("pct_cutoff", C.DEFAULT_PCT_CUTOFF))
    hold_pct = float(P.get("hold_pct", C.HOLD_PCT))
    n_pos = int(P.get("n_positions", C.TARGET_POSITIONS))
    max_w = max(C.MAX_POSITION_W, 1.0 / max(n_pos, 1) + 0.05)
    max_sector = C.MAX_PER_SECTOR if n_pos >= 8 else 2
    f = frame.set_index("ticker")
    for c in ("exp_over_hurdle", "exp_over_floor", "exp_real_12m", "exp_nominal_12m", "p_beat_all", "hurdle_12m"):
        if c not in f.columns:
            f[c] = np.nan
    cur = [t for t in pf["positions"].keys() if t != GOLD_TICKER]
    orders, holds, sells = [], [], []
    if block:
        return [], {"blocked": True, "holds": cur, "sells": [], "buys": []}
    rotate_out = equity_frac <= 0.0
    for t in cur:
        if rotate_out:
            sells.append((t, "ROTATION"))
            continue
        if t not in f.index:
            holds.append(t)                     # no data this month: do not trade blind
            continue
        r = f.loc[t]
        flagged = bool(pf["positions"][t].get("dd_flag", False))
        if bool(r.get("fund_break", False)):
            sells.append((t, "FUND_BREAK"))
        elif float(r["composite_pct"]) < hold_pct:
            sells.append((t, "RANK_EXIT"))
        elif flagged and float(r["composite_pct"]) < cutoff:
            sells.append((t, "DRAWDOWN_CONFIRMED"))   # deep drawdown AND the thesis has weakened
        else:
            if flagged:
                pf["positions"][t]["dd_flag"] = False  # thesis intact: keep holding through the drawdown
            holds.append(t)
    eq_exposure = max(0.0, min(1.0, exposure)) * max(0.0, min(1.0, equity_frac))
    n_target = n_pos if eq_exposure > 0 else 0
    sector_count: Dict[str, int] = {}
    for t in holds:
        s = str(f.loc[t, "sector"]) if t in f.index and pd.notna(f.loc[t].get("sector")) else "NA"
        sector_count[s] = sector_count.get(s, 0) + 1
    elig = (
        (f["composite_pct"] >= cutoff)
        & (pd.to_numeric(f["med_value_traded"], errors="coerce") >= float(state.get("liq_floor_tl") or C.MIN_MEDIAN_VALUE_TRADED_TL))
        & ~f["fund_break"].astype(bool)
        & ~f.index.isin(cur)
    )
    calibrated = cal.get("status") == "CALIBRATED"
    has_real = f["exp_real_12m"].notna().any()
    no_inflation = bool(state.get("_no_inflation", False))
    if no_inflation:
        elig &= False          # inflation unknown -> cannot judge the hurdle -> no new buys
    elif P.get("gate", "hurdle") == "hurdle":
        sum_mode = (state.get("hurdles") or {}).get("mode") == "sum"
        if calibrated and sum_mode and f["exp_over_floor"].notna().any():
            # V3.6: target = CPI+USD+gold+deposit (+3). Entry floor = must at least beat the
            # strongest single alternative by the margin (otherwise that alternative is simply better);
            # candidates are then ranked by distance to the SUM target and its probability.
            elig &= f["exp_over_floor"] >= C.MIN_EDGE_OVER_HURDLE_PCT
        elif calibrated and f["exp_over_hurdle"].notna().any():
            # must be expected to beat CPI, USD (+US inflation), gold and TL deposit by a margin
            elig &= f["exp_over_hurdle"] >= C.MIN_EDGE_OVER_HURDLE_PCT
        elif calibrated and has_real:
            elig &= f["exp_real_12m"] >= C.MIN_EXPECTED_REAL_PCT
    # gate == "rank": the strongest ranks fill the book (score cut-off only)
    if today_change is not None:
        chg = today_change.reindex(f.index)
        elig &= ~(chg >= C.LIMIT_MOVE_PCT)
    cand = f[elig].copy()
    sort_col = "composite" if P.get("gate") == "rank" else (
        "exp_over_hurdle" if cand["exp_over_hurdle"].notna().any() else (
            "exp_real_12m" if cand["exp_real_12m"].notna().any() else "composite"))
    cand = cand.sort_values([sort_col, "composite"], ascending=False)
    buys = []
    for t, r in cand.iterrows():
        if len(holds) + len(buys) >= n_target:
            break
        s = str(r.get("sector")) if pd.notna(r.get("sector")) else "NA"
        if s != "NA" and sector_count.get(s, 0) >= max_sector:
            continue
        sector_count[s] = sector_count.get(s, 0) + 1
        buys.append(t)
    final = holds + buys
    tw = {}
    if final and eq_exposure > 0:
        vol = pd.to_numeric(f["vol_ann_pct"], errors="coerce")
        med = float(vol.median()) if vol.notna().any() else 40.0
        if P.get("weighting") == "equal":
            inv = {t: 1.0 for t in final}
        elif P.get("weighting") == "conviction":
            # stronger score -> bigger weight (rank above the hold line), still vol-aware
            cp = pd.to_numeric(f["composite_pct"], errors="coerce")

            def _conv(t):
                c = cp.get(t, np.nan)
                c = float(c) if pd.notna(c) else hold_pct
                v = vol.get(t, np.nan)
                v = float(v) if pd.notna(v) else med
                return (max(c - hold_pct, 0.0) + 5.0) / max(v, 5.0) ** 0.5
            inv = {t: _conv(t) for t in final}
        else:
            inv = {t: 1.0 / max(float(vol.get(t, med)) if pd.notna(vol.get(t, np.nan)) else med, 5.0) for t in final}
        # the book is sized for n_pos names: fewer qualifying names -> the rest stays in cash
        slots = max(len(final), n_pos)
        tot = sum(inv.values()) * slots / len(final)
        raw = {t: v / tot * eq_exposure for t, v in inv.items()}
        for _ in range(10):                                   # cap and redistribute
            over = {t: w for t, w in raw.items() if w > max_w}
            if not over:
                break
            excess = sum(w - max_w for w in over.values())
            for t in over:
                raw[t] = max_w
            under = [t for t in raw if raw[t] < max_w]
            us = sum(inv[t] for t in under)
            if not under or us <= 0:
                break
            for t in under:
                raw[t] += excess * inv[t] / us
        tw = raw
    cw = pf_weights(pf)
    for t, why in sells:
        orders.append({"ticker": t, "action": "SELL", "reason": why, "target_w": 0.0})
    for t in buys:
        r = f.loc[t]
        orders.append({"ticker": t, "action": "BUY", "reason": "NEW_ENTRY", "target_w": round(tw.get(t, 0.0), 5),
                       "entry_pct": round(float(r["composite_pct"]), 2),
                       "entry_exp_real": None if pd.isna(r["exp_real_12m"]) else round(float(r["exp_real_12m"]), 2),
                       "entry_exp_nominal": None if pd.isna(r["exp_nominal_12m"]) else round(float(r["exp_nominal_12m"]), 2),
                       "entry_hurdle": None if pd.isna(r["hurdle_12m"]) else round(float(r["hurdle_12m"]), 2),
                       "entry_p_beat_all": None if pd.isna(r["p_beat_all"]) else round(float(r["p_beat_all"]), 3)})
    for t in holds:
        if t in tw and abs(cw.get(t, 0.0) - tw[t]) > C.REBALANCE_BAND:
            orders.append({"ticker": t, "action": "REBAL", "reason": "WEIGHT_DRIFT", "target_w": round(tw[t], 5)})
    # gold sleeve (allocation overlay)
    gold_w = float(max(0.0, min(1.0, gold_w))) * max(0.0, min(1.0, exposure))
    have_gold = GOLD_TICKER in pf["positions"]
    if gold_w > 0 and not have_gold:
        orders.append({"ticker": GOLD_TICKER, "action": "BUY", "reason": "ALLOCATION", "target_w": round(gold_w, 5)})
    elif gold_w <= 0 and have_gold:
        orders.insert(0, {"ticker": GOLD_TICKER, "action": "SELL", "reason": "ROTATION", "target_w": 0.0})
    elif have_gold and abs(cw.get(GOLD_TICKER, 0.0) - gold_w) > C.REBALANCE_BAND:
        orders.append({"ticker": GOLD_TICKER, "action": "REBAL", "reason": "ALLOCATION", "target_w": round(gold_w, 5)})
    if gold_w > 0:
        tw[GOLD_TICKER] = gold_w
    summary = {"cutoff": cutoff, "n_target": n_target, "no_inflation_block": no_inflation, "holds": holds,
               "sells": [s for s, _ in sells], "sell_reasons": {t: w for t, w in sells},
               "buys": buys, "target_weights": {k: round(v, 4) for k, v in tw.items()},
               "cash_target": round(1.0 - sum(tw.values()), 4), "strategy": P.get("name"),
               "equity_frac": round(equity_frac, 3), "gold_w": round(gold_w, 3)}
    return orders, summary


def exposure_from(guard: Dict, regime: Dict) -> float:
    """Long-horizon exposure: fully invested by default; guard mode and regime only trim."""
    mode = str(guard.get("mode", "NORMAL")).upper()
    if guard.get("block_new_entries"):
        return 0.0
    base = C.EXPOSURE_BY_MODE.get(mode, 0.9)
    p_off = regime.get("p_risk_off")
    p_off = 0.5 if p_off is None else float(p_off)
    regime_mult = float(np.clip(1.0 - 0.25 * p_off, C.REGIME_EXPOSURE_FLOOR, 1.0))
    return float(base * regime_mult)


# ====================================================================== V3.8 tranche engine
def _month_diff(a: str, b: str) -> int:
    pa, pb = pd.Period(a, "M"), pd.Period(b, "M")
    return (pb - pa).n


def plan_tranche(pf: Dict, frame: pd.DataFrame, state: Dict, today, today_change: Optional[pd.Series] = None,
                 conf_model: Optional[Dict] = None) -> Tuple[List[Dict], Dict]:
    """Orders for the next open + summary. Cohorts persist in pf["cohorts"]."""
    from confidence import predict, grade
    month = pd.Timestamp(today).strftime("%Y-%m")
    K, N, cap = int(C.TRANCHE_MONTHS), int(C.TRANCHE_N), int(C.TRANCHE_SECTOR_CAP)
    f = frame.copy()
    f["confidence"] = predict(f, conf_model).to_numpy()
    f = f.set_index("ticker")
    for c in ("exp_nominal_12m", "exp_real_12m", "p_beat_all", "hurdle_12m"):
        if c not in f.columns:
            f[c] = np.nan
    cohorts = [c for c in pf.get("cohorts", []) if c.get("month") != month and _month_diff(c["month"], month) < K]
    expired = {t for c in pf.get("cohorts", []) if c.get("month") != month and _month_diff(c["month"], month) >= K
               for t in c.get("tickers", [])}
    # ---- new cohort
    floor = float(state.get("liq_floor_tl") or C.MIN_MEDIAN_VALUE_TRADED_TL)
    elig = pd.to_numeric(f["med_value_traded"], errors="coerce") >= floor
    if today_change is not None:
        elig &= ~(today_change.reindex(f.index) >= C.LIMIT_MOVE_PCT)
    vol = pd.to_numeric(f.get("vol_ann_pct"), errors="coerce")
    if getattr(C, "ENTRY_MAX_LOTTERY_PCTILE", None) and "max_1m" in f:
        elig &= ~(pd.to_numeric(f["max_1m"], errors="coerce").rank(pct=True) > C.ENTRY_MAX_LOTTERY_PCTILE)
    if getattr(C, "ENTRY_MAX_VOL_PCTILE", None) and vol.notna().any():
        elig &= ~(vol.rank(pct=True) > C.ENTRY_MAX_VOL_PCTILE)
    if getattr(C, "ENTRY_MIN_MCAP_PCTILE", None) and "market_cap" in f and pd.to_numeric(f["market_cap"], errors="coerce").notna().mean() > 0.5:
        mc = pd.to_numeric(f["market_cap"], errors="coerce")
        elig &= mc.rank(pct=True).fillna(0) >= C.ENTRY_MIN_MCAP_PCTILE
    if getattr(C, "DEFENSIVE_IN_DOWNTREND", False) and state.get("downtrend"):
        mc = pd.to_numeric(f.get("market_cap"), errors="coerce")
        bt = pd.to_numeric(f.get("beta"), errors="coerce")
        if mc.notna().mean() > 0.5:
            elig &= mc.rank(pct=True).fillna(0) >= 0.5
        if bt.notna().mean() > 0.5:
            elig &= bt.rank(pct=True).fillna(1) <= 0.5
    vt = getattr(C, "VALUE_TRAP_MODE", None)
    rank_col = "composite"
    if vt and "ret_3m" in f and "from_52w_low" in f:
        turn = (pd.to_numeric(f["ret_3m"], errors="coerce").rank(pct=True)
                + pd.to_numeric(f["from_52w_low"], errors="coerce").rank(pct=True)) / 2
        f["turn_pct"] = turn.rank(pct=True)
        if vt == "filter":
            elig &= ~(f["turn_pct"] < getattr(C, "VALUE_TRAP_CUT", 1 / 3))
        elif vt == "blend":
            z = (turn - turn.mean()) / (turn.std() or 1.0)
            cz = (f["composite"] - f["composite"].mean()) / (f["composite"].std() or 1.0)
            f["sel_score"] = cz + 0.5 * z.fillna(0)
            rank_col = "sel_score"
    cand = f[elig].sort_values(rank_col, ascending=False)
    # projected portfolio sector weights from the still-active older cohorts (portfolio-level cap)
    sec_of = lambda t: (f.at[t, "sector"] if t in f.index and isinstance(f.at[t, "sector"], str) and f.at[t, "sector"] else "NA")
    old_live = [c for c in cohorts if c.get("tickers")]
    n_after = len(old_live) + 1
    sec_w: Dict[str, float] = {}
    name_w: Dict[str, float] = {}
    for c in old_live:
        for t in c["tickers"]:
            add = 1.0 / (n_after * len(c["tickers"]))
            sec_w[sec_of(t)] = sec_w.get(sec_of(t), 0.0) + add
            name_w[t] = name_w.get(t, 0.0) + add
    pcap = getattr(C, "PORTFOLIO_SECTOR_CAP", None)
    step = 1.0 / (n_after * N)
    picks, sec_n = [], {}
    for t, r in cand.iterrows():
        sec = sec_of(t)
        if sec != "NA" and sec_n.get(sec, 0) >= cap:
            continue
        if pcap and sec != "NA" and sec_w.get(sec, 0.0) + step > pcap + 1e-9:
            continue
        if name_w.get(t, 0.0) > 0 and name_w[t] + step > C.MAX_NAME_W + 1e-9:
            continue                                   # re-pick would breach the single-name cap -> next idea
        picks.append(t)
        sec_n[sec] = sec_n.get(sec, 0) + 1
        sec_w[sec] = sec_w.get(sec, 0.0) + step
        if len(picks) == N:
            break
    if getattr(C, "WINNER_EXTENSION", False):
        for t in sorted(expired):
            p0 = pf["positions"].get(t)
            if p0 and t in f.index and p0.get("level", 1) >= 1.5 and float(f.at[t, "composite_pct"]) >= 80 and t not in picks:
                picks.append(t)                                    # extend a strong winner into the new cohort
    le = getattr(C, "LOSER_EXTENSION", None)          # V3.12: don't realise a loss at cohort expiry
    extended = []
    if le:
        for t in sorted(expired):
            p0 = pf["positions"].get(t)
            if not p0 or t in picks or t not in f.index or p0.get("level", 1) >= 1.0:
                continue
            age = (pd.Period(month, "M") - pd.Period(str(p0.get("entry_date", month))[:7], "M")).n
            if age < C.TRANCHE_MONTHS + int(le.get("max_extra_months", 6)) and \
                    float(f.at[t, "composite_pct"]) >= float(le.get("min_score_pct", 0)):
                picks.append(t)                                    # keep the loser one more cohort
                extended.append(t)
    conf = {t: round(float(f.at[t, "confidence"]), 4) for t in picks}
    wm = getattr(C, "TRANCHE_WEIGHTING", "equal")
    if picks and wm in ("lottery", "lottery_vol"):
        lot = pd.to_numeric(f.get("max_1m"), errors="coerce").rank(pct=True)
        raw = {}
        for t in picks:
            lr = float(lot.get(t)) if pd.notna(lot.get(t, np.nan)) else 0.5
            r = 1.5 - lr                                             # lottery-like names get less weight
            if wm == "lottery_vol":
                v = float(vol.get(t)) if pd.notna(vol.get(t, np.nan)) else float(vol.median())
                r /= max(v, 10.0) ** 0.5
            raw[t] = r
        sw = sum(raw.values())
        cw_in = {t: v / sw for t, v in raw.items()}
    elif picks and wm == "inv_vol":
        iv = {t: 1.0 / max(float(vol.get(t)) if pd.notna(vol.get(t, np.nan)) else float(vol.median()), 10.0) for t in picks}
        sw = sum(iv.values())
        cw_in = {t: v / sw for t, v in iv.items()}
    else:
        cw_in = {t: 1.0 / len(picks) for t in picks} if picks else {}
    if getattr(C, "ENTRY_MAX_LOTTERY_PCTILE", None):
        pass
    cohorts.append({"month": month, "tickers": picks, "confidence": conf, "w": {t: round(v, 5) for t, v in cw_in.items()}})
    # ---- target weights = union of active cohorts
    live = [c for c in cohorts if c.get("tickers")]
    tw: Dict[str, float] = {}
    for c in live:
        for t in c["tickers"]:
            if (pf.get("stopped") or {}).get(t, "") >= c["month"]:
                continue                                     # stopped out after this cohort bought it
            share = (c.get("w") or {}).get(t, 1.0 / len(c["tickers"]))
            tw[t] = tw.get(t, 0.0) + share / len(live)
    ncap = max(C.MAX_NAME_W, 1.0 / max(len(tw), 1))              # never forces cash when few names exist
    for _ in range(10):                                            # cap and redistribute
        over = {t: w for t, w in tw.items() if w > ncap + 1e-12}
        if not over:
            break
        extra = sum(w - ncap for w in over.values())
        for t in over:
            tw[t] = ncap
        under = [t for t in tw if tw[t] < ncap - 1e-12]
        us = sum(tw[t] for t in under)
        if not under or us <= 0:
            break
        for t in under:
            tw[t] += extra * tw[t] / us
    tot_w = sum(tw.values())
    if tot_w > 0 and abs(tot_w - 1.0) > 1e-9:                     # stock-only: always fully invested
        tw = {t: w / tot_w for t, w in tw.items()}
    # ---- orders
    cw = pf_weights(pf)
    held = list(pf["positions"].keys())
    orders, sells, holds, buys = [], {}, [], []
    for t in held:
        if t not in tw:
            sells[t] = "COHORT_EXPIRY" if t in expired else ("ROTATION" if t == GOLD_TICKER else "NOT_SELECTED")
            orders.append({"ticker": t, "action": "SELL", "reason": sells[t], "target_w": 0.0})
        else:
            holds.append(t)
            if not getattr(C, "TRIM_WINNERS", True) and cw.get(t, 0.0) > tw[t] and cw.get(t, 0.0) - tw[t] <= 0.10:
                continue                                           # let the winner run
            if abs(cw.get(t, 0.0) - tw[t]) > C.REBALANCE_BAND:
                orders.append({"ticker": t, "action": "REBAL", "reason": "COHORT_WEIGHT", "target_w": round(tw[t], 5)})
    # stock-only: idle cash (from exits / below-band drift) is re-deployed into under-weight holdings
    rebal_set = {o["ticker"] for o in orders if o["action"] == "REBAL"}
    cash_w = pf.get("cash", 0.0) / max(pf.get("nav", 1.0) or 1.0, 1e-12)
    spare = cash_w + sum(cw.get(t, 0.0) for t in sells) - sum(w for t, w in tw.items() if t not in pf["positions"]) \
        - sum(tw[o["ticker"]] - cw.get(o["ticker"], 0.0) for o in orders if o["action"] == "REBAL")
    if spare > 0.01:
        gaps = sorted(((tw[t] - cw.get(t, 0.0), t) for t in holds if t not in rebal_set and tw[t] - cw.get(t, 0.0) > 0.005),
                      reverse=True)
        for gap, t in gaps:
            if spare <= 0.005:
                break
            orders.append({"ticker": t, "action": "REBAL", "reason": "CASH_REDEPLOY", "target_w": round(tw[t], 5)})
            spare -= gap
    mode = getattr(C, "STOP_MODE", None)
    lotr = pd.to_numeric(f.get("max_1m"), errors="coerce").rank(pct=True) if "max_1m" in f else pd.Series(dtype=float)
    volr = vol.rank(pct=True)
    def stop_pct(t):
        if not mode or t not in f.index:
            return None
        risky = max(float(volr.get(t, 0) or 0), float(lotr.get(t, 0) or 0)) > C.STOP_HIGH_RISK_PCTILE
        need = mode == "all" or (mode == "low_conf" and float(f.at[t, "confidence"]) < C.STOP_LOW_CONF) or (mode == "high_risk" and risky)
        if not need:
            return None
        v = float(vol.get(t)) if pd.notna(vol.get(t, np.nan)) else float(vol.median())
        return min(C.STOP_ATR_MULT * 1.25 * v / 100 / np.sqrt(252), 0.5)
    for t, w in tw.items():
        if t in pf["positions"]:
            continue
        r = f.loc[t] if t in f.index else None
        buys.append(t)
        orders.append({"ticker": t, "action": "BUY", "reason": "NEW_ENTRY", "target_w": round(w, 5), "stop_pct": stop_pct(t),
                       "entry_pct": None if r is None else round(float(r["composite_pct"]), 2),
                       "entry_conf": None if r is None else round(float(r["confidence"]), 4),
                       "entry_exp_real": None if r is None or pd.isna(r["exp_real_12m"]) else round(float(r["exp_real_12m"]), 2),
                       "entry_exp_nominal": None if r is None or pd.isna(r["exp_nominal_12m"]) else round(float(r["exp_nominal_12m"]), 2),
                       "entry_hurdle": None if r is None or pd.isna(r["hurdle_12m"]) else round(float(r["hurdle_12m"]), 2),
                       "entry_p_beat_all": None if r is None or pd.isna(r["p_beat_all"]) else round(float(r["p_beat_all"]), 3)})
    pf["cohorts"] = cohorts
    # suggested split of NEW money across this month's picks: proportional to the confidence edge
    edge = {t: max(conf[t] - 0.5, 0.02) for t in picks}
    es = sum(edge.values()) or 1.0
    split = {t: round(v / es, 4) for t, v in edge.items()}
    exit_month = {t: max((pd.Period(c["month"], "M") + K).strftime("%Y-%m") for c in cohorts if t in c["tickers"])
                  for t in tw}
    nxt = (pd.Period(month, "M") + 1).strftime("%Y-%m")
    expiring_next = sorted(t for t, m in exit_month.items() if m == nxt)
    summary = {"engine": "tranche", "month": month, "expiring_next": expiring_next, "extended": extended,
               "picks": [t for t in picks if t not in extended], "confidence": conf,
               "confidence_grade": {t: grade(conf[t]) for t in picks}, "suggested_split": split,
               "buys": buys, "sells": list(sells), "sell_reasons": sells, "holds": holds,
               "target_weights": {k: round(v, 4) for k, v in tw.items()}, "exit_month": exit_month,
               "n_cohorts": len(live), "cash_target": round(1.0 - sum(tw.values()), 4)}
    return orders, summary
