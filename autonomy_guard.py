"""Autonomy guard: drift / performance / operations / regime stress -> exposure policy.

Fixes of V1:
* Streak counters are CONSECUTIVE (reset when the condition clears); V1 only
  ever incremented them, so persistence requirements silently disappeared.
* The self-test drives the production `transition()` function itself (V1
  tested a separate copy of the logic, i.e. it could never fail).
* Drift is measured on RAW features (z-scores are N(0,1) by construction).
* Regime stress comes from the HMM forecast (P(risk-off) and state
  instability), not from a label that flipped every day.
* Performance drift (V3, long horizon): live 3-month IC of the recorded
  composite, portfolio drawdown and 6-month performance versus XU100.
* In V3 the guard only scales/blocks NEW buys; it never forces selling.
Failure anywhere -> SAFE (no new entries), never a crash.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
import pandas as pd

VERSION = "2.0.0"

MODE_POLICY = {
    "NORMAL": {"exposure_multiplier": 1.00, "pct_cutoff_add": 0.0, "block_new_entries": False},
    "WATCH": {"exposure_multiplier": 0.60, "pct_cutoff_add": 3.0, "block_new_entries": False},
    "RECOVERY": {"exposure_multiplier": 0.35, "pct_cutoff_add": 5.0, "block_new_entries": False},
    "SAFE": {"exposure_multiplier": 0.00, "pct_cutoff_add": 100.0, "block_new_entries": True},
}

DEFAULT_GUARD = {
    "version": VERSION, "mode": "WATCH", "watch_streak": 0, "severe_streak": 0, "healthy_streak": 0,
    "feature_baseline": {}, "reason": "BOOTSTRAP", "last_transition": None, "last_evaluation": None,
}

DRIFT_FEATURES = ["change_pct", "log_value_traded"]


# ------------------------------------------------------------------ helpers
def _num(v, d=0.0):
    try:
        x = float(v)
        return x if np.isfinite(x) else d
    except Exception:
        return d


def _robust(s: pd.Series) -> Optional[Dict]:
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 10:
        return None
    q25, med, q75 = np.percentile(s, [25, 50, 75])
    return {"median": float(med), "q25": float(q25), "q75": float(q75),
            "mad": float(np.median(np.abs(s - med))), "n": int(len(s))}


def _drift(ref: Dict, cur: Dict) -> float:
    scale = max(ref["mad"] * 1.4826, (ref["q75"] - ref["q25"]) / 1.349, 1e-6)
    shift = min(abs(cur["median"] - ref["median"]) / scale / 3.0, 1.0)
    disp = min(abs(np.log(max(cur["q75"] - cur["q25"], 1e-9) / max(ref["q75"] - ref["q25"], 1e-9))) / 1.5, 1.0)
    return float(np.clip(0.7 * shift + 0.3 * disp, 0, 1))


def _ewma(ref: Optional[Dict], cur: Dict, a: float = 0.05) -> Dict:
    if not ref:
        return dict(cur)
    return {k: (1 - a) * ref[k] + a * cur[k] if k != "n" else cur[k] for k in cur}


def feature_drift(features: Optional[pd.DataFrame], baseline: Dict, mode: str) -> float:
    if features is None or features.empty:
        return 0.0
    f = features.copy()
    if "value_traded" in f:
        f["log_value_traded"] = np.log1p(pd.to_numeric(f["value_traded"], errors="coerce").clip(lower=0))
    drifts = []
    for col in DRIFT_FEATURES:
        if col not in f:
            continue
        cur = _robust(f[col])
        if not cur:
            continue
        ref = baseline.get(col)
        d = _drift(ref, cur) if ref else 0.0
        if ref:
            drifts.append(d)
        if mode != "SAFE" and d < 0.35:
            baseline[col] = _ewma(ref, cur)
    if not drifts:
        return 0.0
    return float(np.clip(0.7 * max(drifts) + 0.3 * np.mean(drifts), 0, 1))


def performance_drift(live_ic: Optional[Dict], nav_df: Optional[pd.DataFrame]) -> (float, Dict):
    """Long-horizon health: 3-month live IC of the recorded composite, portfolio
    drawdown and 6-month performance relative to XU100."""
    detail, parts = {}, [0.0]
    if live_ic and live_ic.get("n_dates", 0) >= 12 and live_ic.get("ic_recent") is not None:
        r = float(live_ic["ic_recent"])
        parts.append(float(np.clip(-r / 0.08, 0, 1)))
        detail["ic3m_recent"] = r
    if nav_df is not None and len(nav_df) >= 20:
        nv = pd.to_numeric(nav_df["nav"], errors="coerce").dropna()
        dd = float((nv.iloc[-1] / nv.cummax().iloc[-1] - 1) * 100)
        detail["nav_drawdown_pct"] = round(dd, 2)
        parts.append(0.85 if dd <= -25 else 0.5 if dd <= -15 else 0.0)
        if "xu100" in nav_df and len(nav_df) >= 126:
            x = pd.to_numeric(nav_df["xu100"], errors="coerce").ffill()
            if x.notna().iloc[-126] and x.notna().iloc[-1]:
                rel = (nv.iloc[-1] / nv.iloc[-126] - x.iloc[-1] / x.iloc[-126]) * 100
                detail["rel_vs_xu100_6m_pct"] = round(float(rel), 2)
                parts.append(0.5 if rel <= -15 else 0.0)
    return float(max(parts)), detail


def regime_stress(regime: Dict) -> float:
    p_off = _num(regime.get("p_risk_off"), 0.5)
    hist = regime.get("argmax_history") or []
    flips = sum(1 for a, b in zip(hist[-6:-1], hist[-5:]) if a != b)
    instability = min(flips / 3.0, 1.0)
    base = max(0.9 * p_off, instability)
    if regime.get("degraded", True):
        base = max(base, 0.5)
    return float(np.clip(base, 0, 1))


def ops_score(data_quality: float, rows: Optional[int], min_rows: int = 30) -> float:
    dq = float(np.clip(_num(data_quality, 0.0), 0, 100))
    rh = 100.0 if rows is None else 100.0 * float(np.clip(_num(rows) / max(min_rows, 1), 0, 1))
    return round(0.75 * dq + 0.25 * rh, 2)


# ------------------------------------------------------------------ state machine (tested)
def classify(drift: float, perf: float, ops: float, reg: float) -> Dict[str, bool]:
    severe = ops < 50 or drift >= 0.75 or perf >= 0.80 or reg >= 0.90
    watch = severe or ops < 75 or drift >= 0.45 or perf >= 0.45 or reg >= 0.60
    healthy = (not watch) and ops >= 85 and drift < 0.30 and perf < 0.35 and reg < 0.50
    return {"severe": severe, "watch": watch, "healthy": healthy, "hard_stop": ops < 40}


def transition(guard: Dict, flags: Dict[str, bool]) -> Dict:
    g = guard
    g["severe_streak"] = g.get("severe_streak", 0) + 1 if flags["severe"] else 0
    g["watch_streak"] = g.get("watch_streak", 0) + 1 if flags["watch"] else 0
    g["healthy_streak"] = g.get("healthy_streak", 0) + 1 if flags["healthy"] else 0
    mode = g.get("mode", "WATCH")
    if flags.get("hard_stop"):
        mode = "SAFE"
    elif mode == "NORMAL":
        if g["severe_streak"] >= 2:
            mode = "SAFE"
        elif flags["severe"] or g["watch_streak"] >= 2:
            mode = "WATCH"
    elif mode == "WATCH":
        if g["severe_streak"] >= 2:
            mode = "SAFE"
        elif g["healthy_streak"] >= 2:
            mode = "NORMAL"
    elif mode == "SAFE":
        if g["healthy_streak"] >= 3:
            mode = "RECOVERY"
    elif mode == "RECOVERY":
        if flags["severe"]:
            mode = "SAFE"
        elif g["healthy_streak"] >= 6:
            mode = "NORMAL"
    g["mode"] = mode
    return g


def run_self_test() -> Dict:
    H = {"severe": False, "watch": False, "healthy": True, "hard_stop": False}
    W = {"severe": False, "watch": True, "healthy": False, "hard_stop": False}
    S = {"severe": True, "watch": True, "healthy": False, "hard_stop": False}
    X = {"severe": True, "watch": True, "healthy": False, "hard_stop": True}
    cases = [
        ("NORMAL", [W], "NORMAL"),              # single watch day does not downgrade
        ("NORMAL", [W, W], "WATCH"),
        ("NORMAL", [S], "WATCH"),
        ("NORMAL", [S, S], "SAFE"),
        ("NORMAL", [S, H, S], "WATCH"),         # non-consecutive severe != SAFE (V1 bug)
        ("WATCH", [H, H], "NORMAL"),
        ("WATCH", [H, W, H], "WATCH"),          # healthy streak resets
        ("SAFE", [H, H], "SAFE"),
        ("SAFE", [H, H, H], "RECOVERY"),
        ("RECOVERY", [S], "SAFE"),
        ("RECOVERY", [H] * 6, "NORMAL"),
        ("NORMAL", [X], "SAFE"),
    ]
    passed = 0
    for start, seq, expect in cases:
        g = {"mode": start}
        for f in seq:
            g = transition(g, f)
        passed += int(g["mode"] == expect)
    # classification sanity on the production thresholds
    passed_cls = int(classify(0.1, 0.1, 98, 0.1)["healthy"]) + int(classify(0.9, 0, 98, 0)["severe"]) \
        + int(classify(0, 0, 45, 0)["severe"]) + int(not classify(0.5, 0, 98, 0)["healthy"])
    ok = passed == len(cases) and passed_cls == 4
    return {"passed": bool(ok), "cases": len(cases) + 4, "passed_cases": passed + passed_cls,
            "timestamp": datetime.now(timezone.utc).isoformat()}


# ------------------------------------------------------------------ public API
def evaluate_guard(state: Dict, *, features: Optional[pd.DataFrame], data_quality: float, rows: Optional[int],
                   live_ic: Optional[Dict], nav_df: Optional[pd.DataFrame]) -> Dict:
    try:
        g = deepcopy(DEFAULT_GUARD)
        g.update(state.get("autonomy_guard", {}) or {})
        test = run_self_test()
        g["self_test"] = test
        baseline = g.get("feature_baseline", {}) or {}
        drift = feature_drift(features, baseline, g.get("mode", "WATCH"))
        perf, perf_detail = performance_drift(live_ic, nav_df)
        ops = ops_score(data_quality, rows)
        reg = regime_stress(state.get("regime", {}))
        flags = classify(drift, perf, ops, reg)
        if not test["passed"]:
            flags = {"severe": True, "watch": True, "healthy": False, "hard_stop": True}
        old = g.get("mode", "WATCH")
        g = transition(g, flags)
        pol = MODE_POLICY[g["mode"]]
        reasons = [n for n, v in (("FEATURE_DRIFT", drift >= 0.45), ("PERFORMANCE_DRIFT", perf >= 0.45),
                                  ("OPS_DATA_QUALITY", ops < 75), ("REGIME_STRESS", reg >= 0.60),
                                  ("SELF_TEST_FAILED", not test["passed"])) if v]
        now = datetime.now(timezone.utc).isoformat()
        g.update({
            "feature_baseline": baseline, "drift_score": round(drift, 4), "performance_drift": round(perf, 4),
            "performance_detail": perf_detail, "ops_score": ops, "regime_stress": round(reg, 4),
            "reason": "+".join(reasons) if reasons else "HEALTHY", "last_evaluation": now, **pol,
        })
        if g["mode"] != old:
            g["last_transition"] = now
        state["autonomy_guard"] = g
        return {"mode": g["mode"], **pol, "reason": g["reason"]}
    except Exception as exc:
        pol = MODE_POLICY["SAFE"]
        g = state.get("autonomy_guard", {}) or {}
        g.update({"mode": "SAFE", **pol, "reason": f"GUARD_ERROR:{type(exc).__name__}",
                  "last_evaluation": datetime.now(timezone.utc).isoformat()})
        state["autonomy_guard"] = g
        return {"mode": "SAFE", **pol, "reason": g["reason"]}
