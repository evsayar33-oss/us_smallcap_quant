"""
US Small-Cap Quant - Kur-Unut V1 Autonomy Guard

Layers:
1) Regime stress classification
2) Feature-distribution drift detection (PSI)
3) Outcome/performance drift detection
4) Safe-mode state machine: NORMAL -> WATCH -> SAFE -> RECOVERY -> NORMAL
5) Deterministic internal stress-test (synthetic scenarios are TEST ONLY; never production data)
6) Fail-closed data health gate

The guard does not replace the existing quant scorer. In NORMAL mode the existing
selection logic is preserved. WATCH raises the effective entry threshold; SAFE
blocks new entries; RECOVERY re-opens gradually after clean observations.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


STATE_KEY = "autonomy_guard"
GUARD_VERSION = "1.0.0"

DEFAULTS = {
    "state": "NORMAL",
    "recovery_days": 0,
    "watch_days": 0,
    "safe_days": 0,
    "clean_days": 0,
    "last_update": None,
    "last_reason": "BOOTSTRAP",
    "exposure_multiplier": 1.0,
    "threshold_add": 0.0,
    "regime": "NORMAL",
    "regime_confidence": 0.0,
    "drift": {},
    "performance": {},
    "data_health": {},
    "stress_test": {},
}


def _safe_float(value, default=np.nan) -> float:
    try:
        x = float(value)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _ensure_guard(state: Dict) -> Dict:
    guard = state.setdefault(STATE_KEY, {})
    for k, v in DEFAULTS.items():
        if k not in guard:
            guard[k] = {} if isinstance(v, dict) else v
    guard["version"] = GUARD_VERSION
    return guard


def _series(df: pd.DataFrame, name: str) -> pd.Series:
    if df is None or df.empty or name not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[name], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def _quantile_bins(reference: np.ndarray, n_bins: int = 10) -> np.ndarray:
    q = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(reference, q)
    edges = np.unique(edges)
    if len(edges) < 3:
        lo, hi = float(np.min(reference)), float(np.max(reference))
        if not np.isfinite(lo) or not np.isfinite(hi) or math.isclose(lo, hi):
            return np.array([-np.inf, np.inf], dtype=float)
        edges = np.linspace(lo, hi, min(5, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges.astype(float)


def psi(reference: Iterable[float], current: Iterable[float], n_bins: int = 10) -> float:
    """Population Stability Index. Reference bins are fixed from baseline data."""
    ref = np.asarray([_safe_float(x) for x in reference], dtype=float)
    cur = np.asarray([_safe_float(x) for x in current], dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) < 20 or len(cur) < 5:
        return 0.0
    edges = _quantile_bins(ref, n_bins=n_bins)
    rp, _ = np.histogram(ref, bins=edges)
    cp, _ = np.histogram(cur, bins=edges)
    rp = rp.astype(float) / max(len(ref), 1)
    cp = cp.astype(float) / max(len(cur), 1)
    eps = 1e-4
    rp = np.clip(rp, eps, None)
    cp = np.clip(cp, eps, None)
    return float(np.sum((cp - rp) * np.log(cp / rp)))


def detect_regime(df: pd.DataFrame) -> Tuple[str, float, Dict]:
    """Small-cap specific regime context using breadth, performance, dispersion and participation."""
    if df is None or df.empty:
        return "DATA_STRESS", 1.0, {"reason": "empty"}

    daily = _series(df, "change_%")
    monthly = _series(df, "perf_1m")
    rvol = _series(df, "rvol")

    if daily.empty:
        return "DATA_STRESS", 1.0, {"reason": "missing_change"}

    green = float((daily > 0).mean())
    red = float((daily < 0).mean())
    median_daily = float(daily.median())
    median_monthly = float(monthly.median()) if not monthly.empty else 0.0
    dispersion = float(daily.std(ddof=0)) if len(daily) > 1 else 0.0
    high_rvol = float((rvol >= 1.5).mean()) if not rvol.empty else 0.0

    if red >= 0.80 and median_daily <= -3.0:
        regime = "PANIC"
        confidence = min(0.99, 0.55 + red * 0.35)
    elif red >= 0.65 and median_daily <= -1.25:
        regime = "BROAD_SELL_OFF"
        confidence = min(0.95, 0.50 + red * 0.30)
    elif dispersion >= 5.0 and 0.35 <= red <= 0.65:
        regime = "ROTATION"
        confidence = 0.68
    elif green >= 0.68 and median_monthly >= 2.0:
        regime = "EXPANSION"
        confidence = min(0.95, 0.50 + green * 0.35)
    elif green >= 0.55 and 0.35 <= median_daily and -5.0 <= median_monthly < 1.0:
        regime = "RECOVERY"
        confidence = 0.70
    elif dispersion <= 1.25 and abs(median_daily) <= 0.35:
        regime = "QUIET"
        confidence = 0.72
    elif median_monthly < -3.0 or red >= 0.55:
        regime = "STRESS"
        confidence = 0.72
    elif green >= 0.55 and median_monthly >= 0.0:
        regime = "BULL_SELECTIVE"
        confidence = 0.66
    else:
        regime = "NORMAL"
        confidence = 0.55

    return regime, float(confidence), {
        "breadth_up": round(green, 4),
        "breadth_down": round(red, 4),
        "median_daily": round(median_daily, 3),
        "median_monthly": round(median_monthly, 3),
        "dispersion": round(dispersion, 3),
        "high_rvol_share": round(high_rvol, 4),
    }


def compute_feature_drift(history: pd.DataFrame, current: pd.DataFrame, lookback_days: int = 60) -> Dict:
    """Compare pooled recent historical distributions with the current cross-section."""
    if history is None or history.empty or current is None or current.empty:
        return {"psi_max": 0.0, "psi_mean": 0.0, "features": {}, "status": "NO_REFERENCE"}

    h = history.copy()
    if "tarih" in h.columns:
        h["tarih"] = pd.to_datetime(h["tarih"], errors="coerce").dt.normalize()
        dates = sorted(h["tarih"].dropna().unique())
        if dates:
            cutoff = dates[-1] - pd.Timedelta(days=lookback_days)
            h = h[h["tarih"] >= cutoff]

    features = [
        "rvol", "perf_w", "perf_1m", "perf_y",
        "score_base", "score_quality", "score_flow", "score_ignition",
        "quant_score",
    ]
    results = {}
    for feature in features:
        ref = _series(h, feature)
        cur = _series(current, feature)
        if len(ref) >= 20 and len(cur) >= 5:
            results[feature] = round(psi(ref.values, cur.values), 4)

    if not results:
        return {"psi_max": 0.0, "psi_mean": 0.0, "features": {}, "status": "NO_FEATURES"}

    vals = np.array(list(results.values()), dtype=float)
    return {
        "psi_max": round(float(vals.max()), 4),
        "psi_mean": round(float(vals.mean()), 4),
        "features": results,
        "status": "OK",
    }


def _mature_performance(lifecycle: pd.DataFrame) -> Dict:
    if lifecycle is None or lifecycle.empty:
        return {"n": 0, "win_rate": 0.0, "avg_return": 0.0, "pf": 0.0}

    # Prefer actual 30/90/180-day outcomes when available. Fallback to peak/terminal outcome.
    df = lifecycle.copy()
    ret_col = None
    for c in ("ret_30d", "ret_90d"):
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().sum() >= 10:
            ret_col = c
            break

    if ret_col is not None:
        ret = pd.to_numeric(df[ret_col], errors="coerce").dropna()
    else:
        mature = df[df.get("outcome", pd.Series(index=df.index, dtype=object)).astype(str).str.startswith("WIN") |
                    df.get("outcome", pd.Series(index=df.index, dtype=object)).astype(str).str.startswith("FAIL") |
                    df.get("outcome", pd.Series(index=df.index, dtype=object)).astype(str).str.startswith("TIMEOUT")]
        ret = pd.to_numeric(mature.get("peak_gain", pd.Series(dtype=float)), errors="coerce").dropna()

    if ret.empty:
        return {"n": 0, "win_rate": 0.0, "avg_return": 0.0, "pf": 0.0}

    wins = ret[ret > 0]
    losses = ret[ret <= 0]
    gp = float(wins.sum())
    gl = float(abs(losses.sum()))
    pf = gp / gl if gl > 0 else 5.0
    return {
        "n": int(len(ret)),
        "win_rate": round(float((ret > 0).mean() * 100), 1),
        "avg_return": round(float(ret.mean()), 3),
        "pf": round(float(pf), 3),
    }


def compute_performance_drift(lifecycle: pd.DataFrame) -> Dict:
    if lifecycle is None or lifecycle.empty or "tarih" not in lifecycle.columns:
        return {"recent": {}, "baseline": {}, "status": "NO_DATA", "severity": 0}

    df = lifecycle.copy()
    df["tarih"] = pd.to_datetime(df["tarih"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["tarih"])
    if df.empty:
        return {"recent": {}, "baseline": {}, "status": "NO_DATA", "severity": 0}

    dates = sorted(df["tarih"].unique())
    recent_cut = dates[-1] - pd.Timedelta(days=120)
    base_cut = dates[-1] - pd.Timedelta(days=365)

    recent = _mature_performance(df[df["tarih"] >= recent_cut])
    baseline = _mature_performance(df[(df["tarih"] >= base_cut) & (df["tarih"] < recent_cut)])

    severity = 0
    if recent["n"] >= 15:
        if recent["pf"] < 0.75 or (recent["win_rate"] < 35 and recent["avg_return"] < 0):
            severity = 2
        elif recent["pf"] < 0.90 or (recent["win_rate"] < 40 and recent["avg_return"] < 0):
            severity = 1

    if baseline["n"] >= 20 and recent["n"] >= 15:
        if recent["pf"] < baseline["pf"] * 0.60 and recent["avg_return"] < baseline["avg_return"]:
            severity = max(severity, 2)
        elif recent["pf"] < baseline["pf"] * 0.80:
            severity = max(severity, 1)

    return {
        "recent": recent,
        "baseline": baseline,
        "status": "OK" if severity == 0 else ("WATCH" if severity == 1 else "SEVERE"),
        "severity": severity,
    }


def data_health(df: pd.DataFrame) -> Dict:
    if df is None or df.empty:
        return {"ok": False, "severity": 2, "reason": "empty_current"}

    required = ["ticker", "close", "open", "high", "low", "volume", "change_%", "value_traded", "rvol"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return {"ok": False, "severity": 2, "reason": "missing:" + ",".join(missing)}

    bad = pd.Series(False, index=df.index)
    for c in required[1:]:
        if c == "ticker":
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        bad |= ~np.isfinite(s.to_numpy(dtype=float, na_value=np.nan))

    if {"close", "open", "high", "low"}.issubset(df.columns):
        o = pd.to_numeric(df["open"], errors="coerce").to_numpy(dtype=float)
        h = pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype=float)
        l = pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype=float)
        c = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
        bad |= (o <= 0) | (h <= 0) | (l <= 0) | (c <= 0) | (h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (h < l)

    bad_rate = float(bad.mean()) if len(bad) else 1.0
    severity = 2 if bad_rate >= 0.10 else (1 if bad_rate >= 0.03 else 0)

    return {
        "ok": bool(len(df) >= 50 and bad_rate < 0.10),
        "severity": severity,
        "rows": int(len(df)),
        "bad_rows": int(bad.sum()),
        "bad_rate": round(bad_rate, 4),
    }


def _scenario_frame(kind: str, n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(12345)
    if kind == "PANIC":
        change = rng.normal(-3.5, 2.0, n)
        perf_1m = rng.normal(-10, 4, n)
        rvol = rng.uniform(1.5, 3.5, n)
    elif kind == "EXPANSION":
        change = rng.normal(1.0, 1.0, n)
        perf_1m = rng.normal(5, 3, n)
        rvol = rng.uniform(0.8, 2.5, n)
    elif kind == "ROTATION":
        change = rng.normal(0.0, 6.0, n)
        perf_1m = rng.normal(0, 5, n)
        rvol = rng.uniform(0.7, 2.8, n)
    elif kind == "QUIET":
        change = rng.normal(0.0, 0.4, n)
        perf_1m = rng.normal(0.2, 1.0, n)
        rvol = rng.uniform(0.7, 1.3, n)
    elif kind == "RECOVERY":
        change = rng.normal(0.7, 0.35, n)
        perf_1m = rng.normal(-1.5, 1.2, n)
        rvol = rng.uniform(1.0, 2.4, n)
    elif kind == "STRESS":
        down = rng.normal(-0.9, 0.45, int(n * 0.58))
        up = rng.normal(0.45, 0.35, n - len(down))
        change = np.concatenate([down, up])
        rng.shuffle(change)
        perf_1m = rng.normal(-4.5, 1.2, n)
        rvol = rng.uniform(1.2, 2.5, n)
    else:
        change = rng.normal(0.1, 1.4, n)
        perf_1m = rng.normal(0, 3, n)
        rvol = rng.uniform(0.8, 1.8, n)
    close = np.maximum(1, 50 * np.exp(np.cumsum(rng.normal(0, 0.01, n))))
    high = close * (1 + rng.uniform(0, 0.02, n))
    low = close * (1 - rng.uniform(0, 0.02, n))
    return pd.DataFrame({
        "ticker": [f"T{i:03d}" for i in range(n)],
        "close": close, "open": close, "high": high, "low": low,
        "volume": rng.integers(100000, 500000, n),
        "change_%": change, "value_traded": close * 200000,
        "rvol": rvol, "perf_w": change * 2.0, "perf_1m": perf_1m, "perf_y": perf_1m * 3,
    })


def run_stress_test() -> Dict:
    """Deterministic health test only. It never writes market facts and is not a backtest."""
    expected = {
        "NORMAL": {"NORMAL", "BULL_SELECTIVE"},
        "EXPANSION": {"EXPANSION"},
        "ROTATION": {"ROTATION"},
        "PANIC": {"PANIC"},
        "QUIET": {"QUIET"},
        "RECOVERY": {"RECOVERY"},
        "STRESS": {"STRESS"},
    }
    results = {}
    for case, allowed in expected.items():
        frame = _scenario_frame(case)
        regime, conf, stats = detect_regime(frame)
        results[case] = {
            "ok": bool(regime in allowed and 0.0 <= conf <= 1.0),
            "expected": sorted(allowed),
            "detected": regime,
            "confidence": round(conf, 3),
            "rows": len(frame),
        }
    passed = sum(1 for x in results.values() if x["ok"])
    return {
        "passed": passed == len(results),
        "passed_cases": passed,
        "total_cases": len(results),
        "results": results,
    }


def decide_guard(
    state: Dict,
    current_df: pd.DataFrame,
    history_df: pd.DataFrame | None = None,
    lifecycle_df: pd.DataFrame | None = None,
    run_test: bool = True,
) -> Tuple[Dict, Dict]:
    """Update guard state and return (state, decision)."""
    guard = _ensure_guard(state)
    history_df = history_df if history_df is not None else pd.DataFrame()
    lifecycle_df = lifecycle_df if lifecycle_df is not None else pd.DataFrame()

    regime, regime_conf, regime_stats = detect_regime(current_df)
    drift = compute_feature_drift(history_df, current_df)
    performance = compute_performance_drift(lifecycle_df)
    health = data_health(current_df)
    stress = run_stress_test() if run_test else guard.get("stress_test", {})

    psi_max = float(drift.get("psi_max", 0.0))
    drift_severity = 2 if psi_max >= 0.35 else (1 if psi_max >= 0.20 else 0)
    regime_severe = regime in ("PANIC", "BROAD_SELL_OFF") and regime_conf >= 0.78
    regime_watch = regime in ("PANIC", "BROAD_SELL_OFF", "STRESS") and regime_conf >= 0.65

    severe = 0
    reasons = []
    if not health.get("ok", False):
        severe += 2
        reasons.append("DATA_HEALTH")
    if drift_severity == 2:
        severe += 2
        reasons.append("FEATURE_DRIFT_SEVERE")
    elif drift_severity == 1:
        severe += 1
        reasons.append("FEATURE_DRIFT_WATCH")
    if int(performance.get("severity", 0)) == 2:
        severe += 2
        reasons.append("PERFORMANCE_DRIFT_SEVERE")
    elif int(performance.get("severity", 0)) == 1:
        severe += 1
        reasons.append("PERFORMANCE_DRIFT_WATCH")
    if regime_severe:
        severe += 2
        reasons.append("REGIME_STRESS")
    elif regime_watch:
        severe += 1
        reasons.append("REGIME_WATCH")
    if run_test and not stress.get("passed", False):
        severe += 3
        reasons.append("STRESS_TEST_FAIL")

    current = str(guard.get("state", "NORMAL")).upper()
    recovery_days = int(guard.get("recovery_days", 0) or 0)

    if severe >= 4:
        new_state = "SAFE"
        recovery_days = 0
    elif current == "SAFE":
        if severe == 0:
            recovery_days += 1
            new_state = "RECOVERY" if recovery_days >= 3 else "SAFE"
        else:
            recovery_days = 0
            new_state = "SAFE"
    elif current == "RECOVERY":
        if severe == 0:
            recovery_days += 1
            new_state = "NORMAL" if recovery_days >= 5 else "RECOVERY"
        elif severe >= 2:
            recovery_days = 0
            new_state = "SAFE"
        else:
            recovery_days = 0
            new_state = "RECOVERY"
    elif severe >= 2:
        new_state = "WATCH"
        recovery_days = 0
    elif severe == 1:
        new_state = "WATCH"
        recovery_days = 0
    else:
        new_state = "NORMAL"
        recovery_days = 0

    if new_state == "SAFE":
        exposure = 0.0
        threshold_add = 999.0
    elif new_state == "WATCH":
        exposure = 0.60
        threshold_add = 10.0
    elif new_state == "RECOVERY":
        exposure = 0.40
        threshold_add = 7.0
    else:
        exposure = 1.0
        threshold_add = 0.0

    # In broad panic, even without other issues, SAFE is deliberately fail-closed.
    if regime_severe and new_state != "SAFE":
        new_state = "SAFE"
        exposure = 0.0
        threshold_add = 999.0
        recovery_days = 0
        reasons.append("PANIC_FAIL_CLOSED")

    guard.update({
        "version": GUARD_VERSION,
        "state": new_state,
        "recovery_days": recovery_days,
        "last_update": _now_iso(),
        "last_reason": ",".join(reasons) if reasons else "HEALTHY",
        "exposure_multiplier": exposure,
        "threshold_add": threshold_add,
        "regime": regime,
        "regime_confidence": round(regime_conf, 3),
        "regime_stats": regime_stats,
        "drift": drift,
        "performance": performance,
        "data_health": health,
        "stress_test": stress,
    })

    decision = {
        "state": new_state,
        "entry_allowed": new_state != "SAFE",
        "exposure_multiplier": exposure,
        "threshold_add": threshold_add,
        "regime": regime,
        "regime_confidence": regime_conf,
        "reason": guard["last_reason"],
    }
    return state, decision


def apply_to_scores(scored_df: pd.DataFrame, decision: Dict) -> pd.DataFrame:
    """Preserve normal scoring; only alter eligibility/ranking threshold behavior under guard modes."""
    if scored_df is None or scored_df.empty:
        return scored_df
    out = scored_df.copy()
    mode = str(decision.get("state", "NORMAL")).upper()
    add = float(decision.get("threshold_add", 0.0))

    # Keep a traceable pre-guard score.
    if "quant_score" in out.columns and "quant_score_pre_guard" not in out.columns:
        out["quant_score_pre_guard"] = out["quant_score"]

    out["autonomy_guard_state"] = mode
    out["autonomy_exposure_multiplier"] = float(decision.get("exposure_multiplier", 1.0))
    out["autonomy_guard_reason"] = str(decision.get("reason", ""))

    if mode == "SAFE":
        out["autonomy_entry_blocked"] = True
        return out
    out["autonomy_entry_blocked"] = False

    if add > 0 and "quant_score" in out.columns:
        out["autonomy_effective_threshold"] = 65.0 + add
    else:
        out["autonomy_effective_threshold"] = 65.0
    return out


def update_performance_only(state: Dict, lifecycle_df: pd.DataFrame) -> Dict:
    guard = _ensure_guard(state)
    performance = compute_performance_drift(lifecycle_df)
    severity = int(performance.get("severity", 0))
    current = str(guard.get("state", "NORMAL")).upper()
    if severity >= 2:
        guard.update({
            "state": "SAFE",
            "exposure_multiplier": 0.0,
            "threshold_add": 999.0,
            "recovery_days": 0,
            "last_update": _now_iso(),
            "last_reason": "PERFORMANCE_DRIFT_SEVERE",
            "performance": performance,
        })
    elif severity == 1 and current == "NORMAL":
        guard.update({
            "state": "WATCH",
            "exposure_multiplier": 0.60,
            "threshold_add": 10.0,
            "recovery_days": 0,
            "last_update": _now_iso(),
            "last_reason": "PERFORMANCE_DRIFT_WATCH",
            "performance": performance,
        })
    else:
        guard["performance"] = performance
    return state


def guard_summary(state: Dict) -> str:
    guard = _ensure_guard(state)
    return (
        f"{guard.get('state', 'NORMAL')} | "
        f"Rejim={guard.get('regime', 'NORMAL')} "
        f"Güven={float(guard.get('regime_confidence', 0))*100:.0f}% | "
        f"PSI={float(guard.get('drift', {}).get('psi_max', 0.0)):.3f} | "
        f"PF(recent)={float(guard.get('performance', {}).get('recent', {}).get('pf', 0.0)):.2f}"
    )


if __name__ == "__main__":
    report = run_stress_test()
    print(report)
    if not report["passed"]:
        raise SystemExit(1)
