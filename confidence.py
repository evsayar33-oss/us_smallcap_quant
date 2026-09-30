"""Calibrated confidence ("güven oranı") for every stock the engine proposes.

Definition (what the number means):
    confidence = P( the stock's 12-month return beats the MEDIAN BIST stock's 12-month return )

Why relative and not "beats CPI / USD / gold": research on the real 581-stock panel showed that
whether ANY stock beats the benchmarks in a given year is ~30% driven by the year itself (2022: 71% of
stocks did, 2024: 15%). That market component cannot be forecast reliably with ~12 independent years
(walk-forward Brier skill was negative). For deciding HOW MUCH to put into each stock, only the
relative edge matters — the market component is common to every stock. The relative probability
was well calibrated out-of-sample (predicted 0.60 -> realised 0.61 for the monthly top 10).

Model: L2-regularised logistic regression (plain numpy, no extra dependency) on
    pct      score percentile - 0.5          (the ranking)
    pct3     (pct)^3                          (extra conviction at the extremes)
    agree    share of core factors pointing the same way - 0.5  (value + defensive agreement)
    lottery  cross-sectional rank of the biggest 1-month daily jump - 0.5 (lottery stocks disappoint)
    fundgap  share of missing value fundamentals (less information -> less confidence)
Fitted in the walk-forward backtest on OUT-OF-SAMPLE scores only; the live engine applies the stored
coefficients. A reliability table (predicted vs realised) is stored with the model.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

KEY_FACTORS = ["z_book_yield", "z_sales_yield", "z_earnings_yield", "z_low_vol", "z_dd_resilience"]
VALUE_FACTORS = ["z_book_yield", "z_sales_yield", "z_earnings_yield"]
FEATURES = ["pct", "pct3", "agree", "lottery", "fundgap"]
# fallback when no fitted model exists yet (research fit, 2017-2025 walk-forward, rounded)
DEFAULT_MODEL = {"intercept": -0.0069, "coef": {"pct": 0.186, "pct3": 2.531, "agree": 0.447, "lottery": -0.209, "fundgap": 0.0},
                 "source": "research_default_2019_2026", "oos_top10_predicted": 0.625, "oos_top10_realised": 0.611}


def features(frame: pd.DataFrame) -> pd.DataFrame:
    """frame: one cross-section (one date) with composite_pct (0-100), z_* factors and max_1m."""
    f = pd.DataFrame(index=frame.index)
    pct = pd.to_numeric(frame.get("composite_pct"), errors="coerce") / 100.0
    f["pct"] = pct - 0.5
    f["pct3"] = f["pct"] ** 3
    have = [c for c in KEY_FACTORS if c in frame.columns]
    if have:
        z = frame[have].apply(pd.to_numeric, errors="coerce")
        f["agree"] = (z > 0).sum(axis=1) / len(have) - 0.5
    else:
        f["agree"] = 0.0
    if "max_1m" in frame.columns and pd.to_numeric(frame["max_1m"], errors="coerce").notna().any():
        f["lottery"] = pd.to_numeric(frame["max_1m"], errors="coerce").rank(pct=True) - 0.5
    else:
        f["lottery"] = 0.0
    hv = [c for c in VALUE_FACTORS if c in frame.columns]
    f["fundgap"] = frame[hv].isna().mean(axis=1) if hv else 1.0
    return f.fillna(0.0)


def fit(X: pd.DataFrame, y: pd.Series, l2: float = 1.0, iters: int = 50) -> Dict:
    """Newton/IRLS logistic regression with L2 on the coefficients (not the intercept)."""
    ok = y.notna()
    Xm = np.column_stack([np.ones(ok.sum()), X.loc[ok, FEATURES].to_numpy(float)])
    yv = y[ok].to_numpy(float)
    b = np.zeros(Xm.shape[1])
    R = np.eye(Xm.shape[1]) * l2
    R[0, 0] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Xm @ b))
        W = p * (1 - p)
        H = (Xm * W[:, None]).T @ Xm + R
        g = Xm.T @ (yv - p) - R @ b
        step = np.linalg.solve(H, g)
        b += step
        if np.abs(step).max() < 1e-7:
            break
    return {"intercept": float(b[0]), "coef": {k: float(v) for k, v in zip(FEATURES, b[1:])}, "n": int(ok.sum())}


def predict(frame: pd.DataFrame, model: Optional[Dict] = None) -> pd.Series:
    m = model if model and model.get("coef") else DEFAULT_MODEL
    X = features(frame)
    z = m.get("intercept", 0.0) + sum(float(m["coef"].get(k, 0.0)) * X[k] for k in FEATURES)
    return (1.0 / (1.0 + np.exp(-z))).clip(0.01, 0.99)


def relative_label(df: pd.DataFrame, ret_col: str = "fwd_ret") -> pd.Series:
    """1 if the stock beat the median stock of its own cross-section (date)."""
    med = df.groupby("tarih")[ret_col].transform("median")
    return (df[ret_col] > med).astype(float).where(df[ret_col].notna())


def reliability(p: pd.Series, y: pd.Series) -> List[Dict]:
    d = pd.DataFrame({"p": p, "y": y}).dropna()
    if d.empty:
        return []
    bins = [0, 0.45, 0.5, 0.55, 0.6, 0.65, 1.0]
    d["b"] = pd.cut(d["p"], bins)
    out = []
    for b, g in d.groupby("b", observed=True):
        out.append({"lo": float(b.left), "hi": float(b.right), "predicted": round(float(g["p"].mean()), 3),
                    "realised": round(float(g["y"].mean()), 3), "n": int(len(g))})
    return out


def brier_skill(p: pd.Series, y: pd.Series) -> Optional[float]:
    d = pd.DataFrame({"p": p, "y": y}).dropna()
    if len(d) < 50:
        return None
    return round(float(1 - ((d.p - d.y) ** 2).mean() / ((0.5 - d.y) ** 2).mean()) * 100, 2)


def walk_forward(rows: pd.DataFrame, first_year: int, last_year: int, purge_months: int = 13) -> Dict:
    """rows: OOS cross-sections with tarih, composite_pct, z_*, max_1m, fwd_ret.
    Fit on data whose labels were known before each test year, predict that year.
    Returns the final model (fitted on everything known) + OOS reliability."""
    rows = rows.copy()
    rows["y"] = relative_label(rows)
    X = pd.concat([features(g) for _, g in rows.groupby("tarih")]).reindex(rows.index)
    preds = []
    for y in range(first_year, last_year + 1):
        t0 = pd.Timestamp(f"{y}-01-01")
        tr = rows["tarih"] + pd.DateOffset(months=purge_months) <= t0
        te = (rows["tarih"] >= t0) & (rows["tarih"] < pd.Timestamp(f"{y + 1}-01-01"))
        if tr.sum() < 500 or te.sum() == 0:
            continue
        m = fit(X[tr], rows.loc[tr, "y"])
        p = 1.0 / (1.0 + np.exp(-(m["intercept"] + sum(m["coef"][k] * X.loc[te, k] for k in FEATURES))))
        preds.append(pd.Series(p, index=rows.index[te]))
    known = rows["tarih"] + pd.DateOffset(months=purge_months) <= rows["tarih"].max() + pd.DateOffset(days=1)
    final = fit(X[known], rows.loc[known, "y"]) if known.sum() >= 500 else dict(DEFAULT_MODEL)
    final["source"] = "walk_forward_backtest"
    if preds:
        P = pd.concat(preds)
        yv = rows.loc[P.index, "y"]
        final["oos_reliability"] = reliability(P, yv)
        final["oos_brier_skill_pct"] = brier_skill(P, yv)
        top = rows.loc[P.index].assign(p=P).sort_values("p", ascending=False).groupby("tarih").head(10)
        final["oos_top10_predicted"] = round(float(top["p"].mean()), 3)
        final["oos_top10_realised"] = round(float(top["y"].mean()), 3)
    return final


def grade(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "—"
    return "Yüksek" if p >= 0.62 else ("Orta" if p >= 0.56 else "Düşük")


LIVE_MIN_ROWS = 2000          # ~5 resolved monthly cross-sections before live data starts to count
LIVE_PRIOR_ROWS = 10000       # shrinkage: the backtest model counts like 10k observations


def live_update(prior: Optional[Dict], snaps: Optional[pd.DataFrame]) -> (Dict, Dict):
    """Self-improvement: once live 12-month outcomes exist, refit on them and blend with the
    backtest model (weight n_live / (n_live + LIVE_PRIOR_ROWS)). Returns (model_to_use, meta)."""
    base = prior if prior and prior.get("coef") else dict(DEFAULT_MODEL)
    meta = {"source": base.get("source", "prior"), "n_live": 0, "live_weight": 0.0}
    if snaps is None or snaps.empty or "fwd_ret" not in snaps or "composite_pct" not in snaps:
        return base, meta
    d = snaps.dropna(subset=["fwd_ret", "composite_pct"]).copy()
    if "confidence" in d:                       # only V3.8+ snapshots (same model definition)
        d = d[pd.to_numeric(d["confidence"], errors="coerce").notna()]
    meta["n_live"] = int(len(d))
    if len(d) < LIVE_MIN_ROWS:
        return base, meta
    y = relative_label(d)
    X = pd.concat([features(g) for _, g in d.groupby("tarih")]).reindex(d.index)
    live = fit(X, y)
    w = len(d) / (len(d) + LIVE_PRIOR_ROWS)
    model = {"intercept": (1 - w) * base.get("intercept", 0.0) + w * live["intercept"],
             "coef": {k: (1 - w) * float(base["coef"].get(k, 0.0)) + w * live["coef"][k] for k in FEATURES},
             "source": "backtest+live_blend"}
    for k in ("oos_reliability", "oos_brier_skill_pct", "oos_top10_predicted", "oos_top10_realised"):
        if k in base:
            model[k] = base[k]
    meta.update({"source": model["source"], "live_weight": round(w, 3), "live_coef": {k: round(v, 3) for k, v in live["coef"].items()}})
    return model, meta
