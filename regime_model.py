"""Forward-looking market regime model: sticky Gaussian HMM (pure numpy).

Observed daily features (free data via yfinance):
  1. XU100 log return
  2. log 10-day realised volatility (annualised)
  3. 5-day log change of USDTRY (currency stress channel, dominant for BIST)

Why an HMM instead of V1's one-day breadth rules:
* Regimes are persistent latent states (sticky transition matrix) -> no daily
  label flipping.
* Only the FORWARD (filtering) pass is used for live/backtest decisions, so no
  look-ahead.
* The transition matrix gives a genuine forecast: P(state in k days) and the
  expected market return over the trade horizon, which feeds the absolute
  entry gate (expected net trade return must be positive).
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

import config as C

LABELS_3 = ["RISK_OFF", "NEUTRAL", "RISK_ON"]
STICKY_PSEUDO = 20.0


# ------------------------------------------------------------------ data
def download_regime_series(start: str = C.REGIME_HISTORY_START, end: Optional[str] = None) -> pd.DataFrame:
    import yfinance as yf  # lazy: keeps the rest of the engine importable without it
    raw = yf.download([C.REGIME_TICKER_INDEX, C.REGIME_TICKER_FX], start=start, end=end,
                      interval="1d", auto_adjust=True, progress=False, threads=True, group_by="ticker")
    if raw is None or raw.empty:
        raise RuntimeError("regime series empty")
    idx = raw[C.REGIME_TICKER_INDEX]["Close"].dropna()
    fx = raw[C.REGIME_TICKER_FX]["Close"].dropna()
    idx.index = pd.to_datetime(idx.index).tz_localize(None).normalize()
    fx.index = pd.to_datetime(fx.index).tz_localize(None).normalize()
    df = pd.DataFrame({"idx": idx})
    df["fx"] = fx.reindex(df.index).ffill()
    df = df.dropna()
    if len(df) < 300:
        raise RuntimeError(f"regime series too short ({len(df)})")
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    r = np.log(df["idx"]).diff()
    rv = np.log(r.rolling(10).std() * np.sqrt(252) + 1e-6)
    fx5 = np.log(df["fx"]).diff(5)
    X = pd.DataFrame({"r": r, "rv": rv, "fx5": fx5}).dropna()
    return X


# ------------------------------------------------------------------ HMM core
def _log_emission(Xs: np.ndarray, means: np.ndarray, var: np.ndarray) -> np.ndarray:
    # Xs: T x D, means/var: K x D -> T x K
    diff = Xs[:, None, :] - means[None, :, :]
    return -0.5 * (np.log(2 * np.pi * var)[None, :, :] + diff ** 2 / var[None, :, :]).sum(axis=2)


def _forward(logB: np.ndarray, A: np.ndarray, pi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    T, K = logB.shape
    alpha = np.zeros((T, K))
    c = np.zeros(T)
    m = logB.max(axis=1, keepdims=True)
    B = np.exp(logB - m)
    a = pi * B[0]
    c[0] = a.sum() + 1e-300
    alpha[0] = a / c[0]
    for t in range(1, T):
        a = (alpha[t - 1] @ A) * B[t]
        c[t] = a.sum() + 1e-300
        alpha[t] = a / c[t]
    loglik = float(np.sum(np.log(c)) + m.sum())
    return alpha, c, loglik


def _backward(logB: np.ndarray, A: np.ndarray, c: np.ndarray) -> np.ndarray:
    T, K = logB.shape
    m = logB.max(axis=1, keepdims=True)
    B = np.exp(logB - m)
    beta = np.zeros((T, K))
    beta[-1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
    return beta


def fit_hmm(X: pd.DataFrame, K: int = C.REGIME_STATES, n_iter: int = 150, tol: float = 1e-5) -> Dict:
    Xv = X.to_numpy(dtype=float)
    mu_s, sd_s = Xv.mean(axis=0), Xv.std(axis=0) + 1e-9
    Xs = (Xv - mu_s) / sd_s
    T, D = Xs.shape
    # init: terciles of realised vol
    q = np.quantile(Xs[:, 1], np.linspace(0, 1, K + 1))
    assign = np.clip(np.searchsorted(q[1:-1], Xs[:, 1]), 0, K - 1)
    means = np.array([Xs[assign == k].mean(axis=0) if np.any(assign == k) else Xs.mean(axis=0) for k in range(K)])
    var = np.array([Xs[assign == k].var(axis=0) + 0.05 if np.sum(assign == k) > 2 else np.ones(D) for k in range(K)])
    A = np.full((K, K), 0.05 / (K - 1))
    np.fill_diagonal(A, 0.95)
    pi = np.full(K, 1.0 / K)
    prev = -np.inf
    for _ in range(n_iter):
        logB = _log_emission(Xs, means, var)
        alpha, c, ll = _forward(logB, A, pi)
        beta = _backward(logB, A, c)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300
        m = logB.max(axis=1, keepdims=True)
        B = np.exp(logB - m)
        xi_sum = A * (alpha[:-1].T @ (B[1:] * beta[1:] / c[1:, None]))
        A = xi_sum + 1.0 + np.eye(K) * STICKY_PSEUDO
        A /= A.sum(axis=1, keepdims=True)
        pi = gamma[0] + 1e-3
        pi /= pi.sum()
        w = gamma.sum(axis=0) + 1e-9
        means = (gamma.T @ Xs) / w[:, None]
        var = np.array([(gamma[:, k][:, None] * (Xs - means[k]) ** 2).sum(axis=0) / w[k] for k in range(K)])
        var = np.maximum(var, 0.02)
        if abs(ll - prev) < tol * max(1.0, abs(ll)):
            break
        prev = ll
    # order states by mean daily return of the index (unstandardised)
    r_mean = means[:, 0] * sd_s[0] + mu_s[0]
    order = np.argsort(r_mean)
    means, var, r_mean = means[order], var[order], r_mean[order]
    A = A[np.ix_(order, order)]
    pi = pi[order]
    r_vol = np.sqrt(var[:, 0]) * sd_s[0] * np.sqrt(252)
    labels = LABELS_3 if K == 3 else [f"S{k}" for k in range(K)]
    return {
        "K": K, "means": means.tolist(), "var": var.tolist(), "A": A.tolist(), "pi": pi.tolist(),
        "scaler_mean": mu_s.tolist(), "scaler_std": sd_s.tolist(),
        "state_mean_daily_ret": r_mean.tolist(), "state_ann_vol": r_vol.tolist(),
        "labels": labels, "loglik": float(prev if np.isfinite(prev) else 0.0),
        "n_obs": int(T), "fit_date": datetime.utcnow().strftime("%Y-%m-%d"),
    }


def filtered_probs(params: Dict, X: pd.DataFrame) -> np.ndarray:
    Xs = (X.to_numpy(float) - np.array(params["scaler_mean"])) / np.array(params["scaler_std"])
    logB = _log_emission(Xs, np.array(params["means"]), np.array(params["var"]))
    alpha, _, _ = _forward(logB, np.array(params["A"]), np.array(params["pi"]))
    return alpha


def expected_market_return_pct(params: Dict, p_now: np.ndarray, horizon: int = 5) -> float:
    A = np.array(params["A"])
    mu = np.array(params["state_mean_daily_ret"])
    p = np.asarray(p_now, float)
    total = 0.0
    for _ in range(horizon):
        p = p @ A
        total += float(p @ mu)
    return float((np.exp(total) - 1.0) * 100.0)


def summarize(params: Dict, alpha: np.ndarray, dates) -> Dict:
    p = alpha[-1]
    labels = params["labels"]
    hist = [labels[int(i)] for i in alpha[-10:].argmax(axis=1)]
    return {
        "label": labels[int(np.argmax(p))],
        "probs": {labels[k]: round(float(p[k]), 4) for k in range(len(labels))},
        "p_risk_off": round(float(p[0]), 4),
        "exp_mkt_5d_pct": round(expected_market_return_pct(params, p, 5), 3),
        "exp_mkt_12m_pct": round(expected_market_return_pct(params, p, 252), 2),
        "argmax_history": hist,
        "as_of": str(pd.Timestamp(dates[-1]).date()),
        "persistence": round(float(np.array(params["A"])[int(np.argmax(p)), int(np.argmax(p))]), 4),
    }


# ------------------------------------------------------------------ live update
def breadth_fallback(snapshot: pd.DataFrame) -> Dict:
    if snapshot is None or "change_pct" not in snapshot.columns:
        return {"label": "UNKNOWN", "probs": {}, "p_risk_off": 0.5, "exp_mkt_5d_pct": 0.0, "exp_mkt_12m_pct": None}
    ch = pd.to_numeric(snapshot["change_pct"], errors="coerce").dropna()
    if ch.empty:
        return {"label": "UNKNOWN", "probs": {}, "p_risk_off": 0.5, "exp_mkt_5d_pct": 0.0, "exp_mkt_12m_pct": None}
    down, med = float((ch < 0).mean()), float(ch.median())
    z = 4.0 * (down - 0.5) - 0.6 * med
    p_off = float(1.0 / (1.0 + np.exp(-z)))
    label = "RISK_OFF" if p_off > 0.66 else ("RISK_ON" if p_off < 0.33 else "NEUTRAL")
    return {"label": label, "probs": {"RISK_OFF": round(p_off, 4)}, "p_risk_off": round(p_off, 4),
            "exp_mkt_5d_pct": 0.0, "exp_mkt_12m_pct": None}


def update_regime(state: Dict, snapshot: Optional[pd.DataFrame] = None, today=None,
                  series: Optional[pd.DataFrame] = None, allow_download: bool = True) -> Dict:
    reg = state.setdefault("regime", {})
    today = pd.Timestamp(today or pd.Timestamp.now()).normalize()
    try:
        if series is not None and len(series) >= 300:
            df = series
        elif allow_download:
            df = download_regime_series()
        else:
            raise RuntimeError("regime series unavailable")
        X = make_features(df)
        params = reg.get("params")
        last_fit = pd.Timestamp(reg["last_fit"]) if reg.get("last_fit") else None
        if params is None or last_fit is None or (today - last_fit).days >= C.REGIME_REFIT_DAYS:
            params = fit_hmm(X)
            reg["params"] = params
            reg["last_fit"] = str(today.date())
        alpha = filtered_probs(params, X)
        reg.update(summarize(params, alpha, X.index))
        reg["degraded"] = False
        reg["source"] = "hmm_yfinance"
        reg["state_stats"] = {
            params["labels"][k]: {
                "mean_daily_ret_pct": round(params["state_mean_daily_ret"][k] * 100, 4),
                "ann_vol_pct": round(params["state_ann_vol"][k] * 100, 2),
            } for k in range(params["K"])
        }
    except Exception as exc:
        params = reg.get("params")
        if params and reg.get("probs"):
            # propagate the last filtered distribution with the transition matrix
            labels = params["labels"]
            p = np.array([reg["probs"].get(l, 0.0) for l in labels], float)
            p = p / p.sum() if p.sum() > 0 else np.full(len(labels), 1.0 / len(labels))
            p = p @ np.array(params["A"])
            reg.update({
                "label": labels[int(np.argmax(p))],
                "probs": {labels[k]: round(float(p[k]), 4) for k in range(len(labels))},
                "p_risk_off": round(float(p[0]), 4),
                "exp_mkt_5d_pct": round(expected_market_return_pct(params, p, 5), 3),
                "exp_mkt_12m_pct": round(expected_market_return_pct(params, p, 252), 2),
            })
            reg["source"] = "hmm_propagated"
        else:
            reg.update(breadth_fallback(snapshot if snapshot is not None else pd.DataFrame({"change_pct": []})))
            reg["source"] = "breadth_fallback"
        reg["degraded"] = True
        reg["error"] = str(exc)[:200]
    return reg
