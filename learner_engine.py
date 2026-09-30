"""Controlled self-learning: IC estimation, Grinold-Kahn weighting, champion/challenger.

Fixes of V1:
* Learns ONLY from rows that carry factor values + a resolved label
  (V1 learned from legacy rows with all factors missing).
* Daily cross-sectional rank IC (not pooled): the market's daily move cannot
  masquerade as stock-selection skill.
* Overlapping labels: Newey-West standard errors (lag = horizon-1) and
  effective sample size n_dates / horizon.
* Purged split between challenger-train and test (horizon+1 sessions).
* Collinearity: weights = Omega^-1 * IC with a shrunk factor-correlation matrix.
* Negative IC is kept (sign can flip) instead of being floored away.
* Bayesian shrinkage toward a research prior (walk-forward backtest) or a
  hand prior; live evidence gradually takes over.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config as C

ZCOLS = [f"z_{k}" for k in C.FACTORS]


# ------------------------------------------------------------------ statistics
def newey_west(x, lags: int) -> Tuple[float, float, float, int]:
    s = pd.Series(x, dtype=float).dropna().to_numpy()
    n = s.size
    if n < 3:
        return (float(s.mean()) if n else 0.0), np.inf, 0.0, n
    mu = s.mean()
    e = s - mu
    gamma0 = float(e @ e) / n
    var = gamma0
    for L in range(1, min(lags, n - 1) + 1):
        w = 1.0 - L / (lags + 1.0)
        var += 2.0 * w * float(e[L:] @ e[:-L]) / n
    var = max(var, 1e-12)
    se = float(np.sqrt(var / n))
    return float(mu), se, float(mu / se), n


def daily_rank_ic(df: pd.DataFrame, score_cols: List[str], target: str) -> pd.DataFrame:
    """Dates x score_cols Spearman IC, computed per date on names with a label."""
    d = df.dropna(subset=[target])
    rows = {}
    for dt, g in d.groupby("tarih"):
        if len(g) < C.MIN_CROSS_SECTION:
            continue
        y = g[target].rank()
        rec = {}
        for c in score_cols:
            x = g[c]
            if x.nunique() < 3:
                rec[c] = np.nan
                continue
            rec[c] = float(np.corrcoef(x.rank(), y)[0, 1])
        rows[dt] = rec
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def factor_corr(df: pd.DataFrame) -> Optional[np.ndarray]:
    mats = []
    for _, g in df.groupby("tarih"):
        if len(g) < C.MIN_CROSS_SECTION:
            continue
        Z = g[ZCOLS].to_numpy(float)
        sd = Z.std(axis=0)
        Z = np.where(sd > 0, (Z - Z.mean(axis=0)) / np.where(sd > 0, sd, 1), 0.0)
        M = (Z.T @ Z) / len(Z)
        np.fill_diagonal(M, 1.0)
        mats.append(M)
    if not mats:
        return None
    return np.mean(mats, axis=0)


# ------------------------------------------------------------------ priors
def get_prior(research: Optional[Dict]) -> Dict:
    if research and isinstance(research.get("ic_mean"), dict):
        mu = np.array([float(research["ic_mean"].get(k, 0.0) or 0.0) for k in C.FACTORS])
        om = research.get("omega")
        omega = np.array(om, float) if om is not None and np.shape(om) == (len(C.FACTORS), len(C.FACTORS)) else np.eye(len(C.FACTORS))
        kappa = min(C.PRIOR_STRENGTH_RESEARCH, float(research.get("n_eff_dates", C.PRIOR_STRENGTH_RESEARCH)))
        return {"mu": mu, "omega": omega, "kappa": max(kappa, C.PRIOR_STRENGTH_HAND),
                "source": f"research:{research.get('generated_at', '?')}",
                "regime_ic": research.get("regime_ic", {})}
    mu = np.array([C.PRIOR_IC.get(k, 0.0) for k in C.FACTORS])
    return {"mu": mu, "omega": np.eye(len(C.FACTORS)), "kappa": C.PRIOR_STRENGTH_HAND,
            "source": "hand", "regime_ic": {}}


def weights_from_ic(mu: np.ndarray, omega: np.ndarray) -> Dict[str, float]:
    K = len(C.FACTORS)
    om = (1.0 - C.OMEGA_SHRINK) * np.asarray(omega, float) + C.OMEGA_SHRINK * np.eye(K)
    try:
        w = np.linalg.solve(om, np.asarray(mu, float))
    except np.linalg.LinAlgError:
        w = np.asarray(mu, float)
    if not np.all(np.isfinite(w)) or np.abs(w).sum() < 1e-12:
        w = np.array([C.PRIOR_IC.get(k, 0.0) for k in C.FACTORS])
    w = w / np.abs(w).sum()
    for _ in range(5):  # cap single-factor dominance, re-normalise
        w = np.clip(w, -C.MAX_ABS_WEIGHT, C.MAX_ABS_WEIGHT)
        w = w / max(np.abs(w).sum(), 1e-12)
    return {k: round(float(v), 5) for k, v in zip(C.FACTORS, w)}


def fit_weights(df: pd.DataFrame, prior: Dict, target: str = "fwd_ret") -> Tuple[Dict[str, float], Dict]:
    ic = daily_rank_ic(df, ZCOLS, target) if df is not None and not df.empty else pd.DataFrame()
    n_dates = int(len(ic))
    n_eff = n_dates / float(C.LABEL_HORIZON)
    stats = {}
    mu_live = np.zeros(len(C.FACTORS))
    for i, k in enumerate(C.FACTORS):
        col = f"z_{k}"
        if n_dates and col in ic:
            m, se, t, n = newey_west(ic[col], C.LABEL_HORIZON - 1)
            mu_live[i] = m if np.isfinite(m) else 0.0
            stats[k] = {"ic_mean": round(m, 5), "t_nw": round(t, 3), "n_dates": n}
        else:
            stats[k] = {"ic_mean": None, "t_nw": None, "n_dates": 0}
    kappa = prior["kappa"]
    mu_post = (n_eff * mu_live + kappa * prior["mu"]) / (n_eff + kappa)
    om_live = factor_corr(df) if n_dates >= 5 else None
    omega = prior["omega"] if om_live is None else (n_eff * om_live + kappa * prior["omega"]) / (n_eff + kappa)
    w = weights_from_ic(mu_post, omega)
    for i, k in enumerate(C.FACTORS):
        stats[k]["ic_post"] = round(float(mu_post[i]), 5)
    meta = {"n_dates": n_dates, "n_eff": round(n_eff, 2), "prior_kappa": kappa, "factor_stats": stats,
            "omega": np.round(omega, 4).tolist()}
    return w, meta


def composite_daily_ic(df: pd.DataFrame, weights: Dict[str, float], target: str = "fwd_ret") -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    tmp = df[["tarih", target]].copy()
    tmp["_score"] = sum(float(w) * df[f"z_{k}"].fillna(0.0) for k, w in weights.items() if f"z_{k}" in df)
    ic = daily_rank_ic(tmp, ["_score"], target)
    return ic["_score"] if "_score" in ic else pd.Series(dtype=float)


def _step_limit(old: Dict[str, float], new: Dict[str, float]) -> Dict[str, float]:
    if not old:
        return new
    o = np.array([old.get(k, 0.0) for k in C.FACTORS])
    n = np.array([new.get(k, 0.0) for k in C.FACTORS])
    l1 = np.abs(n - o).sum()
    lam = 1.0 if l1 <= C.MAX_WEIGHT_L1_STEP else C.MAX_WEIGHT_L1_STEP / l1
    w = o + lam * (n - o)
    w = w / max(np.abs(w).sum(), 1e-12)
    return {k: round(float(v), 5) for k, v in zip(C.FACTORS, w)}


# ------------------------------------------------------------------ orchestration
def run_learning(state: Dict, dataset: pd.DataFrame, research: Optional[Dict]) -> Dict:
    model = state.setdefault("model", {})
    prior = get_prior(research)
    now = datetime.utcnow().strftime("%Y-%m-%d")
    ds = dataset.dropna(subset=["fwd_ret"]) if dataset is not None and not dataset.empty else pd.DataFrame()
    dates = sorted(ds["tarih"].unique()) if not ds.empty else []

    # (re)seed the champion from the prior when there is no live evidence yet,
    # or when a new research prior arrives while live evidence is still thin.
    n_eff_live = len(dates) / float(C.LABEL_HORIZON)
    if not model.get("champion_weights") or (
            model.get("prior_source") != prior["source"] and n_eff_live < prior["kappa"] / 2):
        w0, meta0 = fit_weights(ds if len(dates) >= C.MIN_LEARN_DATES else pd.DataFrame(), prior)
        if model.get("champion_weights"):
            model["previous_weights"] = model["champion_weights"]
            model["previous_version"] = model.get("champion_version")
        model["champion_weights"] = w0
        model["champion_version"] = f"prior-{prior['source'].split(':')[0]}-{now}"
        model["prior_source"] = prior["source"]
        model["promotion_date"] = now
        model["status"] = "SEEDED_FROM_PRIOR"

    # regime-conditional weights (prior from research regime ICs, updated live)
    regime_w = {}
    labels = set(prior.get("regime_ic", {}).keys())
    if not ds.empty and "regime_label" in ds:
        labels |= set(ds["regime_label"].dropna().astype(str).unique())
    for lab in labels:
        if lab in ("UNKNOWN", "nan", "None"):
            continue
        rp = prior.get("regime_ic", {}).get(lab)
        sub_prior = dict(prior)
        if rp and isinstance(rp.get("ic_mean"), dict):
            sub_prior["mu"] = np.array([float(rp["ic_mean"].get(k, 0.0) or 0.0) for k in C.FACTORS])
            sub_prior["kappa"] = float(min(C.REGIME_MIN_DATES, max(5.0, rp.get("n_eff_dates", 10))))
        sub = ds[ds["regime_label"].astype(str) == lab] if (not ds.empty and "regime_label" in ds) else pd.DataFrame()
        if rp or (not sub.empty and sub["tarih"].nunique() >= C.REGIME_MIN_DATES):
            w_r, _ = fit_weights(sub if not sub.empty and sub["tarih"].nunique() >= C.MIN_LEARN_DATES else pd.DataFrame(), sub_prior)
            regime_w[lab] = w_r
    model["regime_weights"] = regime_w

    need = C.MIN_LEARN_DATES + C.CHALLENGER_MIN_TEST_DATES
    if len(dates) < need:
        model["status"] = f"LEARNING_WAIT ({len(dates)}/{need} resolved cross-sections)"
        model["last_fit"] = now
        return state

    # ---------- challenger: purged train/test split
    n_test = max(C.CHALLENGER_MIN_TEST_DATES, int(len(dates) * 0.25))
    test_dates = dates[-n_test:]
    train_dates = dates[: max(0, len(dates) - n_test - (C.LABEL_HORIZON + 1))]
    train = ds[ds["tarih"].isin(train_dates)]
    test = ds[ds["tarih"].isin(test_dates)]
    w_ch, meta_tr = fit_weights(train, prior)
    ic_ch = composite_daily_ic(test, w_ch)
    ic_cp = composite_daily_ic(test, model["champion_weights"])
    both = pd.concat([ic_ch.rename("ch"), ic_cp.rename("cp")], axis=1).dropna()
    diff_mu, _, diff_t, n_diff = newey_west(both["ch"] - both["cp"], C.LABEL_HORIZON - 1) if len(both) else (0.0, 0, 0.0, 0)
    model["challenger"] = {
        "weights": w_ch, "test_dates": int(n_diff),
        "ic_challenger": round(float(both["ch"].mean()), 5) if len(both) else None,
        "ic_champion": round(float(both["cp"].mean()), 5) if len(both) else None,
        "ic_gain": round(diff_mu, 5), "t_nw": round(diff_t, 3), "evaluated": now,
    }

    w_full, meta_full = fit_weights(ds, prior)
    model["ic_stats"] = meta_full["factor_stats"]
    model["omega"] = meta_full["omega"]
    model["n_resolved_dates"] = len(dates)

    promoted = False
    if n_diff >= C.CHALLENGER_MIN_TEST_DATES and diff_mu >= C.PROMOTION_MIN_IC_GAIN and diff_t >= C.PROMOTION_MIN_T:
        model["previous_weights"] = model["champion_weights"]
        model["previous_version"] = model.get("champion_version")
        model["champion_weights"] = _step_limit(model["champion_weights"], w_full)
        model["champion_version"] = f"champion-{now}"
        model["promotion_date"] = now
        model["promotions"] = int(model.get("promotions", 0)) + 1
        model["status"] = f"PROMOTED (IC +{diff_mu:.4f}, t={diff_t:.2f})"
        promoted = True

    # ---------- rollback: champion's out-of-sample IC after its promotion date
    if not promoted and model.get("previous_weights"):
        pdate = pd.Timestamp(model.get("promotion_date") or dates[0])
        post = ds[ds["tarih"] > pdate]
        ic_post = composite_daily_ic(post, model["champion_weights"]).tail(C.ROLLBACK_WINDOW)
        if len(ic_post) >= C.ROLLBACK_WINDOW // 2:
            m, _, t, _ = newey_west(ic_post, C.LABEL_HORIZON - 1)
            if m < C.ROLLBACK_IC and t < C.ROLLBACK_T:
                bad = model["champion_weights"]
                model["champion_weights"] = model["previous_weights"]
                model["champion_version"] = f"rollback-{now}"
                model["previous_weights"] = bad
                model["promotion_date"] = now
                model["rollbacks"] = int(model.get("rollbacks", 0)) + 1
                model["status"] = f"ROLLBACK (post-promotion IC {m:.4f}, t={t:.2f})"
                promoted = True
    if not promoted:
        model["status"] = f"CHAMPION_KEPT (gain {diff_mu:+.4f}, t={diff_t:.2f}, n={n_diff})"
    model["last_fit"] = now
    return state


def live_composite_ic(dataset: pd.DataFrame) -> Dict:
    """Out-of-sample quality of the composite exactly as it was recorded live."""
    if dataset is None or dataset.empty or "composite" not in dataset:
        return {"n_dates": 0}
    d = dataset.dropna(subset=["fwd_ret", "composite"])
    ic = daily_rank_ic(d, ["composite"], "fwd_ret")
    if ic.empty:
        return {"n_dates": 0}
    s = ic["composite"]
    m, se, t, n = newey_west(s, C.LABEL_HORIZON - 1)
    recent = s.tail(20)
    return {"n_dates": n, "ic_mean": round(m, 5), "t_nw": round(t, 3),
            "ic_recent20": round(float(recent.mean()), 5) if len(recent) else None,
            "series_tail": [round(float(x), 4) for x in s.tail(60).tolist()]}
