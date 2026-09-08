import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime
from scipy.stats import spearmanr
from state_manager import load_ai_state, save_ai_state, load_lifecycle_signals, LIFECYCLE_LOG_FILE

def fetch_current_us_snapshot():
    """TradingView üzerinden ABD piyasa fiyatlarını çeker."""
    url = "https://scanner.tradingview.com/america/scan"
    payload = {
        "filter": [
            {"left": "type", "operation": "equal", "right": "stock"},
            {"left": "exchange", "operation": "in_range", "right": ["AMEX", "NASDAQ", "NYSE"]}
        ],
        "columns": ["name", "close", "high", "low"],
        "sort": {"sortBy": "Value.Traded", "sortOrder": "desc"},
        "range": [0, 800]
    }
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    try:
        res = requests.post(url, json=payload, headers=headers, timeout=15)
        if res.status_code == 200:
            data = res.json().get("data", [])
            price_map = {}
            for item in data:
                d = item["d"]
                ticker = d[0]
                close_p = float(d[1]) if d[1] is not None else 0.0
                high_p = float(d[2]) if d[2] is not None else close_p
                low_p = float(d[3]) if d[3] is not None else close_p
                price_map[ticker] = {"close": close_p, "high": high_p, "low": low_p}
            return price_map
    except Exception as e:
        print(f"⚠️ US Denetçi piyasa verisi çekemedi: {e}")
    return {}

def update_signal_lifecycle(df_signals, market_prices, state):
    if df_signals.empty or not market_prices:
        return df_signals, []

    today = pd.Timestamp.now().normalize()
    exit_alerts = []

    for idx, row in df_signals.iterrows():
        ticker = row["ticker"]
        if ticker not in market_prices:
            continue

        entry_p = float(row["entry_price"])
        target_p = float(row.get("target_cup", entry_p * 2.0))
        initial_stop = float(row.get("stop_price", entry_p * 0.88))

        if entry_p <= 0:
            continue

        curr_p = market_prices[ticker]["close"]
        curr_low = market_prices[ticker]["low"]
        curr_high = market_prices[ticker]["high"]

        gain_from_entry = ((curr_p - entry_p) / entry_p) * 100.0
        low_from_entry = ((curr_low - entry_p) / entry_p) * 100.0
        high_from_entry = ((curr_high - entry_p) / entry_p) * 100.0

        prev_drawdown = float(row.get("max_drawdown", 0.0)) if pd.notna(row.get("max_drawdown")) else 0.0
        df_signals.at[idx, "max_drawdown"] = round(min(prev_drawdown, low_from_entry), 2)

        prev_peak = float(row.get("peak_gain", 0.0)) if pd.notna(row.get("peak_gain")) else 0.0
        peak_gain = max(prev_peak, high_from_entry)
        df_signals.at[idx, "peak_gain"] = round(peak_gain, 2)

        sig_date = pd.to_datetime(row["tarih"])
        days_passed = (today - sig_date).days

        if days_passed >= 30 and pd.isna(row.get("ret_30d")):
            df_signals.at[idx, "ret_30d"] = round(gain_from_entry, 2)
        if days_passed >= 90 and pd.isna(row.get("ret_90d")):
            df_signals.at[idx, "ret_90d"] = round(gain_from_entry, 2)
        if days_passed >= 180 and pd.isna(row.get("ret_180d")):
            df_signals.at[idx, "ret_180d"] = round(gain_from_entry, 2)

        # Hedefe İlerleme Oranına Göre Dinamik İzleyen Stop
        total_target_distance = target_p - entry_p
        trailing_stop = initial_stop

        if total_target_distance > 0:
            peak_price = entry_p * (1.0 + peak_gain / 100.0)
            target_progress = (peak_price - entry_p) / total_target_distance

            if target_progress >= 0.25:
                trailing_stop = max(trailing_stop, round(entry_p * 1.02, 2))
            if target_progress >= 0.50:
                trailing_stop = max(trailing_stop, round(entry_p + (total_target_distance * 0.30), 2))
            if target_progress >= 0.80:
                trailing_stop = max(trailing_stop, round(entry_p + (total_target_distance * 0.65), 2))

        df_signals.at[idx, "stop_price"] = trailing_stop

        curr_status = row.get("outcome", "INCUBATING")
        if curr_status in ["INCUBATING", "PENDING"]:
            if curr_low <= initial_stop and peak_gain < 15.0:
                df_signals.at[idx, "outcome"] = "FAIL_BASE_BREAKDOWN"
                exit_alerts.append({
                    "ticker": ticker,
                    "type": "STOP_LOSS",
                    "msg": f"Taban desteği kırıldı (${curr_p:.2f}). Zararı kesin."
                })
            elif curr_low <= trailing_stop and peak_gain >= 25.0:
                df_signals.at[idx, "outcome"] = "WIN_PROFIT_LOCKED"
                exit_alerts.append({
                    "ticker": ticker,
                    "type": "TAKE_PROFIT",
                    "msg": f"İzleyen stop tetiklendi (${curr_p:.2f}). %+ {gain_from_entry:.1f} Dolar kârını cebe koyun!"
                })
            elif curr_high >= target_p:
                df_signals.at[idx, "outcome"] = "WIN_CUP_BREAKOUT"
                exit_alerts.append({
                    "ticker": ticker,
                    "type": "TARGET_HIT",
                    "msg": f"1. Çanak hedefine (${target_p:.2f}) ulaşıldı! Ana kârı realize edin."
                })

    df_signals.to_csv(LIFECYCLE_LOG_FILE, index=False)
    return df_signals, exit_alerts

def run_feedback_loop_optimization(df_signals, state):
    mature = df_signals[df_signals["outcome"].isin(["WIN_MULTI_BAGGER", "WIN_CUP_BREAKOUT", "WIN_PROFIT_LOCKED", "FAIL_BASE_BREAKDOWN"])]
    min_samples = state.get("learning_params", {}).get("min_sample_size", 10)

    if len(mature) < min_samples:
        state["audit_summary"]["status"] = f"🦅 US KULUÇKA TAKİBİNDE ({len(mature)}/{min_samples} Sinyal)"
        state["audit_summary"]["total_signals_audited"] = len(mature)
        save_ai_state(state)
        return state

    wins = mature[mature["outcome"].str.startswith("WIN")]
    win_rate = round((len(wins) / len(mature)) * 100.0, 1)

    factors = {
        "macro_base": "score_base",
        "growth_quality": "score_quality",
        "volume_flow": "score_flow",
        "ignition": "score_ignition"
    }
    
    target_returns = mature["peak_gain"].values
    ic_scores = {}

    for f_key, col in factors.items():
        x = pd.to_numeric(mature[col], errors='coerce').fillna(50.0).values
        if np.std(x) > 0 and np.std(target_returns) > 0:
            corr, _ = spearmanr(x, target_returns)
            ic_scores[f_key] = max(corr if not np.isnan(corr) else 0.05, 0.05)
        else:
            ic_scores[f_key] = 0.05

    total_ic = sum(ic_scores.values())
    raw_weights = {k: ic_scores[k] / total_ic for k in ic_scores}

    lr = state["learning_params"].get("learning_rate", 0.05)
    current_weights = state["weights"]

    updated = {}
    for k in current_weights:
        new_w = (1.0 - lr) * current_weights[k] + lr * raw_weights[k]
        updated[k] = min(max(new_w, 0.10), 0.50)

    w_sum = sum(updated.values())
    final_weights = {k: round(v / w_sum, 3) for k, v in updated.items()}

    state["weights"] = final_weights
    state["audit_summary"]["total_signals_audited"] = len(mature)
    state["audit_summary"]["win_rate_6m"] = win_rate
    state["audit_summary"]["last_audit_date"] = datetime.now().strftime("%Y-%m-%d")
    state["audit_summary"]["status"] = f"🧠 US AI KALİBRE EDİLDİ (Win: %{win_rate})"

    save_ai_state(state)
    return state

def audit_and_calibrate():
    state = load_ai_state()
    df_signals = load_lifecycle_signals()
    if df_signals.empty:
        return state, []
    market_prices = fetch_current_us_snapshot()
    if not market_prices:
        return state, []
    df_updated, exit_alerts = update_signal_lifecycle(df_signals, market_prices, state)
    state = run_feedback_loop_optimization(df_updated, state)
    return state, exit_alerts

if __name__ == "__main__":
    audit_and_calibrate()
