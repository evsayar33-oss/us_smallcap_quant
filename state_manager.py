import json
import os
import pandas as pd
from datetime import datetime

STATE_FILE = "us_ai_state.json"
LIFECYCLE_LOG_FILE = "signals_lifecycle.csv"

def load_ai_state():
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ State okuma hatası: {e}")
        return {}

def save_ai_state(state):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"❌ State kayıt hatası: {e}")
        return False

def load_lifecycle_signals():
    if not os.path.exists(LIFECYCLE_LOG_FILE):
        cols = [
            "tarih", "ticker", "entry_price", "stop_price", "target_cup", "target_bagger",
            "quant_score", "regime", "score_base", "score_quality", "score_flow", "score_ignition",
            "ret_30d", "ret_90d", "ret_180d", "max_drawdown", "peak_gain", "outcome"
        ]
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(LIFECYCLE_LOG_FILE)
        df["tarih"] = pd.to_datetime(df["tarih"])
        return df
    except Exception as e:
        print(f"⚠️ Sinyal defteri okuma hatası: {e}")
        return pd.DataFrame()
