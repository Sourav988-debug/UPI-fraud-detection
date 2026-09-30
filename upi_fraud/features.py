"""Feature engineering for batch training and live scoring."""
import numpy as np
import pandas as pd

DEFAULT_GAP_MIN = 1440.0

def build_profiles(df):
    g = df.groupby("sender_id")
    prof = pd.DataFrame({
        "avg_amount": g.amount.mean(),
        "std_amount": g.amount.std().fillna(0),
        "home_location": g.location.agg(lambda s: s.mode().iat[0]),
        "usual_device": g.device_type.agg(lambda s: s.mode().iat[0]),
    })
    return {
        "users": prof.to_dict("index"),
        "default": {
            "avg_amount": float(df.amount.median()),
            "std_amount": float(df.amount.std()),
            "home_location": None,
            "usual_device": None,
        },
    }

def _user_features(u, p):
    u = u.sort_values("timestamp", kind="stable").copy()
    s = u.set_index("timestamp")
    avg = max(float(p["avg_amount"]), 1.0)
    std = float(p["std_amount"])
    u["user_avg_amount"] = avg
    u["amount_dev_from_user_avg"] = (u.amount - avg) / max(std, 0.5 * avg)
    u["amount_ratio"] = u.amount / avg
    u["txn_count_last_hour"] = s["amount"].rolling("1h").count().to_numpy()
    u["spend_last_24h"] = s["amount"].rolling("24h").sum().to_numpy()
    u["spend_last_24h_ratio"] = u.spend_last_24h / avg
    gap = u.timestamp.diff().dt.total_seconds().div(60).fillna(DEFAULT_GAP_MIN)
    u["time_since_last_txn_min"] = gap
    u["time_since_last_txn_log"] = np.log1p(gap)
    times = u.timestamp.to_numpy()
    rec = u.receiver_id.to_numpy()
    window = np.timedelta64(24, "h")
    u["recipient_repetition_count"] = [
        int(((rec[:i + 1] == rec[i]) & (times[:i + 1] > times[i] - window)).sum())
        for i in range(len(u))
    ]
    u["device_change"] = (
        (u.device_type != p["usual_device"]).astype(int) if p["usual_device"] else 0
    )
    u["location_change"] = (
        (u.location != p["home_location"]).astype(int) if p["home_location"] else 0
    )

    def roll_z(series, window, floor):
        prev = series.shift(1).rolling(window, min_periods=5)
        return ((series - prev.mean()) / np.maximum(prev.std(), floor)).fillna(0.0)

    u["ts_amount_z"] = roll_z(u.amount.reset_index(drop=True), 10, 0.25 * avg).to_numpy()
    u["ts_freq_z"] = roll_z(u.txn_count_last_hour.reset_index(drop=True), 20, 0.5).to_numpy()
    u["ts_spend_z"] = roll_z(u.spend_last_24h.reset_index(drop=True), 20, 0.5 * avg).to_numpy()
    u["amount_log"] = np.log1p(u.amount)
    u["is_night"] = u.timestamp.dt.hour.between(0, 5).astype(int)
    return u

def build_features(df, profiles):
    parts = [
        _user_features(g, profiles["users"].get(sid, profiles["default"]))
        for sid, g in df.groupby("sender_id", sort=False)
    ]
    return pd.concat(parts).sort_values("timestamp", kind="stable").reset_index(drop=True)

def hourly_series(df):
    h = df.set_index("timestamp").resample("1h").agg(
        txn_count=("amount", "size"), total_amount=("amount", "sum")
    )
    for c in ["txn_count", "total_amount"]:
        r = h[c].rolling(24 * 7, min_periods=24)
        h[f"{c}_roll_mean"], h[f"{c}_roll_std"] = r.mean(), r.std()
        h[f"{c}_z"] = (h[c] - h[f"{c}_roll_mean"]) / h[f"{c}_roll_std"]
    h["daily_amount"] = h.total_amount.rolling(24).sum()
    return h
