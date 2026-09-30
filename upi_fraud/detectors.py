"""Steps 3-6: IQR, Isolation Forest, time-series detection and combined fraud logic."""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .config import *  # noqa: F401,F403

class IQRDetector:
    COLS = {"amount_log": (IQR_K, 0.0), "amount_dev_from_user_avg": (IQR_DEV_K, 0.0),
            "txn_count_last_hour": (IQR_K, 1.0)}
    def fit(self, df):
        self.fences = {}
        for c, (k, min_iqr) in self.COLS.items():
            q1, q3 = df[c].quantile([.25, .75])
            self.fences[c] = float(q3 + k * max(q3 - q1, min_iqr))
        return self
    def flags(self, df):
        out = pd.DataFrame({f"iqr_{c}": (df[c] > f).astype(int) for c, f in self.fences.items()}, index=df.index)
        out["iqr_flag"] = out.max(axis=1)
        return out

class IsolationForestDetector:
    def __init__(self, contamination=IF_CONTAMINATION, n_estimators=300, seed=SEED):
        self.features = IF_FEATURES
        self.model = make_pipeline(StandardScaler(), IsolationForest(
            n_estimators=n_estimators, contamination=contamination,
            random_state=seed, n_jobs=-1))
    def fit(self, df):
        self.model.fit(df[self.features])
        return self
    def score(self, df):
        return -self.model.decision_function(df[self.features])
    def flags(self, df):
        return pd.DataFrame({"if_score": self.score(df),
                             "if_flag": (self.model.predict(df[self.features]) == -1).astype(int)}, index=df.index)

class TimeSeriesDetector:
    def __init__(self, z=TS_Z_THRESHOLD):
        self.z = z
    def flags(self, df):
        amt = (df.ts_amount_z > self.z) & (df.amount_ratio >= 2.5)
        freq = (df.ts_freq_z > self.z) & (df.txn_count_last_hour >= 3)
        spend = (df.ts_spend_z > self.z) & (df.spend_last_24h_ratio >= 5)
        out = pd.DataFrame({"ts_amount_spike": amt.astype(int), "ts_freq_spike": freq.astype(int),
                            "ts_spend_spike": spend.astype(int)}, index=df.index)
        out["ts_flag"] = out.max(axis=1)
        return out

def combine(iqr_flag, if_flag, ts_flag):
    n = iqr_flag + if_flag + ts_flag
    level = np.select([n >= 2, n == 1], ["HIGH", "MEDIUM"], default="LOW")
    return pd.DataFrame({"n_methods_flagged": n, "fraud_score": np.round(n / 3, 3), "risk_level": level})

class FraudDetector:
    def fit(self, feats):
        self.iqr = IQRDetector().fit(feats)
        self.iforest = IsolationForestDetector().fit(feats)
        self.ts = TimeSeriesDetector()
        return self
    def predict(self, feats):
        a, b, c = self.iqr.flags(feats), self.iforest.flags(feats), self.ts.flags(feats)
        return pd.concat([a, b, c, combine(a.iqr_flag, b.if_flag, c.ts_flag)], axis=1)
