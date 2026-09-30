"""Step 2: clean and prepare the data."""
import numpy as np
import pandas as pd
from .config import *  # noqa: F401,F403

def preprocess(df: pd.DataFrame, verbose=True) -> pd.DataFrame:
    n0 = len(df)
    df = df.copy()
    df = df.drop_duplicates().drop_duplicates(subset="transaction_id")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["hour_of_day"] = df.timestamp.dt.hour
    df["day_of_week"] = df.timestamp.dt.day_name()
    df["is_weekend"] = (df.timestamp.dt.dayofweek >= 5).astype(int)
    df["is_night"] = df.hour_of_day.between(0, 5).astype(int)
    missing = df[["amount", "location", "device_type", "merchant_category"]].isna().sum().to_dict()
    df = df.dropna(subset=["amount"])
    df = df[df.amount > 0]
    for col in ["location", "device_type"]:
        mode = df.groupby("sender_id")[col].agg(lambda s: s.mode().iat[0] if s.notna().any() else np.nan)
        df[col] = df[col].fillna(df.sender_id.map(mode)).fillna(df[col].mode().iat[0])
    df["merchant_category"] = df["merchant_category"].fillna("Unknown")
    df["amount_log"] = np.log1p(df.amount)
    if verbose:
        print(f"[preprocess] rows in: {n0:,}  rows out: {len(df):,}  duplicates removed: {n0 - len(df):,}")
        print(f"[preprocess] missing values filled: {missing}")
    return df.reset_index(drop=True)

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, cats, prefix in [("transaction_type", TXN_TYPES, "type"),
                              ("upi_channel", CHANNELS, "chan"),
                              ("merchant_category", MERCHANT_CATEGORIES, "cat")]:
        cat = pd.Categorical(out[col], categories=cats)
        out = pd.concat([out, pd.get_dummies(cat, prefix=prefix, dtype=int).set_index(out.index)], axis=1)
    return out

if __name__ == "__main__":
    raw = pd.read_csv(RAW_PATH)
    clean = encode_categoricals(preprocess(raw))
    clean.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved -> {PROCESSED_PATH}  ({clean.shape[1]} columns)")
