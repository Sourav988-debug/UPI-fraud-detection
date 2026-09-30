import numpy as np, pandas as pd
from .config import DATA_DIR, RAW_PATH, SEED

def generate(n=3000, seed=SEED):
    rng=np.random.default_rng(seed)
    users=[f"U{i:04d}" for i in range(150)]
    cities=["Mumbai","Delhi","Bengaluru","Hyderabad","Chennai","Kolkata","Pune"]
    devices=["Android","iOS","Web"]
    channels=["GPay","PhonePe","Paytm","BHIM"]
    types=["P2P","P2M","Bill Payment","Collect Request"]
    cats=["Groceries","Food & Dining","Transport","Utilities","Shopping","Entertainment"]
    ts=pd.date_range("2026-01-01",periods=n,freq="15min")
    sender=rng.choice(users,n); avg=rng.lognormal(5.2,0.7,n)
    amount=np.maximum(10,avg)
    fraud=rng.random(n)<.05
    amount[fraud]*=rng.uniform(4,12,fraud.sum())
    receiver=np.array([f"M{rng.integers(1,400):04d}" for _ in range(n)])
    df=pd.DataFrame({"transaction_id":[f"TXN{i+1:07d}" for i in range(n)],"timestamp":ts,
      "sender_id":sender,"receiver_id":receiver,"amount":amount,
      "device_type":rng.choice(devices,n,p=[.6,.3,.1]),"location":rng.choice(cities,n),
      "merchant_category":rng.choice(cats,n),"transaction_type":rng.choice(types,n),
      "upi_channel":rng.choice(channels,n),"is_anomaly":fraud.astype(int),
      "anomaly_type":np.where(fraud,"amount_spike","normal")})
    return df

if __name__=="__main__":
    DATA_DIR.mkdir(exist_ok=True)
    generate().to_csv(RAW_PATH,index=False)
