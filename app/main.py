import os, sqlite3, uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import joblib, numpy as np, pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from upi_fraud.config import ARTIFACT_PATH, DATA_DIR
from upi_fraud.features import _user_features

STATE={}
DB_PATH=os.getenv("FLAGGED_DB",str(DATA_DIR/"flagged_live.db"))

class Transaction(BaseModel):
    transaction_id:str|None=None
    timestamp:datetime|None=None
    sender_id:str
    receiver_id:str
    amount:float=Field(gt=0)
    merchant_category:str="Unknown"
    transaction_type:str="P2P"
    location:str|None=None
    device_type:str|None=None
    upi_channel:str|None=None

@asynccontextmanager
async def lifespan(app):
    art=joblib.load(ARTIFACT_PATH)
    STATE.update(art)
    os.makedirs(DATA_DIR,exist_ok=True)
    with sqlite3.connect(DB_PATH) as db:
        db.execute("CREATE TABLE IF NOT EXISTS flagged_transactions(transaction_id TEXT PRIMARY KEY,scored_at TEXT,sender_id TEXT,amount REAL,risk_level TEXT,fraud_score REAL)")
        db.commit()
    yield

app=FastAPI(title="UPI Fraud Detection API",version="1.0",lifespan=lifespan)

@app.get("/health")
def health():
    return {"status":"ok","users_known":len(STATE["profiles"]["users"])}

@app.get("/")
def root():
    return FileResponse("app/static/dashboard.html")

@app.get("/dashboard")
def dashboard():
    return FileResponse("app/static/dashboard.html")

@app.post("/score")
def score(tx:Transaction):
    prof=STATE["profiles"]["users"].get(tx.sender_id,STATE["profiles"]["default"])
    ts=pd.Timestamp(tx.timestamp or datetime.now(timezone.utc).replace(tzinfo=None))
    row=pd.DataFrame([{"transaction_id":tx.transaction_id or f"TXN-{uuid.uuid4().hex[:12]}","timestamp":ts,"sender_id":tx.sender_id,"receiver_id":tx.receiver_id,"amount":tx.amount,"device_type":tx.device_type or prof["usual_device"] or "Android","location":tx.location or prof["home_location"] or "Unknown"}])
    hist=STATE["history"].get(tx.sender_id)
    full=row if hist is None else pd.concat([hist,row],ignore_index=True)
    f=_user_features(full,prof)
    f["amount_log"]=np.log1p(f.amount)
    f["is_night"]=f.timestamp.dt.hour.between(0,5).astype(int)
    cur=f.tail(1)
    r=STATE["detector"].predict(cur).iloc[0]
    methods=[n for n,v in [("IQR",r.iqr_flag),("IsolationForest",r.if_flag),("TimeSeries",r.ts_flag)] if v]
    out={"transaction_id":row.transaction_id.iloc[0],"risk_level":str(r.risk_level),"fraud_score":float(r.fraud_score),"alert":bool(r.n_methods_flagged>=2),"needs_review":bool(r.n_methods_flagged>=1),"methods_flagged":methods}
    if out["needs_review"]:
        with sqlite3.connect(DB_PATH) as db:
            db.execute("INSERT OR REPLACE INTO flagged_transactions VALUES(?,?,?,?,?,?)",(out["transaction_id"],datetime.now(timezone.utc).isoformat(),tx.sender_id,float(tx.amount),out["risk_level"],out["fraud_score"]))
            db.commit()
    return out

@app.get("/alerts")
def alerts(limit:int=50):
    with sqlite3.connect(DB_PATH) as db:
        rows=db.execute("SELECT * FROM flagged_transactions ORDER BY scored_at DESC LIMIT ?",(limit,)).fetchall()
    return [{"transaction_id":r[0],"scored_at":r[1],"sender_id":r[2],"amount":r[3],"risk_level":r[4],"fraud_score":r[5]} for r in rows]
