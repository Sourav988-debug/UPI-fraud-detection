import json, joblib, matplotlib
matplotlib.use("Agg")
from .config import *
from .generate_data import generate
from .preprocess import preprocess
from .features import build_profiles, build_features
from .detectors import FraudDetector

def main():
    for d in (DATA_DIR,MODEL_DIR,REPORT_DIR): d.mkdir(exist_ok=True)
    raw=generate(); raw.to_csv(RAW_PATH,index=False)
    clean=preprocess(raw,verbose=False)
    profiles=build_profiles(clean)
    feats=build_features(clean,profiles)
    det=FraudDetector().fit(feats)
    res=det.predict(feats)
    scored=__import__("pandas").concat([feats,res],axis=1)
    scored.to_csv(SCORED_PATH,index=False)
    joblib.dump({"detector":det,"profiles":profiles,"history":{sid:g.tail(100)[["transaction_id","timestamp","sender_id","receiver_id","amount","device_type","location"]] for sid,g in clean.groupby("sender_id")}},ARTIFACT_PATH)
    metrics={"rows":len(clean),"fraud_rows":int(clean.is_anomaly.sum()),"flagged":int((res.n_methods_flagged>=1).sum())}
    (REPORT_DIR/"metrics.json").write_text(json.dumps(metrics,indent=2))
    print(metrics)
if __name__=="__main__": main()
