# UPI Fraud Detection

End-to-end synthetic UPI fraud detection project using behavioural features, IQR outlier detection, Isolation Forest, time-series anomaly detection, and a FastAPI scoring service.

## Structure

- `upi_fraud/generate_data.py` - synthetic transaction generation
- `upi_fraud/preprocess.py` - cleaning and preprocessing
- `upi_fraud/features.py` - behavioural and time-series features
- `upi_fraud/detectors.py` - IQR, Isolation Forest, time-series and combined risk logic
- `upi_fraud/train.py` - training/evaluation pipeline
- `app/main.py` - FastAPI backend
- `app/static/dashboard.html` - fraud monitoring dashboard
- `tests/test_api.py` - API tests
- `data/` - generated transaction datasets
- `models/` - trained model artifact
- `reports/` - evaluation metrics and charts

## Run locally

```bash
pip install -r requirements.txt
python -m upi_fraud.train
pytest -v
uvicorn app.main:app --reload
```

Dashboard: `http://localhost:8000/dashboard`

API docs: `http://localhost:8000/docs`

## Detection

The project combines three signals:

1. IQR for explainable extreme-value detection.
2. Isolation Forest for multivariate transaction anomalies.
3. Rolling time-series checks for sudden behavioural changes.

A transaction flagged by one method is MEDIUM risk. Two or more methods produce HIGH risk.

## Evaluation

The supplied project evaluates detectors on a time-based holdout and reports precision, recall, F1, ROC-AUC and recall by fraud pattern in `reports/metrics.json`.

The data is synthetic. The reported metrics describe this generated dataset and should not be interpreted as production fraud-detection performance.

## Deployment

The repository includes a Dockerfile and Railway configuration for deployment.
