# UPI Fraud Detection

An end-to-end UPI fraud detection system built around behavioural profiling, statistical outlier detection, machine-learning anomaly detection, time-series signals, and a FastAPI scoring service.

## Live Application

- **Live dashboard:** https://upi-fraud-detection-hkhd.onrender.com
- **API documentation:** https://upi-fraud-detection-hkhd.onrender.com/docs
- **Health check:** https://upi-fraud-detection-hkhd.onrender.com/health

The deployed service trains its model during the Docker build from reproducible synthetic data, so the repository does not need to store the generated model artifact or CSV datasets.

## Features

- Synthetic UPI transaction generation for reproducible development and testing
- Per-user behavioural profiles
- Amount and spending deviation features
- Transaction-frequency and recipient-repetition features
- Device and location change signals
- IQR-based outlier detection
- Isolation Forest anomaly detection
- Rolling time-series anomaly detection
- Combined LOW / MEDIUM / HIGH risk classification
- FastAPI transaction scoring API
- Live fraud-monitoring dashboard
- SQLite storage for transactions flagged during live scoring
- Docker and Render deployment
- Automated API tests

## Detection Logic

Three independent detectors contribute to the final decision:

1. **IQR** identifies unusually large amounts, behavioural deviations, and transaction-frequency outliers.
2. **Isolation Forest** evaluates multiple behavioural and contextual features together.
3. **Time-Series detection** looks for sudden changes against the user's recent transaction behaviour.

The combined logic is:

- **0 detectors flagged:** LOW
- **1 detector flagged:** MEDIUM
- **2 or more detectors flagged:** HIGH

The displayed fraud score is the fraction of the three detectors that flagged the transaction.

## Architecture

```
Synthetic Transactions
        |
        v
   Preprocessing
        |
        v
Behavioural Features
        |
        +------------------+
        |                  |
        v                  v
       IQR          Isolation Forest
        |                  |
        +--------+---------+
                 |
                 v
          Time-Series Check
                 |
                 v
       Combined Risk Engine
                 |
        +--------+---------+
        |                  |
        v                  v
    FastAPI API       Live Dashboard
        |
        v
 Flagged Transactions
      (SQLite)
```

## Repository Structure

```
.
├── app/
│   ├── main.py
│   └── static/
│       └── dashboard.html
├── upi_fraud/
│   ├── config.py
│   ├── detectors.py
│   ├── features.py
│   ├── generate_data.py
│   ├── preprocess.py
│   └── train.py
├── tests/
│   └── test_api.py
├── Dockerfile
├── .dockerignore
├── .gitignore
├── render.yaml
├── railway.json
├── requirements.txt
└── README.md
```

Generated files such as CSV datasets, the trained Joblib artifact, reports, and the live SQLite database are created locally or during deployment and are intentionally excluded from the repository.

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Live dashboard |
| `/dashboard` | GET | Live dashboard |
| `/health` | GET | Service health and known-user count |
| `/score` | POST | Score a UPI transaction |
| `/alerts` | GET | Return recently flagged live transactions |
| `/docs` | GET | Interactive Swagger API documentation |

### Example transaction

```json
{
  "sender_id": "U0001",
  "receiver_id": "M0001",
  "amount": 5000,
  "location": "Mumbai",
  "device_type": "Android",
  "transaction_type": "P2P",
  "upi_channel": "GPay"
}
```

## Tech Stack

**Backend:** Python, FastAPI, Uvicorn

**Data / ML:** Pandas, NumPy, scikit-learn, Joblib

**Dashboard:** HTML, CSS, JavaScript

**Database:** SQLite

**Deployment:** Docker, Render

**Testing:** Pytest, FastAPI TestClient

## Run Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate data and train the model

```python
python -m upi_fraud.train
```

### 3. Start the API

```bash
uvicorn app.main:app --reload
```

Then open:

- Dashboard: http://localhost:8000
- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### 4. Run tests

```bash
pytest -v
```

## Docker

Build and run the same container used for deployment:

```bash
docker build -t upi-fraud-detection .
docker run -p 8000:8000 upi-fraud-detection
```

## Deployment

The project includes:

- `Dockerfile` for the application container
- `render.yaml` for Render deployment configuration
- Automatic model/data generation during the image build
- `/health` as the service health-check endpoint

The current live deployment is hosted on Render:

https://upi-fraud-detection-hkhd.onrender.com

## Important Limitation

This project uses **synthetic transaction data**. Its fraud labels and evaluation results are intended for academic demonstration and development, not for measuring production fraud-detection performance.

A production system would require validated real-world transaction data, stronger data governance, model monitoring, threshold calibration, authentication/authorization, secure secrets management, persistent production storage, and appropriate privacy and compliance controls.

## Project Status

**Deployment:** Live on Render

**Dashboard:** Live

**API:** Live

**Automated tests:** Included

**Data:** Synthetic

