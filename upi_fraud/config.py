"""Shared constants for the UPI fraud detection project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

RAW_PATH = DATA_DIR / "upi_transactions_raw.csv"
PROCESSED_PATH = DATA_DIR / "upi_transactions_processed.csv"
SCORED_PATH = DATA_DIR / "upi_transactions_scored.csv"
ARTIFACT_PATH = MODEL_DIR / "fraud_artifacts.joblib"

SEED = 42
N_TRANSACTIONS = 10_000

CITIES = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai",
          "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow"]
DEVICES = ["Android", "iOS", "Web"]
CHANNELS = ["GPay", "PhonePe", "Paytm", "BHIM", "AmazonPay"]
TXN_TYPES = ["P2P", "P2M", "Bill Payment", "Collect Request"]
MERCHANT_CATEGORIES = ["Groceries", "Food & Dining", "Transport", "Utilities",
                       "Shopping", "Entertainment", "Healthcare", "Education",
                       "Travel", "Person-to-Person", "Unknown"]

IF_FEATURES = [
    "amount_log", "amount_dev_from_user_avg", "amount_ratio",
    "txn_count_last_hour", "spend_last_24h_ratio", "time_since_last_txn_log",
    "recipient_repetition_count", "device_change", "location_change", "is_night",
]
IQR_K = 1.5
IQR_DEV_K = 3.0
IF_CONTAMINATION = 0.04
TS_Z_THRESHOLD = 3.0
