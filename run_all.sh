#!/usr/bin/env bash
set -e
pip install -r requirements.txt
python -m upi_fraud.train
pytest -v
uvicorn app.main:app --reload
