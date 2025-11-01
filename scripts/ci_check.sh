#!/usr/bin/env bash
set -euo pipefail
echo "[CI] Running data ingestion..."
python scripts/ingest_metrics.py
python scripts/make_tables.py || true
echo "[CI] Building paper..."
make -j || make
echo "[CI] OK"

