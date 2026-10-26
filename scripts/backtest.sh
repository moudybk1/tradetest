#!/usr/bin/env bash
# One-click backtest on Mac/Linux: install, download 2 years of XAUUSD M1, compare all strategies.
# Run from anywhere: bash scripts/backtest.sh   Results are saved in data/results_*.txt
set -euo pipefail
cd "$(dirname "$0")/.."

[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements.txt
[ -f config.yaml ] || cp config.example.yaml config.yaml
mkdir -p data

if [ ! -f data/xauusd_m1.csv ]; then
    python -m xau_agent download-dukascopy --days 730 --out data/xauusd_m1.csv
fi

run() {  # run <title> <output file> <extra args...>
    local title=$1 out=$2; shift 2
    echo; echo "=== $title ==="
    python -m xau_agent compare --csv data/xauusd_m1.csv --split 2026-01-01 "$@" | tee "$out"
}
run "M5 signals, CFD/MT5 costs (spread from data + 0.5 pip slippage)" data/results_m5.txt
run "M1 signals" data/results_m1.txt --timeframe 1m
run "M5 signals with Bybit perpetual taker fee (0.055% per side)" data/results_m5_bybit_fee.txt --fee-pct 0.055
echo; echo "Saved: data/results_m5.txt, data/results_m1.txt, data/results_m5_bybit_fee.txt"
