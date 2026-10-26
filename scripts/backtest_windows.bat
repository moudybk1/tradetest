@echo off
REM One-click backtest on Windows: install, download 2 years of XAUUSD M1, compare all strategies.
REM Double-click this file (or run it from the repo folder). Results are saved in data\results_*.txt
cd /d "%~dp0\.."

where py >nul 2>nul && (set PY=py -3) || (set PY=python)
if not exist .venv (
    echo Creating Python environment...
    %PY% -m venv .venv || (echo Python not found. Install it from https://www.python.org/downloads/ and tick "Add to PATH". & pause & exit /b 1)
)
call .venv\Scripts\activate
pip install -q -r requirements.txt || (echo pip install failed & pause & exit /b 1)
if not exist config.yaml copy config.example.yaml config.yaml >nul
if not exist data mkdir data

if not exist data\xauusd_m1.csv (
    python -m xau_agent download-dukascopy --days 730 --out data\xauusd_m1.csv || (
        echo Download failed. With MT5 open you can use instead:
        echo   python -m xau_agent export-mt5 --days 730 --out data\xauusd_m1.csv
        pause & exit /b 1
    )
)

echo.
echo === M5 signals, CFD/MT5 costs (spread from data + 0.5 pip slippage) ===
python -m xau_agent compare --csv data\xauusd_m1.csv --split 2026-01-01 > data\results_m5.txt
type data\results_m5.txt
echo.
echo === M1 signals ===
python -m xau_agent compare --csv data\xauusd_m1.csv --timeframe 1m --split 2026-01-01 > data\results_m1.txt
type data\results_m1.txt
echo.
echo === M5 signals with Bybit perpetual taker fee (0.055%% per side) ===
python -m xau_agent compare --csv data\xauusd_m1.csv --fee-pct 0.055 --split 2026-01-01 > data\results_m5_bybit_fee.txt
type data\results_m5_bybit_fee.txt
echo.
echo Saved: data\results_m5.txt, data\results_m1.txt, data\results_m5_bybit_fee.txt
pause
