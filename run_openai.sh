#!/bin/bash

set -e

# --- 파라미터 설정 ---
START_DATE=${1:-"2022-10-06"}
END_DATE=${2:-"2023-04-10"}
OUTPUT_PATH=${3:-"data/05_model_output/tsla_default_run"}


# 사용할 설정 파일과 데이터 파일 경로
CONFIG_FILE="config/tsla_gpt_config.toml"
MARKET_DATA_FILE="data/03_model_input/add_filing_tsla.pkl"


echo "=================================================="
echo "Starting FinMem Simulation"
echo "--------------------------------------------------"
echo "Trading Symbol: (from config)"
echo "Config File:    $CONFIG_FILE"
echo "Market Data:    $MARKET_DATA_FILE"
echo "Simulation Period: $START_DATE to $END_DATE"
echo "Output & Checkpoint Path: $OUTPUT_PATH"
echo "=================================================="
echo ""

python run.py \
    --config "$CONFIG_FILE" \
    --market-data "$MARKET_DATA_FILE" \
    --start-date "$START_DATE" \
    --end-date "$END_DATE" \
    --output-path "$OUTPUT_PATH"

echo ""
echo "=================================================="
echo "Simulation script finished."
echo "Final results are saved in: $OUTPUT_PATH"
echo "=================================================="


 python save_file.py \
    --output-path data/05_model_output/tsla_default_run \
    --portfolio-out tsla_portfolio_new.csv \
    --analysis-out tsla_full_report.json \
    --performance-out tsla_performance.json