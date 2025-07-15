#!/bin/bash

set -e

# --- 파라미터 설정 ---
START_DATE=${1:-"2021-11-16"}
END_DATE=${2:-"2021-11-17"}
OUTPUT_PATH=${3:-"data/05_model_output/tsla_4o_mini_run"}


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


TRADING_SYMBOL=$(basename "$CONFIG_FILE" | cut -d'_' -f1)

# 추출된 심볼과 OUTPUT_PATH 변수를 사용하여 동적으로 파일 이름을 생성합니다.
python save_file.py \
    --output-path "$OUTPUT_PATH" \
    --portfolio-out "${TRADING_SYMBOL}_portfolio_4o_mini.csv" \
    --analysis-out "${TRADING_SYMBOL}_full_report_4o_mini.json" \
    --performance-out "${TRADING_SYMBOL}_performance_4o_mini.json"