import sys
import os
import requests

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from datetime import date
from typing import Any, Dict

import pickle
import subprocess
from pathlib import Path

from fastapi import APIRouter, Query, HTTPException
from fastapi.encoders import jsonable_encoder

from puppy import LLMAgent
from past_report import filter_reports
from dotenv import load_dotenv

import yfinance as yf

load_dotenv(override=True)
hf_token = os.getenv("HF_TOKEN")
stocknewsapi_api_key = os.getenv("STOCKNEWSAPI_API_KEY")

router = APIRouter()

@router.get("/report")
async def report(
    ticker: str = Query(..., description="티커 심볼"),
    is_today: bool = Query(False, description="오늘 리포트 여부, true면 오늘만"),
    start_date: date = Query(None, description="시작일 (YYYY-MM-DD)"),
    end_date:   date = Query(None, description="종료일 (YYYY-MM-DD)"),
) -> Dict[str, Any]:
    
    input_path = f"data/03_model_input/add_filing_{ticker.lower()}.pkl"
    output_path = f"data/05_model_output/{ticker.lower()}_4o_mini_run"

    if is_today:
        print('today')
        # set today's date
        today = date.today()

        if input_path:
            with open(input_path, "rb") as pf:
                report_dict: Dict[date, Any] = pickle.load(pf)
        else:
            report_dict = {}
        
        if today in report_dict:
            report = report_dict[today]
        else: 
            params = {
            'tickers': ticker,
            'items': 10,
            'page': 1,
            'token': stocknewsapi_api_key
            }
            res = requests.get('https://stocknewsapi.com/api/v1', params=params)
            res.raise_for_status()
            today_articles = []
            today_data = res.json().get('data', [])
            today_articles = [article.get("text", "") for article in today_data]

            yf_tkr = yf.Ticker(ticker)
            hist = yf_tkr.history(period="1d") 
            if hist.empty:
                price_val = None
            else:
                price_val = float(hist["Close"].iloc[-1])

            report = {
                "price": {ticker.upper(): price_val},
                "news": {ticker.upper(): today_articles},
                "filing_q": "",
                "filing_k": ""
            }
            report_dict[today] = report
            with open(input_path, "wb") as pf:
                pickle.dump(report_dict, pf)
        
            try:
                project_root = Path(__file__).parent.parent.resolve()
                script_path = project_root / "run_openai.sh"
                if not script_path.exists():
                    raise RuntimeError(f"스크립트가 없습니다: {script_path}")
                print("실행 명령어:", " ".join([
                    "python",
                    "run.py",
                    "--config", f"{ticker.lower()}_gpt_config.toml",
                    "--market-data", f"{input_path}",
                    "--start-date", "2021-11-16",
                    "--end-date", date.today().isoformat(),
                    "--output-path", str(project_root / f"data/05_model_output/{ticker.lower()}_4o_mini_run"),
                    "--update-market",
                ]))

                result = subprocess.run(
                    [
                        "python",
                        "run.py",
                        "--config", f"config/{ticker.lower()}_gpt_config.toml",
                        "--market-data", f"{input_path}",
                        "--start-date", "2021-11-16",
                        "--end-date", date.today().isoformat(),
                        "--output-path", str(project_root / f"data/05_model_output/{ticker.lower()}_4o_mini_run"),
                        "--update-market",
                    ],
                    cwd=str(project_root), 
                    capture_output=False,
                    text=True,
                    check=True
                )
                print(result.stdout)
                print(result.stderr)
                print('b')
            except subprocess.CalledProcessError as e:
                return {"error": e.stderr, "returncode": e.returncode}
            
    if start_date is None or end_date is None:
        raise HTTPException(status_code=400, detail="start_date와 end_date를 모두 지정하세요.")

    agent_checkpoint_path = os.path.join(output_path, "agent")
    print('load agent')
    try:
        agent = LLMAgent.load_checkpoint(path=agent_checkpoint_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent 로드 실패: {e}")
    print('load full report')
    full_report: Dict[date, Any] = agent.reflection_result_series_dict
    filtered = filter_reports(full_report, start_date, end_date)

    return jsonable_encoder(filtered)
