import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from datetime import date
from typing import Any, Dict

from fastapi import APIRouter, Query, HTTPException
from fastapi.encoders import jsonable_encoder

from puppy import LLMAgent
from past_report import filter_reports
from dotenv import load_dotenv

load_dotenv(override=True)
hf_token = os.getenv("HF_TOKEN")

router = APIRouter()

@router.get("/report")
async def report(
    ticker: str = Query(..., description="티커 심볼"),
    is_today: bool = Query(False, description="오늘 리포트 여부, true면 오늘만"),
    start_date: date = Query(None, description="시작일 (YYYY-MM-DD)"),
    end_date:   date = Query(None, description="종료일 (YYYY-MM-DD)"),
) -> Dict[str, Any]:
    if is_today:
        return {"message": "오늘 리포트 기능은 아직 구현되지 않았습니다."}

    if start_date is None or end_date is None:
        raise HTTPException(status_code=400, detail="start_date와 end_date를 모두 지정하세요.")

    output_path = f"data/05_model_output/{ticker.lower()}_4o_mini_run"
    agent_checkpoint_path = os.path.join(output_path, "agent")

    try:
        agent = LLMAgent.load_checkpoint(path=agent_checkpoint_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent 로드 실패: {e}")

    full_report: Dict[date, Any] = agent.reflection_result_series_dict
    filtered = filter_reports(full_report, start_date, end_date)

    return jsonable_encoder(filtered)
