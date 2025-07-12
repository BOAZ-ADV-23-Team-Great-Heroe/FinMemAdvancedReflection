import os
import json
import pickle
import typer
import polars as pl
from decimal import Decimal
from puppy import LLMAgent
from typing import Dict, Any
from datetime import date
from pathlib import Path


class CustomEncoder(json.JSONEncoder):
    """
    JSON으로 저장할 수 없는 추가 데이터 타입(date, Decimal)을 처리하는 커스텀 인코더.
    """
    def default(self, obj):
        if isinstance(obj, date):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)  # Decimal 객체를 float으로 변환
        return super().default(obj)


def filter_reports(
    reports: Dict[date, Any],
    start_date: date,
    end_date: date
) -> Dict[str, Any]:
    """
    date-keys를 가진 dict에서 start_date ≤ key ≤ end_date 인 항목만
    isoformat된 문자열 key로 리턴한다.
    """
    return {
        dt.isoformat(): report
        for dt, report in reports.items()
        if start_date <= dt <= end_date
    }


def main(
    ticker: str,
    start_date: date = typer.Option(..., help="시작일 (YYYY-MM-DD)"),
    end_date: date = typer.Option(..., help="종료일 (YYYY-MM-DD)"),
):
    # 체크포인트 로드
    output_path = 'data/05_model_output/nvda_4o_mini_run'
    agent_checkpoint_path = os.path.join(output_path, "agent")
    agent = LLMAgent.load_checkpoint(path=agent_checkpoint_path)

    full_report: Dict[date, Any] = agent.reflection_result_series_dict

    # 날짜 범위로 필터링
    filtered = filter_reports(full_report, start_date, end_date)


if __name__ == "__main__":
    typer.run(main)
