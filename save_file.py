import os
import json
import typer
import pickle
import polars as pl
from decimal import Decimal
from puppy import LLMAgent
from typing import Dict, Any
from datetime import date


class CustomEncoder(json.JSONEncoder):
    """
    JSON으로 저장할 수 없는 추가 데이터 타입(date, Decimal)을 처리하는 커스텀 인코더.
    """
    def default(self, obj):
        if isinstance(obj, date):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj) # Decimal 객체를 float으로 변환
        return super().default(obj)

def save_simulation_results(
    output_path: str = typer.Option(
        ..., 
        "--output-path", "-o",
        help="The output directory of the simulation run (e.g., 'data/05_model_output/default_run')."
    ),
    portfolio_csv_path: str = typer.Option(
        "final_portfolio_actions.csv",
        "--portfolio-out", "-p",
        help="Path to save the final portfolio actions CSV file."
    ),
    performance_json_path: str = typer.Option(
        "final_performance.json",
        "--performance-out", "-m",
        help="Path to save the performance metrics JSON file."
    ),
    analysis_json_path: str = typer.Option(
        "final_analysis_report.json",
        "--analysis-out", "-a",
        help="Path to save the full JSON analysis report."
    )
):
    """
    시뮬레이션이 끝난 에이전트를 불러와,
    1. 포트폴리오 거래 내역 (CSV)
    2. 최종 성과 지표 (JSON)
    3. 전체 분석 리포트 (JSON)
    를 파일로 저장합니다.
    """
    agent_checkpoint_path = os.path.join(output_path, "agent")
    if not os.path.exists(agent_checkpoint_path):
        print(f"❌ 오류: 에이전트 체크포인트 경로를 찾을 수 없습니다 -> {agent_checkpoint_path}")
        raise typer.Exit(code=1)
    
    try:
        # 1. 최종 에이전트 로드
        print(f"'{agent_checkpoint_path}'에서 최종 에이전트 상태를 불러옵니다...")
        agent = LLMAgent.load_checkpoint(path=agent_checkpoint_path)
        print("✅ 에이전트 로드 완료!")

        # 2. 포트폴리오 상세 거래 내역 저장
        print(f"\n포트폴리오 거래 내역을 '{portfolio_csv_path}' 파일로 저장합니다...")
        portfolio_df = agent.portfolio.get_action_df()
        if not portfolio_df.is_empty():
            portfolio_df.write_csv(portfolio_csv_path)
            print(f"✅ 포트폴리오 거래 내역 저장 완료!")
        else:
            print("⚠️ 거래 내역이 없어 포트폴리오 파일을 저장하지 않았습니다.")

        # 3. 전문 성과 지표 계산 및 JSON 저장
        print(f"\n성과 지표를 계산하여 '{performance_json_path}' 파일로 저장합니다...")
        performance_metrics = agent.portfolio.get_performance_metrics()
        with open(performance_json_path, 'w', encoding='utf-8') as f:
            json.dump(performance_metrics, f, ensure_ascii=False, indent=4)
        print("✅ 성과 지표:")
        for key, value in performance_metrics.items():
            print(f"  - {key}: {value}")

        # 4. 전체 분석 리포트 JSON 저장
        print(f"\n전체 분석 리포트를 '{analysis_json_path}' 파일로 저장합니다...")
        full_report = agent.reflection_result_series_dict
        
        report_for_json = {
            dt.isoformat(): report 
            for dt, report in full_report.items()
        }
        
        with open(analysis_json_path, 'w', encoding='utf-8') as f:
            # --- 최종 수정된 부분: 확장된 CustomEncoder 사용 ---
            json.dump(report_for_json, f, ensure_ascii=False, indent=4, cls=CustomEncoder)
        print(f"✅ 전체 분석 리포트 저장 완료!")

    except Exception as e:
        print(f"오류 발생: {e}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    typer.run(save_simulation_results)