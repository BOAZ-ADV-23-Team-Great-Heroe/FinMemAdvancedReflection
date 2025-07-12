from fastapi import APIRouter
import subprocess

router = APIRouter()

@router.get("/report")
async def report(ticker: str, is_today: str, start_date: str, end_date: str, output_path: str):
    config_file = f"config/{ticker}_gpt_config.toml"
    market_data_file = "data/03_model_input/add_filing_{ticker}.pkl"

    if is_today == "true":
        return {
            'message': 'not yet'
        }
    else:
        try:
            past_report = subprocess.run(
                [
                    "python", "past_report.py",
                    "--ticker", ticker,
                    "--start-date", start_date,
                    "--end-date", end_date
                ],
                capture_output=True,
                text=True,
                check=True
            )

            return {
                "report_output": past_report.stdout,
            }

        except subprocess.CalledProcessError as e:
            return {
                "error": e.stderr,
                "returncode": e.returncode
            }
