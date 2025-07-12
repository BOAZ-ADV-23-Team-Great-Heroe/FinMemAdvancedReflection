import os
import configparser
import typer
import logging
import pickle
import warnings
from tqdm import tqdm
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, Any
from puppy import MarketEnvironment, LLMAgent

warnings.filterwarnings("ignore")
load_dotenv(override=True)

def load_config(config_path: str) -> Dict[str, Any]:
    """ .ini 설정 파일을 읽어 중첩된 딕셔너리로 변환합니다. """
    parser = configparser.ConfigParser()
    parser.read(config_path, encoding='utf-8')
    config_dict = {}
    for section in parser.sections():
        keys = section.split('.')
        d = config_dict
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = dict(parser.items(section))
    return config_dict

def run_simulation(
    config_path: str = typer.Option(
        "config/aapl_gpt_config.ini",
        "-c", "--config",
        help="Path to the INI configuration file."
    ),
    market_data_path: str = typer.Option(
        "data/03_model_input/add_filing_aapl.pkl",
        "-m", "--market-data",
        help="Path to the market data pickle file."
    ),
    start_date_str: str = typer.Option(
        "2022-10-06",
        "-s", "--start-date",
        help="Simulation start date in YYYY-MM-DD format."
    ),
    end_date_str: str = typer.Option(
        "2023-04-10",
        "-e", "--end-date",
        help="Simulation end date in YYYY-MM-DD format."
    ),
    output_path: str = typer.Option(
        "result/default_run",
        "-o", "--output-path",
        help="Directory to save all checkpoints and final results. Resumes from here if checkpoints exist."
    ),
):
    """
    에이전트 시뮬레이션을 시작하거나 체크포인트에서 재개합니다.
    """
    os.makedirs(output_path, exist_ok=True)
    
    log_file_path = os.path.join(output_path, "run.log")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logging_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_file_path, mode="a")
    file_handler.setFormatter(logging_formatter)
    
    if not logger.handlers:
        logger.addHandler(file_handler)

    config = load_config(config_path)

    agent_checkpoint_path = os.path.join(output_path, "agent")
    env_checkpoint_path = os.path.join(output_path, "env")

    if os.path.exists(agent_checkpoint_path) and os.path.exists(env_checkpoint_path):
        logger.info(f"Resuming simulation from checkpoint at '{output_path}'")
        environment = MarketEnvironment.load_checkpoint(path=env_checkpoint_path)
        the_agent = LLMAgent.load_checkpoint(path=agent_checkpoint_path, config=config)
    else:
        logger.info("No checkpoint found. Starting a new simulation.")
        with open(market_data_path, "rb") as f:
            env_data_pkl = pickle.load(f)
        
        environment = MarketEnvironment(
            symbol=config["general"]["trading_symbol"],
            env_data_pkl=env_data_pkl,
            start_date=datetime.strptime(start_date_str, "%Y-%m-%d").date(),
            end_date=datetime.strptime(end_date_str, "%Y-%m-%d").date(),
        )
        the_agent = LLMAgent.from_config(config)

    with tqdm(total=environment.simulation_length, initial=environment.current_step, desc="Simulating Agent") as pbar:
        while not environment.is_done:
            market_info = environment.step()
            if market_info[-1]:
                logger.info("Reached the end of the simulation period.")
                break
            
            the_agent.step(market_info=market_info)
            pbar.update(1)
            
            if the_agent.counter % 50 == 0:
                the_agent.save_checkpoint(path=agent_checkpoint_path, force=True)
                environment.save_checkpoint(path=env_checkpoint_path, force=True)
    
    the_agent.save_checkpoint(path=agent_checkpoint_path, force=True)
    environment.save_checkpoint(path=env_checkpoint_path, force=True)

    logger.info("Simulation completed successfully.")
    logger.info(f"Final results and checkpoints are saved in '{output_path}'.")

if __name__ == "__main__":
    typer.run(run_simulation)