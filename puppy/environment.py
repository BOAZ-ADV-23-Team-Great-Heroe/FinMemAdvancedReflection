import os
import pickle
import shutil
from datetime import date
from typing import List, Dict, Tuple, Union, Any

from pydantic import BaseModel, ValidationError

# 타입 별칭: 시뮬레이션에서 한 스텝의 마켓 정보 구조 정의
# agent.py의 step 함수가 받는 인자 순서와 타입에 맞춤
# (cur_date, cur_price, cur_news, cur_filing_k, cur_filing_q, is_done)
market_info_type = Tuple[
    date,
    float,
    Union[List[str], None],
    Union[Dict[str, str], None],
    Union[Dict[str, str], None],
    bool,
]
terminated_market_info_type = Tuple[None, None, None, None, None, bool]


class MarketEnvironment:
    """
    시뮬레이션 환경 클래스. 날짜별로 가격, 공시, 뉴스 등 환경 데이터를 관리하며 step()으로 한 스텝씩 진행.
    """
    def __init__(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        env_data_pkl: Dict[date, Dict[str, Any]],
    ):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.env_data = env_data_pkl

        # 시뮬레이션 구간에 해당하는 날짜만 필터링 및 정렬
        self.date_series = sorted([
            dt for dt in self.env_data.keys() 
            if start_date <= dt <= end_date
        ])
        
        self.simulation_length = len(self.date_series)
        self.current_step = 0
        self.is_done = False

    def _get_market_info_for_date(self, target_date: date) -> market_info_type:
        """특정 날짜에 대한 시장 정보를 가져옵니다."""
        
        daily_data = self.env_data.get(target_date, {})
        
        cur_price = daily_data.get("price", {}).get(self.symbol, 0.0)
        
        news_data = daily_data.get("news", {}).get(self.symbol)
        cur_news = news_data if isinstance(news_data, list) else None

        filing_k_data = daily_data.get("filing_k", "")
        cur_filing_k = {self.symbol: filing_k_data} if filing_k_data else None
        
        filing_q_data = daily_data.get("filing_q", "")
        cur_filing_q = {self.symbol: filing_q_data} if filing_q_data else None

        return (
            target_date,
            cur_price,
            cur_news,
            cur_filing_k,
            cur_filing_q,
            self.is_done,
        )

    def step(self) -> Union[market_info_type, terminated_market_info_type]:
        """시뮬레이션을 한 스텝 진행. 다음 날짜로 이동하며 시장 정보를 반환합니다."""
        if self.current_step >= self.simulation_length:
            self.is_done = True
            return None, None, None, None, None, True

        # +++ 최종 수정된 부분: self.dates -> self.date_series로 변경 +++
        cur_date = self.date_series[self.current_step]
        self.current_step += 1
        
        if self.current_step >= self.simulation_length:
            self.is_done = True

        market_data = self._get_market_info_for_date(cur_date)
        # _get_market_info_for_date의 반환값은 6개 튜플이므로, is_done을 제외하고 반환
        date_val, price, news, filing_k, filing_q, _ = market_data
        
        # agent.py의 step 함수가 받는 인자 순서에 맞게 재구성
        return (date_val, price, news, filing_k, filing_q, self.is_done)

    def reset(self) -> None:
        """시뮬레이션을 초기 상태로 리셋합니다."""
        self.current_step = 0
        self.is_done = False

    def save_checkpoint(self, path: str, force: bool = False) -> None:
        """현재 환경의 상태를 파일로 저장합니다."""
        if os.path.exists(path) and not force:
            raise FileExistsError(f"Path {path} already exists")
        if os.path.exists(path):
            shutil.rmtree(path)
        
        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, "env.pkl"), "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_checkpoint(cls, path: str) -> "MarketEnvironment":
        """저장된 체크포인트에서 환경 객체를 복원합니다."""
        with open(os.path.join(path, "env.pkl"), "rb") as f:
            return pickle.load(f)
