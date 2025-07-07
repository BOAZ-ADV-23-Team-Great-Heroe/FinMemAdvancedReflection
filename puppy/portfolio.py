import polars as pl
import numpy as np
from datetime import date
from typing import Dict, List, Union

class Portfolio:
    """
    고도화된 포트폴리오 관리 클래스.
    - 현금, 주식, 총자산 가치 관리
    - 비율 기반 포지션 사이징 및 거래 비용(수수료) 시뮬레이션
    - 일별 포트폴리오 가치 및 수익률 기록
    - 전문적인 성과 지표 (샤프 지수, 최대 낙폭) 계산
    """
    def __init__(
        self,
        symbol: str,
        initial_capital: float = 1_000_000.0,
        commission_rate: float = 0.001,
        lookback_window_size: int = 7
    ):
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.lookback_window_size = lookback_window_size

        self.cash = initial_capital
        self.holding_shares = 0
        self.market_price = 0.0
        self.total_value = initial_capital

        self.cur_date: Union[date, None] = None
        self.date_series: List[date] = []
        self.action_history: List[Dict] = []
        self.value_history: List[float] = [initial_capital]

    def update_market_info(self, new_market_price: float, cur_date: date) -> None:
        """새로운 시장 가격과 날짜를 받아 포트폴리오 상태를 업데이트합니다."""
        self.cur_date = cur_date
        self.market_price = new_market_price
        self.date_series.append(cur_date)

        current_stock_value = self.holding_shares * self.market_price
        self.total_value = self.cash + current_stock_value
        self.value_history.append(self.total_value)

    def _calculate_trade_quantity(self, position_sizing: float) -> int:
        """
        포지션 사이징 비율에 따라 거래할 주식 수를 계산합니다.
        """
        if self.market_price <= 0:
            return 0
        # +++ 최종 수정된 부분: position_sizing을 float으로 변환하여 타입 오류 해결 +++
        target_amount = self.total_value * float(position_sizing)
        return int(target_amount // self.market_price)

    def record_action(self, action: Dict[str, Union[str, float]]) -> None:
        """
        ActionableTradePlan을 받아 거래를 실행하고 자산을 업데이트합니다.
        """
        decision = action.get("investment_decision")
        position_sizing = float(action.get("position_sizing", 0.0))

        trade_executed = False
        direction = 0
        quantity = 0
        commission = 0.0

        if decision == "buy":
            quantity = self._calculate_trade_quantity(position_sizing)
            trade_cost = quantity * self.market_price
            commission = trade_cost * self.commission_rate
            total_cost = trade_cost + commission

            if quantity > 0 and self.cash >= total_cost:
                self.holding_shares += quantity
                self.cash -= total_cost
                direction = 1
                trade_executed = True
        
        elif decision == "sell":
            quantity = self._calculate_trade_quantity(position_sizing)
            
            if quantity > 0 and self.holding_shares > 0:
                sell_quantity = min(quantity, self.holding_shares)
                trade_revenue = sell_quantity * self.market_price
                commission = trade_revenue * self.commission_rate
                
                self.holding_shares -= sell_quantity
                self.cash += (trade_revenue - commission)
                direction = -1
                trade_executed = True

        if trade_executed:
            self.action_history.append({
                "date": self.cur_date, "symbol": self.symbol, "direction": direction,
                "quantity": quantity, "price": self.market_price,
                "commission": commission, "portfolio_value": self.total_value,
            })

    def get_action_df(self) -> pl.DataFrame:
        """거래 기록을 polars DataFrame으로 반환합니다."""
        if not self.action_history:
            return pl.DataFrame()
        return pl.DataFrame(self.action_history)
        
    def get_performance_metrics(self) -> Dict[str, float]:
        """시뮬레이션 종료 후 전체 기간에 대한 성과 지표를 계산합니다."""
        if len(self.value_history) < 2:
            return {
                "final_portfolio_value": self.total_value, "total_return_pct": 0,
                "sharpe_ratio": 0, "max_drawdown_pct": 0
            }
        
        portfolio_values = np.array(self.value_history)
        daily_returns = (portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[:-1]
        
        total_return_pct = ((self.total_value - self.initial_capital) / self.initial_capital) * 100

        if np.std(daily_returns) > 0:
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
            
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown_pct = np.min(drawdown) * 100 if len(drawdown) > 0 else 0.0

        return {
            "final_portfolio_value": round(self.total_value, 2),
            "total_return_pct": round(total_return_pct, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2)
        }

    def get_feedback_response(self) -> Union[Dict[str, Union[int, date]], None]:
        """최근 lookback_window_size 구간의 포트폴리오 가치 변화를 기반으로 피드백을 생성합니다."""
        if len(self.value_history) <= self.lookback_window_size:
            return None

        value_change = self.value_history[-1] - self.value_history[-self.lookback_window_size]
        
        feedback = 0
        if value_change > 0.001:  # 유의미한 수익이 났을 때
            feedback = 1
        elif value_change < -0.001: # 유의미한 손실이 났을 때
            feedback = -1
            
        return {
            "feedback": feedback,
            "date": self.date_series[-self.lookback_window_size],
        }