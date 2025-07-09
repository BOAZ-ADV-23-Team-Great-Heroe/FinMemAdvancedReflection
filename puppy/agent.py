import os
import shutil
import pickle
import logging
import random 
from datetime import date
from decimal import Decimal
from .memorydb import BrainDB
from .portfolio import Portfolio
from abc import ABC, abstractmethod
from .chat import ChatOpenAICompatible
from .environment import market_info_type
from typing import Dict, Union, Any, List, Tuple
from .reflection import run_investment_committee_analysis, analyze_failed_trade
from transformers import AutoTokenizer

class TextTruncator:
    def __init__(self, tokenizer_name="roberta-base", max_length=8192):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def truncate(self, text: str) -> str:
        if not isinstance(text, str): return ""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > self.max_length: tokens = tokens[:self.max_length]
        return self.tokenizer.decode(tokens)

class Agent(ABC):
    @abstractmethod
    def from_config(self, config: Dict[str, Any]) -> "Agent": pass
    @abstractmethod
    def step(self, market_info: market_info_type) -> None: pass

class LLMAgent(Agent):
    def __init__(
        self,
        agent_name: str, trading_symbol: str, brain_db: BrainDB,
        chat_config: Dict[str, Any], portfolio_config: Dict[str, Any], 
        top_k: int = 7, look_back_window_size: int = 7,
        truncation_max_length: int = 8192
    ):
        self.counter = 1
        self.top_k = top_k
        self.agent_name = agent_name
        self.trading_symbol = trading_symbol
        self.look_back_window_size = look_back_window_size
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        log_path = os.path.join("data", "04_model_output_log", f"{self.trading_symbol}_run.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        handler = logging.FileHandler(log_path, mode="a")
        handler.setFormatter(formatter)
        if not self.logger.handlers: self.logger.addHandler(handler)
        
        self.brain = brain_db
        self.portfolio = Portfolio(
            symbol=self.trading_symbol,
            initial_capital=portfolio_config.get("initial_capital", 1_000_000.0),
            commission_rate=portfolio_config.get("commission_rate", 0.001),
            lookback_window_size=self.look_back_window_size
        )
        
        self.chat_config_save = chat_config.copy()
        self.base_system_message = chat_config.pop("system_message")
        self.endpoint_func = None # step에서 동적으로 생성
        self.reflection_result_series_dict = {}
        self.truncator = TextTruncator(max_length=truncation_max_length)
        self.failure_memory: List[str] = []
        self.consecutive_losses = 0

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LLMAgent":
        return cls(
            agent_name=config["general"]["agent_name"],
            trading_symbol=config["general"]["trading_symbol"],
            brain_db=BrainDB.from_config(config=config),
            top_k=config["general"].get("top_k", 7),
            chat_config=config["chat"],
            portfolio_config=config.get("portfolio", {}),
            look_back_window_size=config["general"].get("look_back_window_size", 7),
            truncation_max_length=config["general"].get("truncation_max_length", 8192)
        )

    def _handling_news(self, cur_date: date, news: Union[List[str], None]) -> None:
        if news and isinstance(news, list):
            for news_item in news:
                if news_item and isinstance(news_item, str) and news_item.strip():
                    self.brain.add_memory_short(self.trading_symbol, cur_date, news_item)

    def _handling_filings(self, cur_date: date, filing_q: Union[Dict, None], filing_k: Union[Dict, None]) -> None:
        if filing_q and isinstance(filing_q, dict):
            content = filing_q.get(self.trading_symbol)
            if content and isinstance(content, str) and content.strip():
                self.logger.info(f"Adding filing Q memory: {content[:100]}...")
                self.brain.add_memory_mid(self.trading_symbol, cur_date, content)
        
        if filing_k and isinstance(filing_k, dict):
            content = filing_k.get(self.trading_symbol)
            if content and isinstance(content, str) and content.strip():
                self.logger.info(f"Adding filing K memory: {content[:100]}...")
                self.brain.add_memory_long(self.trading_symbol, cur_date, content)

    def _reflect(self, cur_date: date) -> None:
        self.logger.info(f"'{self.trading_symbol}'에 대한 투자 위원회 분석 시작 (날짜: {cur_date})")
        
        # 1-1. 시장 국면 분석
        market_regime = self.portfolio.get_market_regime()
        regime_warning = ""
        if market_regime == "Bear":
            regime_warning = "**CRITICAL ALERT: The market is in a confirmed Bearish trend. Prioritize capital preservation. Be extremely skeptical of bullish narratives. Your default bias should be 'HOLD' or 'SELL'.**\n"
        elif market_regime == "Bull":
             regime_warning = "**Market Insight: The market is in a confirmed Bullish trend. Your primary focus is profit maximization. Actively seek compelling growth opportunities.**\n"
        
        # 1-2. "오답 노트" 준비
        recent_failures = "\n".join(self.failure_memory[-3:])
        failure_context = ""
        if recent_failures:
            failure_context = f"Before you begin, review and learn from these recent failed trades:\n{recent_failures}\n"

        # 1-3. 최종 컨텍스트 결합
        initial_context = regime_warning + failure_context + ("---\n" if (regime_warning or failure_context) else "")

        # 엔드포인트 함수를 동적 시스템 메시지로 재생성
        chat_config_copy = self.chat_config_save.copy()
        end_point = chat_config_copy.pop("end_point")
        model = chat_config_copy.pop("model")
        self.endpoint_func = ChatOpenAICompatible(
            end_point=end_point, model=model, system_message=self.base_system_message,
            other_parameters=chat_config_copy,
        ).guardrail_endpoint()
     
        persona_queries = {
            "value": "financial health, debt, cash flow, intrinsic value, margin of safety",
            "growth": "new products, market share, innovation, user growth, future catalyst",
            "technical": "stock chart, moving average, trading volume, market sentiment, short-term news"
        }
        retrieved_docs = {}
        for query in persona_queries.values():
            for query_func in [self.brain.query_short, self.brain.query_mid, self.brain.query_long]:
                texts, ids = query_func(query, self.top_k, self.trading_symbol)
                for doc_id, text in zip(ids, texts):
                    retrieved_docs[doc_id] = self.truncator.truncate(text)
        
        all_docs_with_ids = list(retrieved_docs.items())
        random.shuffle(all_docs_with_ids)
        
        reflection_result = run_investment_committee_analysis(
            endpoint_func=self.endpoint_func,
            short_memory_texts_with_ids=all_docs_with_ids,
            mid_long_memory_texts_with_ids=all_docs_with_ids,
            initial_context=initial_context
        )
        self.reflection_result_series_dict[cur_date] = reflection_result
        self.logger.info(f"분석 완료. 최종 투자 계획: {reflection_result.get('final_actionable_plan', '생성 실패')}")

    def _process_action(self, cur_date: date) -> Dict[str, Union[str, float]]:
        reflection_result = self.reflection_result_series_dict.get(cur_date, {})
        final_plan = reflection_result.get("final_actionable_plan")
        if not final_plan or "error" in final_plan:
            self.logger.warning(f"{cur_date}: 분석 실패 또는 오류로 '보유(hold)' 결정")
            return {"investment_decision": "hold", "position_sizing": 0.0}
        
        # --- 행동 강제 (Circuit Breaker) 로직 추가 ---
        if self.consecutive_losses >= 3 and final_plan.get("investment_decision") == "buy":
            self.logger.warning(f"CIRCUIT BREAKER ENGAGED: 3회 연속 손실로 '매수' 결정을 강제로 '보유'로 변경합니다.")
            final_plan["investment_decision"] = "hold"
            final_plan["primary_reason"] = "Circuit breaker triggered due to repeated losses. Overriding buy decision to manage risk."
        # ----------------------------------------------------------

        return {
            "investment_decision": final_plan.get("investment_decision", "hold"),
            "position_sizing": float(final_plan.get("position_sizing", 0.0))
        }

    def _update_access_counter(self):
        feedback = self.portfolio.get_feedback_response()
        if not feedback: return

        if feedback["feedback"] == 1:
            self.consecutive_losses = 0 # 성공 시, 연속 손실 카운터 리셋
        
        cur_date = feedback["date"]
        cur_reflection = self.reflection_result_series_dict.get(cur_date)
        if not cur_reflection: return

        final_plan = cur_reflection.get("final_actionable_plan")
        if not final_plan: return
            
        if feedback["feedback"] == -1:
            self.consecutive_losses += 1 # 실패 시, 연속 손실 카운터 증가
            analysis_context = str(cur_reflection)
            failure_analysis_result = analyze_failed_trade(self.endpoint_func, final_plan, analysis_context)
            if reason := failure_analysis_result.get("failure_reason"):
                self.logger.warning(f"실패 원인 분석 완료: {reason}")
                self.failure_memory.append(f"- On {cur_date}, a '{final_plan.get('investment_decision')}' decision failed. Reason: {reason}")
        
        if doc_ids := final_plan.get("supporting_document_ids"):
            if not doc_ids: return
            self.logger.info(f"피드백({feedback['feedback']})에 따라 문서 ID {doc_ids}의 중요도를 업데이트합니다.")
            self.brain.update_access_count_with_feed_back(self.trading_symbol, doc_ids, feedback["feedback"])


    def step(self, market_info: market_info_type) -> None:
        cur_date, cur_price, cur_news, cur_filing_k, cur_filing_q, is_done = market_info
        
        if is_done:
            self.logger.info("시뮬레이션의 마지막 스텝입니다. 종료합니다.")
            return

        self.logger.info(f"=== 스텝 {self.counter} 시작 (날짜: {cur_date}) ===")
        
        self._handling_news(cur_date, cur_news)
        self._handling_filings(cur_date, cur_filing_q, cur_filing_k)
        self.portfolio.update_market_info(new_market_price=cur_price, cur_date=cur_date)
        self._reflect(cur_date=cur_date)
        
        cur_action = self._process_action(cur_date=cur_date)
        self.portfolio.record_action(action=cur_action)
        
        self._update_access_counter()
        self.brain.step()
        
        self.counter += 1
        self.logger.info(f"=== 스텝 {self.counter-1} 완료 ===\n")

    def save_checkpoint(self, path: str, force: bool = False) -> None:
        if os.path.exists(path) and force: shutil.rmtree(path)
        os.makedirs(os.path.join(path, "brain"), exist_ok=True)
        
        state_to_save = self.__dict__.copy()
        del state_to_save['logger'], state_to_save['endpoint_func'], state_to_save['truncator'], state_to_save['brain']
        
        with open(os.path.join(path, "state_dict.pkl"), "wb") as f:
            pickle.dump(state_to_save, f)
        self.brain.save_checkpoint(path=os.path.join(path, "brain"), force=force)

    @classmethod
    def load_checkpoint(cls, path: str) -> "LLMAgent":
        with open(os.path.join(path, "state_dict.pkl"), "rb") as f:
            state_dict = pickle.load(f)
            
        brain = BrainDB.load_checkpoint(path=os.path.join(path, "brain"))
        
        portfolio_config = state_dict["portfolio"].__dict__

        agent = cls(
            agent_name=state_dict["agent_name"], trading_symbol=state_dict["trading_symbol"],
            brain_db=brain, chat_config=state_dict["chat_config_save"],
            portfolio_config=portfolio_config, top_k=state_dict.get("top_k", 7),
            look_back_window_size=state_dict.get("look_back_window_size", 7),
            truncation_max_length=state_dict.get("truncator").max_length if "truncator" in state_dict else 8192
        )
        
        for key, value in state_dict.items():
            if key not in ['brain', 'portfolio_config', 'chat_config_save', 'truncator']:
                setattr(agent, key, value)
        
        agent.failure_memory = state_dict.get("failure_memory", [])
        agent.consecutive_losses = state_dict.get("consecutive_losses", 0)
                
        return agent