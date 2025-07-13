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
# [추가됨] 표준 ChatOpenAI 라이브러리를 직접 import 합니다.
from langchain_openai import ChatOpenAI
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
        self.top_k = int(top_k)
        self.agent_name = agent_name
        self.trading_symbol = trading_symbol
        self.look_back_window_size = int(look_back_window_size)
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            log_path = os.path.join("data", "04_model_output_log", f"{self.trading_symbol}_run.log")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            handler = logging.FileHandler(log_path, mode="a")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.brain = brain_db
        self.portfolio = Portfolio(
            symbol=self.trading_symbol,
            initial_capital=float(portfolio_config.get("initial_capital", 1_000_000.0)),
            commission_rate=float(portfolio_config.get("commission_rate", 0.001)),
            lookback_window_size=self.look_back_window_size
        )
        
        self.chat_config_save = chat_config.copy()
        self.base_system_message = chat_config.pop("system_message")
        self.endpoint_func = None
        self.reflection_result_series_dict: Dict[date, Any] = {}
        self.truncator = TextTruncator(max_length=int(truncation_max_length))
        self.failure_memory: List[str] = []
        self.consecutive_losses = 0

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LLMAgent":
        general_config = config["general"]
        chat_config = config["chat"]
        portfolio_config = config["portfolio"]

        return cls(
            agent_name=general_config["agent_name"],
            trading_symbol=general_config["trading_symbol"],
            brain_db=BrainDB.from_config(config=config),
            top_k=int(general_config.get("top_k", 7)),
            chat_config=chat_config,
            portfolio_config=portfolio_config,
            look_back_window_size=int(general_config.get("look_back_window_size", 7)),
            truncation_max_length=int(general_config.get("truncation_max_length", 8192))
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
        self.logger.info(f"'{self.trading_symbol}' on investment committee analysis (Date: {cur_date})")
        
        market_regime = self.portfolio.get_market_regime()
        regime_warning = ""
        if market_regime == "Bear":
            regime_warning = "**CRITICAL ALERT: The market is in a confirmed Bearish trend. Prioritize capital preservation.**\n"
        elif market_regime == "Bull":
            regime_warning = "**Market Insight: The market is in a confirmed Bullish trend. Your primary focus is profit maximization.**\n"
        
        recent_failures = "\n".join(self.failure_memory[-3:])
        failure_context = f"Before you begin, review and learn from these recent failed trades:\n{recent_failures}\n" if recent_failures else ""

        recent_events_texts, _ = self.brain.query_short(
            query_text="corporate actions, shareholder changes, regulatory issues, new product launch, litigation",
            top_k=2,
            symbol=self.trading_symbol
        )
        recent_events_summary = " ".join(recent_events_texts) if recent_events_texts else "general investment decision"
        
        # --- [수정됨] RAG 성능 향상 로직 (표준 ChatOpenAI 사용) ---
        
        # 1. Step-back 및 준법감시용 표준 LLM 인스턴스 생성
        temp_llm = ChatOpenAI(
            model=self.chat_config_save.get("model", "gpt-4o-mini"),
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            temperature=0, # 일관성 있는 답변을 위해 0으로 설정
        )
        
        # 2. Step-back 질문 생성
        step_back_prompt = f"""
        사용자의 구체적인 상황: '{recent_events_summary[:200]}...'을 고려한 {self.trading_symbol} 주식 투자 결정을 앞두고 있다.
        이 상황의 법적 문제를 해결하기 위해 알아야 할 근본적이고 일반적인 금융소비자보호법 원칙에 대한 질문을 1개 생성해줘.
        """
        step_back_response = temp_llm.invoke(step_back_prompt)
        step_back_question = step_back_response.content
        self.logger.info(f"Generated Step-back Question: {step_back_question}")
        
        # 3. Step-back 질문으로 '핵심 원칙' 검색
        step_back_context_texts = self.brain.query_legal(step_back_question, top_k=2)
        step_back_context = "\n\n---\n\n".join(step_back_context_texts) if step_back_context_texts else "Not Found"

        # 4. 기존 질문으로 '구체적 조항' 검색
        original_question = (
            f"Regarding an investment decision for {self.trading_symbol} stock, "
            f"what are the key legal issues under the Financial Consumer Protection Act concerning recent events like '{recent_events_summary[:200]}...'? "
            "Focus on disclosure obligations, unfair trade practices, and investor protection."
        )
        normal_context_texts = self.brain.query_legal(original_question, top_k=2)
        normal_context = "\n\n---\n\n".join(normal_context_texts) if normal_context_texts else "Not Found"

        # 5. 두 검색 결과를 합쳐서 최종 프롬프트 구성
        legal_context_prompt_section = f"""
**MANDATORY LEGAL COMPLIANCE CHECK:** Review the following legal provisions. Ensure your analysis is compliant.

[Fundamental Principles]
{step_back_context}

[Specific Clauses]
{normal_context}
---
"""
        self.logger.info("Constructed legal context section.")
        
        # --- [수정 완료] ---
        
        initial_context = regime_warning + failure_context + legal_context_prompt_section
        
        chat_config_copy = self.chat_config_save.copy()
        self.endpoint_func = ChatOpenAICompatible(
            end_point=chat_config_copy.pop("end_point"),
            model=chat_config_copy.pop("model"),
            system_message=self.base_system_message,
            other_parameters=chat_config_copy,
        ).guardrail_endpoint()
        
        retrieved_docs = {}
        persona_queries = {
            "value": "financial health, debt, cash flow, intrinsic value, margin of safety",
            "growth": "new products, market share, innovation, user growth, future catalyst",
            "technical": "stock chart, moving average, trading volume, market sentiment, short-term news"
        }
        for query in persona_queries.values():
            for query_func in [self.brain.query_short, self.brain.query_mid, self.brain.query_long]:
                texts, ids = query_func(query_text=query, top_k=self.top_k, symbol=self.trading_symbol)
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
        
        # --- [수정됨] 준법 감시인(Compliance Officer) 검토 로직 (표준 LLM 사용) ---
        initial_plan = reflection_result.get("final_actionable_plan")
        if initial_plan:
            self.logger.info(f"Initial plan generated: {initial_plan.get('investment_decision')}. Starting compliance check...")
            
            compliance_check_prompt = f"""
            [Investment Plan Draft]
            {initial_plan}

            [Applicable Legal Provisions]
            {legal_context_prompt_section}

            You are a meticulous Compliance Officer. Review the Investment Plan Draft based on the provided legal provisions.
            Point out any potential violations or risks regarding the Financial Consumer Protection Act.
            If there are issues, specify which clause is at risk and suggest how to amend the plan.
            If the plan is fully compliant, respond ONLY with: 'Compliance confirmed.'
            """
            compliance_feedback_response = temp_llm.invoke(compliance_check_prompt)
            compliance_feedback = compliance_feedback_response.content
            self.logger.info(f"Compliance Check Feedback: {compliance_feedback}")
            
            if "Compliance confirmed." not in compliance_feedback:
                self.logger.warning("Compliance issues found. Refining the investment plan...")
                refinement_prompt = f"""
                [Original Investment Plan]
                {initial_plan}

                [Compliance Officer's Feedback]
                {compliance_feedback}

                You are the Chief Investment Officer. Revise the original plan to fully address the compliance officer's feedback.
                Output the revised plan in the same JSON format as the original.
                """
                
                final_plan_response = temp_llm.invoke(refinement_prompt)
                final_plan_str = final_plan_response.content
                try:
                    final_plan = eval(final_plan_str) 
                    reflection_result['final_actionable_plan'] = final_plan
                    self.logger.info(f"Plan refined and updated: {final_plan.get('investment_decision')}")
                except Exception as e:
                    self.logger.error(f"Failed to parse refined plan. Keeping original. Error: {e}")
            else:
                self.logger.info("Compliance check passed. No changes to the plan.")
        
        # --- [수정 완료] ---

        self.reflection_result_series_dict[cur_date] = reflection_result
        self.logger.info(f"Analysis complete. Final investment plan: {reflection_result.get('final_actionable_plan', 'Failed to generate')}")

    def _process_action(self, cur_date: date) -> Dict[str, Union[str, float]]:
        reflection_result = self.reflection_result_series_dict.get(cur_date, {})
        final_plan = reflection_result.get("final_actionable_plan")
        if not final_plan or not isinstance(final_plan, dict) or "error" in final_plan:
            self.logger.warning(f"{cur_date}: Analysis failed or error occurred. Deciding 'hold'.")
            return {"investment_decision": "hold", "position_sizing": 0.0}
        
        return {
            "investment_decision": final_plan.get("investment_decision", "hold"),
            "position_sizing": float(final_plan.get("position_sizing", 0.0))
        }

    def _update_access_counter(self):
        feedback = self.portfolio.get_feedback_response()
        if not feedback: return

        if feedback["feedback"] == 1:
            self.consecutive_losses = 0
        
        cur_date = feedback["date"]
        cur_reflection = self.reflection_result_series_dict.get(cur_date)
        if not cur_reflection: return

        final_plan = cur_reflection.get("final_actionable_plan")
        if not final_plan or not isinstance(final_plan, dict): return
            
        if feedback["feedback"] == -1:
            self.consecutive_losses += 1
            analysis_context = str(cur_reflection)
            # [수정됨] 준법 감시인 LLM을 재사용하여 실패 분석
            failure_analysis_result = analyze_failed_trade(self.endpoint_func, final_plan, analysis_context)
            if reason := failure_analysis_result.get("failure_reason"):
                self.logger.warning(f"Failure analysis complete: {reason}")
                self.failure_memory.append(f"- On {cur_date}, a '{final_plan.get('investment_decision')}' decision failed. Reason: {reason}")
        
        if doc_ids := final_plan.get("supporting_document_ids"):
            if not doc_ids: return
            self.logger.info(f"Updating importance of doc IDs {doc_ids} based on feedback ({feedback['feedback']}).")
            self.brain.update_access_count_with_feed_back(self.trading_symbol, doc_ids, feedback["feedback"])

    def step(self, market_info: market_info_type) -> None:
        cur_date, cur_price, cur_news, cur_filing_k, cur_filing_q, is_done = market_info
        
        if is_done:
            self.logger.info("Final step of the simulation. Terminating.")
            return

        self.logger.info(f"=== Step {self.counter} Start (Date: {cur_date}) ===")
        
        self._handling_news(cur_date, cur_news)
        self._handling_filings(cur_date, cur_filing_q, cur_filing_k)
        self.portfolio.update_market_info(new_market_price=cur_price, cur_date=cur_date)
        self._reflect(cur_date=cur_date)
        
        cur_action = self._process_action(cur_date=cur_date)
        self.portfolio.record_action(action=cur_action)
        
        self._update_access_counter()
        self.brain.step()
        
        self.counter += 1
        self.logger.info(f"=== Step {self.counter-1} End ===\n")

    def save_checkpoint(self, path: str, force: bool = False) -> None:
        if os.path.exists(path) and force:
            shutil.rmtree(path)
        os.makedirs(os.path.join(path, "brain"), exist_ok=True)
        
        state_to_save = self.__dict__.copy()
        del state_to_save['logger']
        del state_to_save['endpoint_func']
        del state_to_save['truncator']
        del state_to_save['brain']
        
        with open(os.path.join(path, "agent_state.pkl"), "wb") as f:
            pickle.dump(state_to_save, f)
        self.brain.save_checkpoint(path=os.path.join(path, "brain"), force=force)
        self.logger.info(f"Agent checkpoint saved to {path}")

    @classmethod
    def load_checkpoint(cls, path: str, config: Dict[str, Any]) -> "LLMAgent":
        with open(os.path.join(path, "agent_state.pkl"), "rb") as f:
            state_dict = pickle.load(f)
            
        brain = BrainDB.load_checkpoint(path=os.path.join(path, "brain"), config=config)
        
        chat_config = state_dict["chat_config_save"]
        portfolio = state_dict["portfolio"]
        
        agent = cls(
            agent_name=state_dict["agent_name"],
            trading_symbol=state_dict["trading_symbol"],
            brain_db=brain,
            chat_config=chat_config,
            portfolio_config=portfolio.__dict__,
            top_k=state_dict.get("top_k", 7),
            look_back_window_size=state_dict.get("look_back_window_size", 7),
            truncation_max_length=state_dict.get('truncator').max_length if 'truncator' in state_dict else 8192
        )
        
        agent.counter = state_dict.get("counter", 1)
        agent.portfolio = portfolio
        agent.reflection_result_series_dict = state_dict.get("reflection_result_series_dict", {})
        agent.failure_memory = state_dict.get("failure_memory", [])
        agent.consecutive_losses = state_dict.get("consecutive_losses", 0)
        
        print(f"Agent checkpoint loaded from {path}")
        return agent