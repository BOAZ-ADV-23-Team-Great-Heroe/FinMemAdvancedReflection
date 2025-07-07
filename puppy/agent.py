# import os
# import shutil
# import pickle
# import logging
# from datetime import date
# from .run_type import RunMode
# from .memorydb import BrainDB
# from .portfolio import Portfolio
# from abc import ABC, abstractmethod
# from .chat import ChatOpenAICompatible
# from .environment import market_info_type
# from typing import Dict, Union, Any, List, Optional
# from .reflection import trading_reflection
# from transformers import AutoTokenizer


# class TextTruncator:
#     """
#     텍스트를 토큰 단위로 자르거나, 리스트 내 여러 텍스트의 토큰 합계를 제한하는 유틸리티 클래스
#     """
#     def __init__(self, tokenization_model_name):
#         self.tokenization_model_name = tokenization_model_name
#         self.token = os.environ.get("HF_TOKEN", None)
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             self.tokenization_model_name, auth_token=self.token
#         )

#     def _tokenize_cnt_texts(self, input_text):
#         # 텍스트를 토크나이즈하고 토큰 개수 반환
#         encoded_input = self.tokenizer(input_text)
#         num_tokens = len(encoded_input["input_ids"])
#         return encoded_input, num_tokens

#     def process_list_of_texts(self, list_of_texts, max_total_tokens=320):
#         # 여러 텍스트의 토큰 합이 max_total_tokens를 넘지 않도록 자름
#         if "gpt" in self.tokenization_model_name:
#             return list_of_texts

#         truncated_list = []
#         total_tokens = 0
#         for text in list_of_texts:
#             encoded_input, num_tokens = self._tokenize_cnt_texts(text)

#             if total_tokens + num_tokens <= max_total_tokens:
#                 truncated_list.append(text)
#                 total_tokens += num_tokens
#             else:
#                 # 남은 토큰만큼만 자름
#                 remaining_tokens = max_total_tokens - total_tokens
#                 if remaining_tokens > 0:
#                     truncated_input_ids = encoded_input["input_ids"][:remaining_tokens]
#                     truncated_text = self.tokenizer.decode(
#                         truncated_input_ids, skip_special_tokens=True
#                     )
#                     truncated_list.append(truncated_text)
#                     total_tokens += len(truncated_input_ids)
#                 break

#         return truncated_list, total_tokens

#     # 단일 텍스트 토큰 자르기
#     def truncate_text(self, input_text, max_tokens):
#         encoded_input, num_tokens = self.tokenize_cnt_texts(input_text)

#         if len(encoded_input["input_ids"]) <= max_tokens:
#             return input_text, len(encoded_input["input_ids"])
#         encoded_input["input_ids"] = encoded_input["input_ids"][:max_tokens]
#         encoded_input["attention_mask"] = encoded_input["attention_mask"][:max_tokens]
#         output_text = self.tokenizer.decode(encoded_input["input_ids"])
#         num_tokens = max_tokens
#         return output_text, num_tokens


# class Agent(ABC):
#     """
#     에이전트 추상 클래스 (상속용)
#     """
#     @abstractmethod
#     def from_config(self, config: Dict[str, Any]) -> "Agent":
#         pass

#     @abstractmethod
#     def step(self) -> None:
#         pass


# # LLM 기반 에이전트
# class LLMAgent(Agent):
#     def __init__(
#         self,
#         agent_name: str,
#         trading_symbol: str,
#         character_string: str,
#         brain_db: BrainDB,
#         chat_config: Dict[str, Any],
#         top_k: int = 1,
#         look_back_window_size: int = 7,
#     ):
#         """
#     - 에이전트 이름, 종목, 캐릭터 문자열, 브레인 데이터베이스, 채팅 모델 설정 등을 통해객체를 생성.
#     - 포트폴리오 생성.
#     - 모델별 토큰 제한 값 저장.
#     - ChatOpenAICompatible 객체 생성하여 guardrail 엔드포인트 저장. """
#         # 기본 정보
#         self.counter = 1
#         self.top_k = top_k
#         self.agent_name = agent_name
#         self.trading_symbol = trading_symbol
#         self.character_string = character_string
#         self.look_back_window_size = look_back_window_size
#         # 로깅 설정
#         self.logger = logging.getLogger(__name__)
#         self.logger.setLevel(logging.INFO)
#         logging_formatter = logging.Formatter(
#             "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#             datefmt="%Y-%m-%d %H:%M:%S",
#         )
#         file_handler = logging.FileHandler(
#             os.path.join(
#                 "data",
#                 "04_model_output_log",
#                 f"{self.trading_symbol}_run.log",
#             ),
#             mode="a",
#         )
#         file_handler.setFormatter(logging_formatter)
#         self.logger.addHandler(file_handler)
#         # 브레인 DB (메모리 관리)
#         self.brain = brain_db
#         # 포트폴리오 관리
#         self.portfolio = Portfolio(
#             symbol=self.trading_symbol, lookback_window_size=self.look_back_window_size
#         )
#         self.chat_config_save = chat_config.copy()
#         chat_config = chat_config.copy()
#         end_point = chat_config["end_point"]
#         model = chat_config["model"]
#         system_message = chat_config["system_message"] # truncator
#         self.model_name = chat_config["model"]
#         self.max_token_short = chat_config.get("max_token_short", None)
#         self.max_token_mid = chat_config.get("max_token_mid", None)
#         self.max_token_long = chat_config.get("max_token_long", None)
#         self.max_token_reflection = chat_config.get("max_token_reflection", None)
#         del chat_config["end_point"]
#         del chat_config["model"]
#         del chat_config["system_message"]
#         if self.max_token_short:
#             self.truncator = TextTruncator(
#                 tokenization_model_name=chat_config["tokenization_model_name"]
#             )
#         # LLM API 엔드포인트 래퍼
#         self.guardrail_endpoint = ChatOpenAICompatible(
#             end_point=end_point,
#             model=model,
#             system_message=system_message,
#             other_parameters=chat_config,
#         ).guardrail_endpoint()
#         # 리플렉션 결과, 접근 카운터 등 기록용
#         self.reflection_result_series_dict = {}
#         self.access_counter = {}

#     @classmethod
#     def from_config(cls, config: Dict[str, Any]) -> "LLMAgent":
#         """
#         config 딕셔너리로부터 LLM 에이전트 인스턴스 생성
#         ./config/tsla_gpt_config.toml 참고
#         """
#         return cls(
#             agent_name=config["general"]["agent_name"],
#             trading_symbol=config["general"]["trading_symbol"],
#             character_string=config["general"]["character_string"],
#             brain_db=BrainDB.from_config(config=config),
#             top_k=config["general"].get("top_k", 5),
#             chat_config=config["chat"],
#             look_back_window_size=config["general"]["look_back_window_size"],
#         )

#     def _handling_news(self, cur_date: date, news: Union[List[str], Dict[str, List[str]]]) -> None:
#         """
#         뉴스 데이터를 short memory에 저장
#         - 기존 코드는 해당 날짜의 모든 뉴스를 리스트로 저장
#         - 아래는 기사 단위 청크로 저장
#         """
#         if news is not None:
#             if isinstance(news, list):
#                 for news_item in news:
#                     if news_item and news_item.strip():
#                         self.logger.info(f"Adding news memory: {news_item[:100]}...")
#                         self.brain.add_memory_short(
#                             symbol=self.trading_symbol, date=cur_date, text=news_item
#                         )
#             else:
#                 self.logger.warning(f"Unexpected news type for {cur_date}: {type(news)}")

#     def _handling_filings(self, cur_date: date, filing_q: Optional[Dict[str, str]], filing_k: Optional[Dict[str, str]]) -> None:
#         """
#         공시 데이터를 mid/long memory에 저장
#         """
#         if filing_q and isinstance(filing_q, dict) and filing_q:  # 빈 딕셔너리가 아닌 경우에만 처리
#             for ticker, content in filing_q.items():
#                 if content and content.strip():
#                     self.logger.info(f"Adding filing Q memory: {content[:100]}...")
#                     self.brain.add_memory_mid(
#                         symbol=self.trading_symbol, date=cur_date, text=content
#                     )
#         elif filing_q is not None and not isinstance(filing_q, dict):
#             self.logger.warning(f"Unexpected filing_q type for {cur_date}: {type(filing_q)}")
        
#         if filing_k and isinstance(filing_k, dict) and filing_k:  # 빈 딕셔너리가 아닌 경우에만 처리
#             for ticker, content in filing_k.items():
#                 if content and content.strip():
#                     self.logger.info(f"Adding filing K memory: {content[:100]}...")
#                     self.brain.add_memory_long(
#                         symbol=self.trading_symbol,
#                         date=cur_date,
#                         text=content,
#                     )
#         elif filing_k is not None and not isinstance(filing_k, dict):
#             self.logger.warning(f"Unexpected filing_k type for {cur_date}: {type(filing_k)}")

#     def __query_info_for_reflection(self, run_mode: RunMode):
#         """
#         리플렉션(자기성찰)용으로 메모리에서 top-k를 쿼리하고, 토큰 제한이 있으면 자름
#         - Train 모드: short/mid/long/reflection memory 각각 top-k 쿼리 및 토큰 제한 적용 후, 각 쿼리 결과와 id를 튜플로 반환
#         - Test 모드: 위 쿼리 결과에 더해 포트폴리오 모멘텀(momentum) 정보도 함께 반환
#         - 모델이 tgi 계열이면 토큰 제한 적용, gpt 계열이면 전체 반환
#         - 반환값 구조가 Train/Test에서 다름에 주의
#         """
#         self.logger.info(f"Symbol: {self.trading_symbol}\n")
#         # 1. Short memory 쿼리
#         cur_short_queried, cur_short_memory_id = self.brain.query_short(
#             query_text=self.character_string,
#             top_k=self.top_k,
#             symbol=self.trading_symbol,
#         )
#         # 2. 토큰 제한 적용 (tgi 계열 모델만)
#         if self.model_name.startswith("tgi"):
#             cur_short_queried_truc, cur_short_num_tokens = (
#                 self.truncator.process_list_of_texts(
#                     cur_short_queried, max_total_tokens=self.max_token_short
#                 )
#             )
#             cur_short_memory_id_truc = [
#                 cur_short_memory_id[k] for k in range(len(cur_short_queried_truc))
#             ]
#             for cur_id, cur_memory in zip(
#                 cur_short_memory_id_truc, cur_short_queried_truc
#             ):
#                 self.logger.info(f"Top-k Short: {cur_id}: {cur_memory}\n")
#             self.logger.info(f"Total tokens of Short Memory: {cur_short_num_tokens}\n")
#         else:
#             for cur_id, cur_memory in zip(cur_short_memory_id, cur_short_queried):
#                 self.logger.info(f"Top-k Short: {cur_id}: {cur_memory}\n")

#         # 3. Mid memory 쿼리 및 토큰 제한
#         cur_mid_queried, cur_mid_memory_id = self.brain.query_mid(
#             query_text=self.character_string,
#             top_k=self.top_k,
#             symbol=self.trading_symbol,
#         )
#         if self.model_name.startswith("tgi"):
#             cur_mid_queried_truc, cur_mid_num_tokens = (
#                 self.truncator.process_list_of_texts(
#                     cur_mid_queried, max_total_tokens=self.max_token_mid
#                 )
#             )
#             cur_mid_memory_id_truc = [
#                 cur_mid_memory_id[k] for k in range(len(cur_mid_queried_truc))
#             ]
#             for cur_id, cur_memory in zip(cur_mid_memory_id_truc, cur_mid_queried_truc):
#                 self.logger.info(f"Top-k Mid: {cur_id}: {cur_memory}\n")
#             self.logger.info(f"Total tokens of Middle Memory: {cur_mid_num_tokens}\n")
#         else:
#             for cur_id, cur_memory in zip(cur_mid_memory_id, cur_mid_queried):
#                 self.logger.info(f"Top-k Mid: {cur_id}: {cur_memory}\n")

#         # 4. Long memory 쿼리 및 토큰 제한
#         cur_long_queried, cur_long_memory_id = self.brain.query_long(
#             query_text=self.character_string,
#             top_k=self.top_k,
#             symbol=self.trading_symbol,
#         )
#         if self.model_name.startswith("tgi"):
#             cur_long_queried_truc, cur_long_num_tokens = (
#                 self.truncator.process_list_of_texts(
#                     cur_long_queried, max_total_tokens=self.max_token_long
#                 )
#             )
#             cur_long_memory_id_truc = [
#                 cur_long_memory_id[k] for k in range(len(cur_long_queried_truc))
#             ]
#             for cur_id, cur_memory in zip(
#                 cur_long_memory_id_truc, cur_long_queried_truc
#             ):
#                 self.logger.info(f"Top-k Long: {cur_id}: {cur_memory}\n")
#             self.logger.info(f"Total tokens of Long Memory: {cur_long_num_tokens}\n")
#         else:
#             for cur_id, cur_memory in zip(cur_long_memory_id, cur_long_queried):
#                 self.logger.info(f"Top-k Long: {cur_id}: {cur_memory}\n")

#         # 5. Reflection memory 쿼리 및 토큰 제한
#         (
#             cur_reflection_queried,
#             cur_reflection_memory_id,
#         ) = self.brain.query_reflection(
#             query_text=self.character_string,
#             top_k=self.top_k,
#             symbol=self.trading_symbol,
#         )
#         if self.model_name.startswith("tgi"):
#             cur_reflection_queried_truc, cur_reflection_num_tokens = (
#                 self.truncator.process_list_of_texts(
#                     cur_reflection_queried, max_total_tokens=self.max_token_reflection
#                 )
#             )
#             cur_reflection_memory_id_truc = [
#                 cur_reflection_memory_id[k]
#                 for k in range(len(cur_reflection_queried_truc))
#             ]
#             for cur_id, cur_memory in zip(
#                 cur_reflection_memory_id_truc, cur_reflection_queried_truc
#             ):
#                 self.logger.info(f"Top-k Reflection: {cur_id}: {cur_memory}\n")
#             self.logger.info(
#                 f"Total tokens of Reflection Memory: {cur_reflection_num_tokens}\n"
#             )
#         else:
#             for cur_id, cur_memory in zip(
#                 cur_reflection_memory_id, cur_reflection_queried
#             ):
#                 self.logger.info(f"Top-k Reflection: {cur_id}: {cur_memory}\n")

#         # 6. 전체 토큰 수 로그 (tgi 계열)
#         if self.model_name.startswith("tgi"):
#             cur_all_num_tokens = (
#                 cur_short_num_tokens
#                 + cur_mid_num_tokens
#                 + cur_long_num_tokens
#                 + cur_reflection_num_tokens
#             )
#             self.logger.info(f"Total tokens of **ALL** Memory: {cur_all_num_tokens}\n")

#         # 7. Test 모드에서는 포트폴리오 모멘텀 정보도 반환
#         if run_mode == RunMode.Test:
#             cur_moment_ret = self.portfolio.get_moment(moment_window=3)
#             cur_moment = (
#                 cur_moment_ret["moment"] if cur_moment_ret is not None else None
#             )

#         # 8. 반환값 구조: Train/Test에 따라 다름
#         if run_mode == RunMode.Train:
#             if self.model_name.startswith("tgi"):
#                 # tgi 모델: 토큰 제한 적용된 쿼리 결과 반환
#                 return (
#                     cur_short_queried_truc,
#                     cur_short_memory_id_truc,
#                     cur_mid_queried_truc,
#                     cur_mid_memory_id_truc,
#                     cur_long_queried_truc,
#                     cur_long_memory_id_truc,
#                     cur_reflection_queried_truc,
#                     cur_reflection_memory_id_truc,
#                 )
#             else:
#                 # gpt 등: 전체 쿼리 결과 반환
#                 return (
#                     cur_short_queried,
#                     cur_short_memory_id,
#                     cur_mid_queried,
#                     cur_mid_memory_id,
#                     cur_long_queried,
#                     cur_long_memory_id,
#                     cur_reflection_queried,
#                     cur_reflection_memory_id,
#                 )
#         elif run_mode == RunMode.Test:
#             if self.model_name.startswith("tgi"):
#                 # tgi 모델: momentum 정보까지 포함해 반환
#                 return (
#                     cur_short_queried_truc,
#                     cur_short_memory_id_truc,
#                     cur_mid_queried_truc,
#                     cur_mid_memory_id_truc,
#                     cur_long_queried_truc,
#                     cur_long_memory_id_truc,
#                     cur_reflection_queried_truc,
#                     cur_reflection_memory_id_truc,
#                     cur_moment,  # type: ignore
#                 )
#             else:
#                 # gpt 등: momentum 정보까지 포함해 반환
#                 return (
#                     cur_short_queried,
#                     cur_short_memory_id,
#                     cur_mid_queried,
#                     cur_mid_memory_id,
#                     cur_long_queried,
#                     cur_long_memory_id,
#                     cur_reflection_queried,
#                     cur_reflection_memory_id,
#                     cur_moment,  # type: ignore
#                 )

#     def __reflection_on_record(
#         self,
#         cur_date: date,
#         run_mode: RunMode,
#         cur_record: Union[float, None] = None,
#     ) -> Dict[str, Any]:
#         """
#         리플렉션(자기성찰) 실행 및 결과 저장
#         - Train 모드: 미래 수익률(cur_record)을 활용해 자기성찰 프롬프트 생성 및 결과 저장
#         - Test 모드: momentum 등 추가 정보와 함께 자기성찰 프롬프트 생성 및 결과 저장
#         - 각 메모리(Short/Mid/Long/Reflection)에서 top-k 쿼리 후, 토큰 제한이 있으면 자름
#         - trading_reflection 함수로 LLM 호출 및 결과 반환
#         - 결과가 있으면 reflection memory에 저장
#         """
#         if (run_mode == RunMode.Train) and (not cur_record):
#             # Train 모드에서 수익률 정보가 없으면 리플렉션 실행하지 않음
#             self.logger.info("No record\n")
#             return {}
#         # reflection
#         if run_mode == RunMode.Train:
#             # 1. 각 메모리(Short/Mid/Long/Reflection)에서 top-k 쿼리 및 토큰 제한 적용
#             (
#                 cur_short_queried,
#                 cur_short_memory_id,
#                 cur_mid_queried,
#                 cur_mid_memory_id,
#                 cur_long_queried,
#                 cur_long_memory_id,
#                 cur_reflection_queried,
#                 cur_reflection_memory_id,
#             ) = self.__query_info_for_reflection(  # type: ignore
#                 run_mode=run_mode
#             )
#             # 2. trading_reflection 함수로 LLM reflection 실행 (미래 수익률 포함)
#             reflection_result = trading_reflection(
#                 cur_date=cur_date,
#                 symbol=self.trading_symbol,
#                 run_mode=run_mode,
#                 endpoint_func=self.guardrail_endpoint,
#                 short_memory=cur_short_queried,
#                 short_memory_id=cur_short_memory_id,
#                 mid_memory=cur_mid_queried,
#                 mid_memory_id=cur_mid_memory_id,
#                 long_memory=cur_long_queried,
#                 long_memory_id=cur_long_memory_id,
#                 reflection_memory=cur_reflection_queried,
#                 reflection_memory_id=cur_reflection_memory_id,
#                 future_record=cur_record,  # type: ignore
#                 logger=self.logger,
#             )
#         elif run_mode == RunMode.Test:
#             # 1. 각 메모리(Short/Mid/Long/Reflection)에서 top-k 쿼리 및 토큰 제한 적용
#             (
#                 cur_short_queried,
#                 cur_short_memory_id,
#                 cur_mid_queried,
#                 cur_mid_memory_id,
#                 cur_long_queried,
#                 cur_long_memory_id,
#                 cur_reflection_queried,
#                 cur_reflection_memory_id,
#                 cur_moment,
#             ) = self.__query_info_for_reflection(  # type: ignore
#                 run_mode=run_mode
#             )
#             # 2. trading_reflection 함수로 LLM reflection 실행 (momentum 등 추가)
#             reflection_result = trading_reflection(
#                 cur_date=cur_date,
#                 symbol=self.trading_symbol,
#                 run_mode=run_mode,
#                 endpoint_func=self.guardrail_endpoint,
#                 short_memory=cur_short_queried,
#                 short_memory_id=cur_short_memory_id,
#                 mid_memory=cur_mid_queried,
#                 mid_memory_id=cur_mid_memory_id,
#                 long_memory=cur_long_queried,
#                 long_memory_id=cur_long_memory_id,
#                 reflection_memory=cur_reflection_queried,
#                 reflection_memory_id=cur_reflection_memory_id,
#                 momentum=cur_moment,
#                 logger=self.logger,
#             )

#         # 3. 결과가 정상적으로 생성되었으면 reflection memory에 저장
#         if (reflection_result is not {}) and ("summary_reason" in reflection_result):
#             self.brain.add_memory_reflection(
#                 symbol=self.trading_symbol,
#                 date=cur_date,
#                 text=reflection_result["summary_reason"],
#             )
#         else:
#             # 결과가 없거나 수렴하지 않은 경우 로그만 남김
#             self.logger.info("No reflection result , not converged\n")
#         # 4. 결과 반환 (dict)
#         return reflection_result

#     def _reflect(
#         self,
#         cur_date: date,
#         run_mode: RunMode,
#         cur_record: Union[float, None] = None,
#     ) -> None:
#         """
#         리플렉션 결과를 기록하고 로그로 남김
#         """
#         if run_mode == RunMode.Train:
#             reflection_result_cur_date = self.__reflection_on_record(
#                 cur_date=cur_date,
#                 cur_record=cur_record,
#                 run_mode=run_mode,
#             )
#         else:
#             reflection_result_cur_date = self.__reflection_on_record(
#                 cur_date=cur_date, run_mode=run_mode
#             )
#         self.reflection_result_series_dict[cur_date] = reflection_result_cur_date
#         if run_mode == RunMode.Train:
#             self.logger.info(
#                 f"{self.trading_symbol}-Day {cur_date}\nreflection summary: {reflection_result_cur_date.get('summary_reason')}\n\n"
#             )
#         elif run_mode == RunMode.Test:
#             if len(reflection_result_cur_date) != 0:
#                 self.logger.info(
#                     f"!!trading decision: {reflection_result_cur_date['investment_decision']} !! {self.trading_symbol}-Day {cur_date}\ninvestment reason: {reflection_result_cur_date.get('summary_reason')}\n\n"
#                 )
#             else:
#                 self.logger.info("no decision")

#     def _construct_train_actions(self, cur_record: float) -> Dict[str, int]:
#         """
#         수익률에 따라 매수(수익률이 양수)/매도 방향 결정
#         """
#         cur_direction = 1 if cur_record > 0 else -1
#         return {"direction": cur_direction, "quantity": 1}

#     def _portfolio_step(self, cur_action: Dict[str, int]) -> None:
#         """
#         포트폴리오에 액션 기록 및 업데이트
#         """
#         self.portfolio.record_action(action=cur_action)  # type: ignore
#         self.portfolio.update_portfolio_series()

#     def __update_short_memory_access_counter(
#         self,
#         feedback: Dict[str, Union[int, date]],
#         cur_memory: Dict[str, Any],
#     ) -> None:
#         if "short_memory_index" in cur_memory:
#             self.__update_access_counter_sub(
#                 cur_memory=cur_memory,
#                 layer_index_name="short_memory_index",
#                 feedback=feedback,
#             )

#     def __update_mid_memory_access_counter(
#         self,
#         feedback: Dict[str, Union[int, date]],
#         cur_memory: Dict[str, Any],
#     ) -> None:
#         if "middle_memory_index" in cur_memory:
#             self.__update_access_counter_sub(
#                 cur_memory=cur_memory,
#                 layer_index_name="middle_memory_index",
#                 feedback=feedback,
#             )

#     def __update_long_memory_access_counter(
#         self,
#         feedback: Dict[str, Union[int, date]],
#         cur_memory: Dict[str, Any],
#     ) -> None:
#         if "long_memory_index" in cur_memory:
#             self.__update_access_counter_sub(
#                 cur_memory=cur_memory,
#                 layer_index_name="long_memory_index",
#                 feedback=feedback,
#             )

#     def __update_reflection_memory_access_counter(
#         self,
#         feedback: Dict[str, Union[int, date]],
#         cur_memory: Dict[str, Any],
#     ) -> None:
#         if "reflection_memory_index" in cur_memory:
#             self.__update_access_counter_sub(
#                 cur_memory=cur_memory,
#                 layer_index_name="reflection_memory_index",
#                 feedback=feedback,
#             )

#     def __update_access_counter_sub(self, cur_memory, layer_index_name, feedback):
#         if cur_memory[layer_index_name] is not None:
#             cur_ids = []
#             for i in cur_memory[layer_index_name]:
#                 cur_id = i["memory_index"]
#                 if cur_id not in cur_ids:
#                     cur_ids.append(cur_id)
#             self.brain.update_access_count_with_feed_back(
#                 symbol=self.trading_symbol,
#                 ids=cur_ids,
#                 feedback=feedback["feedback"],
#             )

#     @staticmethod
#     def __process_test_action(test_reflection_result: Dict[str, Any]) -> Dict[str, int]:
#         """
#         테스트 모드에서 리플렉션 결과에 따라 액션 결정
#         """
#         if (
#             test_reflection_result
#             and test_reflection_result["investment_decision"] == "buy"
#         ):
#             return {"direction": 1}
#         elif (
#             len(test_reflection_result) != 0
#             and test_reflection_result["investment_decision"] == "hold"
#             or not test_reflection_result
#         ):
#             return {"direction": 0}
#         else:
#             return {"direction": -1}

#     def _update_access_counter(self):
#         """
#         피드백에 따라 메모리 접근 카운터 업데이트
#         """
#         if not (feedback := self.portfolio.get_feedback_response()):
#             return
#         if feedback["feedback"] != 0:
#             # update short memory if it is not none
#             cur_date = feedback["date"]
#             cur_memory = self.reflection_result_series_dict[cur_date]
#             self.__update_short_memory_access_counter(
#                 feedback=feedback, cur_memory=cur_memory
#             )
#             self.__update_mid_memory_access_counter(
#                 feedback=feedback, cur_memory=cur_memory
#             )
#             self.__update_long_memory_access_counter(
#                 feedback=feedback, cur_memory=cur_memory
#             )
#             self.__update_reflection_memory_access_counter(
#                 feedback=feedback, cur_memory=cur_memory
#             )

#     def step(
#         self,
#         market_info: market_info_type,
#         run_mode: RunMode,
#     ) -> None:
#         """
#         시뮬레이션 한 스텝 실행: 가격/뉴스/공시 처리, 메모리 업데이트, 체크포인트 저장 등
#         """
#         # 로그 설정
#         log_dir = os.path.join("data", "04_model_output_log")
#         os.makedirs(log_dir, exist_ok=True)
        
#         log_file = os.path.join(log_dir, f"{self.trading_symbol}_run.log")
#         print(f"Logging to: {log_file}")  # 디버깅용 출력
        
#         # 기존 핸들러 제거
#         for handler in self.logger.handlers[:]:
#             self.logger.removeHandler(handler)
        
#         # 로그 레벨 설정
#         self.logger.setLevel(logging.INFO)
        
#         # 파일 핸들러 설정
#         file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
#         file_handler.setLevel(logging.INFO)
        
#         # 포맷터 설정
#         formatter = logging.Formatter(
#             "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#             datefmt="%Y-%m-%d %H:%M:%S"
#         )
#         file_handler.setFormatter(formatter)
        
#         # 핸들러 추가
#         self.logger.addHandler(file_handler)
        
#         # 콘솔 출력도 추가
#         console_handler = logging.StreamHandler()
#         console_handler.setLevel(logging.INFO)
#         console_handler.setFormatter(formatter)
#         self.logger.addHandler(console_handler)
        
#         # 테스트 로그
#         self.logger.info("=== Starting new step ===")

#         # 1. 데이터 언패킹
#         cur_date = market_info[0] # date
#         cur_price = market_info[1] # (int, float)
#         cur_filing_q = market_info[2]
#         cur_filing_k = market_info[3]
#         cur_news = market_info[4] # (list)
#         cur_record = market_info[5] if run_mode == RunMode.Train else None
            
#         self.logger.info(f"Processing date: {cur_date}")

#         # 2. 뉴스 데이터 처리
#         self._handling_news(cur_date, cur_news)

#         # # 3. filing 데이터 처리
#         self._handling_filings(cur_date, cur_filing_q, cur_filing_k)


#         # 4. 포트폴리오에 가격 업데이트
#         self.portfolio.update_market_info(
#             new_market_price_info=cur_price,
#             cur_date=cur_date,
#         )
#         # 4-1. 리플렉션(자기성찰) 실행: 현재 날짜, 모드, 수익률 정보를 바탕으로 LLM reflection을 수행하고,
#         #     - Train 모드에서는 미래 수익률을 활용해 자기성찰 프롬프트를 생성하여 LLM에 질의
#         #     - Test 모드에서는 momentum 등 추가 정보를 활용해 LLM에 질의
#         #     - 결과(투자 판단, 요약 등)는 reflection_result_series_dict에 저장되고, 로그로도 남김
#         #     - 이후 포트폴리오 액션 결정, 메모리 업데이트 등에 활용됨
#         self._reflect(
#             cur_date=cur_date,
#             run_mode=run_mode,
#             cur_record=cur_record,
#         )
#         # 5. 포트폴리오 액션 결정
#         #     - Train 모드: 미래 수익률(cur_record)이 양수면 매수, 음수면 매도 액션 생성
#         #     - Test 모드: 리플렉션 결과(투자 판단)에 따라 buy/hold/sell 액션 생성
#         if run_mode == RunMode.Train:
#             cur_action = self._construct_train_actions(
#                 cur_record=cur_record  # type: ignore
#             )
#         elif run_mode == RunMode.Test:
#             cur_action = self.__process_test_action(
#                 test_reflection_result=self.reflection_result_series_dict[cur_date]
#             )
        
#         # 6. 포트폴리오 업데이트
#         #     - 위에서 결정된 액션을 포트폴리오에 기록 및 반영
#         self._portfolio_step(cur_action=cur_action)  # type: ignore
       
#         # 7. 메모리 접근 카운터 업데이트
#         #     - 포트폴리오 피드백(수익률 등)에 따라 각 메모리의 access counter를 업데이트
#         self._update_access_counter()
        
#         # 8. 메모리 점프(브레인 내부 상태 업데이트)
#         self.logger.info("Running brain.step()...")
#         self.brain.step()
#         # 9. 체크포인트 저장
#         #     - 에이전트 및 브레인 상태를 파일로 저장 (중간 저장)
#         self.logger.info("Saving checkpoint...")
#         self.save_checkpoint(path="data/05_train_model_output/brain", force=True)

#         self.counter += 1
#         self.logger.info("=== Step completed ===\n")

#     def save_checkpoint(self, path: str, force: bool = False) -> None:
#         """
#         에이전트 상태 및 브레인 메모리 체크포인트 저장
#         """
#         if os.path.exists(path):
#             if not force:
#                 raise FileExistsError(f"Path {path} already exists")
#             shutil.rmtree(path)
#         os.makedirs(path, exist_ok=True)
#         os.makedirs(os.path.join(path, "brain"), exist_ok=True)
#         state_dict = {
#             "agent_name": self.agent_name,
#             "character_string": self.character_string,
#             "top_k": self.top_k,
#             "counter": self.counter,
#             "trading_symbol": self.trading_symbol,
#             "portfolio": self.portfolio,
#             "chat_config": self.chat_config_save,
#             "reflection_result_series_dict": self.reflection_result_series_dict,  #
#             "access_counter": self.access_counter,
#         }
#         with open(os.path.join(path, "state_dict.pkl"), "wb") as f:
#             pickle.dump(state_dict, f)
#         self.brain.save_checkpoint(path=os.path.join(path, "brain"), force=force)

#     @classmethod
#     def load_checkpoint(cls, path: str) -> "LLMAgent":
#         """
#         체크포인트에서 LLM 에이전트 객체 복원
#         """
#         # load state dict
#         with open(os.path.join(path, "state_dict.pkl"), "rb") as f:
#             state_dict = pickle.load(f)
#         # load brain
#         brain = BrainDB.load_checkpoint(path=os.path.join(path, "brain"))
#         class_obj = cls(
#             agent_name=state_dict["agent_name"],
#             trading_symbol=state_dict["trading_symbol"],
#             character_string=state_dict["character_string"],
#             brain_db=brain,
#             top_k=state_dict["top_k"],
#             chat_config=state_dict["chat_config"],
#         )
#         class_obj.portfolio = state_dict["portfolio"]
#         class_obj.reflection_result_series_dict = state_dict[
#             "reflection_result_series_dict"
#         ]
#         class_obj.access_counter = state_dict["access_counter"]
#         class_obj.counter = state_dict["counter"]
#         return class_obj


# agent.py (Multi-Persona Investment Committee Version)
# agent.py (Final Version compatible with Advanced Portfolio & Reflection)

import os
import shutil
import pickle
import logging
from datetime import date
from .memorydb import BrainDB
from .portfolio import Portfolio
from abc import ABC, abstractmethod
from .chat import ChatOpenAICompatible
from .environment import market_info_type
from typing import Dict, Union, Any, List, Tuple
from .reflection import run_investment_committee_analysis
from transformers import AutoTokenizer

class TextTruncator:
    def __init__(self, tokenizer_name="roberta-base", max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def truncate(self, text: str) -> str:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        return self.tokenizer.decode(tokens)

class Agent(ABC):
    @abstractmethod
    def from_config(self, config: Dict[str, Any]) -> "Agent":
        pass
    @abstractmethod
    def step(self) -> None:
        pass

class LLMAgent(Agent):
    def __init__(
        self,
        agent_name: str,
        trading_symbol: str,
        brain_db: BrainDB,
        chat_config: Dict[str, Any],
        portfolio_config: Dict[str, Any], 
        top_k: int = 7,
        look_back_window_size: int = 7,
    ):
        self.counter = 1
        self.top_k = top_k
        self.agent_name = agent_name
        self.trading_symbol = trading_symbol
        self.look_back_window_size = look_back_window_size
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        logging_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler = logging.FileHandler(
            os.path.join("data", "04_model_output_log", f"{self.trading_symbol}_run.log"),
            mode="a",
        )
        file_handler.setFormatter(logging_formatter)
        if not self.logger.handlers: self.logger.addHandler(file_handler)
        
        self.brain = brain_db
        self.portfolio = Portfolio(
            symbol=self.trading_symbol,
            initial_capital=portfolio_config.get("initial_capital", 1_000_000.0),
            commission_rate=portfolio_config.get("commission_rate", 0.001),
            lookback_window_size=self.look_back_window_size
        )
        
        self.chat_config_save = chat_config.copy()
        chat_config_copy = chat_config.copy()
        end_point = chat_config_copy.pop("end_point")
        model = chat_config_copy.pop("model")
        system_message = chat_config_copy.pop("system_message")
        self.endpoint_func = ChatOpenAICompatible(
            end_point=end_point, model=model, system_message=system_message,
            other_parameters=chat_config_copy,
        ).guardrail_endpoint()
        self.reflection_result_series_dict = {}
        self.truncator = TextTruncator()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LLMAgent":
        return cls(
            agent_name=config["general"]["agent_name"],
            trading_symbol=config["general"]["trading_symbol"],
            brain_db=BrainDB.from_config(config=config),
            top_k=config["general"].get("top_k", 7),
            chat_config=config["chat"],
            portfolio_config=config.get("portfolio", {}),
            look_back_window_size=config["general"]["look_back_window_size"],
            # character_string은 더 이상 사용하지 않으므로 제거
        )

    def _handling_news(self, cur_date: date, news: Union[List[str], Dict[str, List[str]]]) -> None:
        if news:
            if isinstance(news, list):
                for news_item in news:
                    if news_item and news_item.strip():
                        self.brain.add_memory_short(symbol=self.trading_symbol, date=cur_date, text=news_item)
            else:
                self.logger.warning(f"Unexpected news type for {cur_date}: {type(news)}")
    
    def _handling_filings(self, cur_date: date, filing_q, filing_k) -> None:
        """
        공시 데이터를 mid/long memory에 저장
        """
        if filing_q and isinstance(filing_q, dict) and filing_q:  # 빈 딕셔너리가 아닌 경우에만 처리
            for ticker, content in filing_q.items():
                if content and content.strip():
                    self.logger.info(f"Adding filing Q memory: {content[:100]}...")
                    self.brain.add_memory_mid(
                        symbol=self.trading_symbol, date=cur_date, text=content
                    )
        elif filing_q is not None and not isinstance(filing_q, dict):
            self.logger.warning(f"Unexpected filing_q type for {cur_date}: {type(filing_q)}")
        
        if filing_k and isinstance(filing_k, dict) and filing_k:  # 빈 딕셔너리가 아닌 경우에만 처리
            for ticker, content in filing_k.items():
                if content and content.strip():
                    self.logger.info(f"Adding filing K memory: {content[:100]}...")
                    self.brain.add_memory_long(
                        symbol=self.trading_symbol,
                        date=cur_date,
                        text=content,
                    )
        elif filing_k is not None and not isinstance(filing_k, dict):
            self.logger.warning(f"Unexpected filing_k type for {cur_date}: {type(filing_k)}")
    
    def _reflect(self, cur_date: date) -> None:
        self.logger.info(f"'{self.trading_symbol}'에 대한 투자 위원회 분석 시작 (날짜: {cur_date})")

        persona_queries = {
            "value": "financial health, debt, cash flow, intrinsic value, margin of safety",
            "growth": "new products, market share, innovation, user growth, future catalyst",
            "technical": "stock chart, moving average, trading volume, market sentiment, short-term news"
        }

        retrieved_docs = {}
        for query in persona_queries.values():
            short_texts, short_ids = self.brain.query_short(query, self.top_k, self.trading_symbol)
            mid_texts, mid_ids = self.brain.query_mid(query, self.top_k, self.trading_symbol)
            long_texts, long_ids = self.brain.query_long(query, self.top_k, self.trading_symbol)
            
            for id, text in zip(short_ids + mid_ids + long_ids, short_texts + mid_texts + long_texts):
                retrieved_docs[id] = self.truncator.truncate(text)

        all_docs_with_ids = list(retrieved_docs.items())
        
        reflection_result = run_investment_committee_analysis(
            endpoint_func=self.endpoint_func,
            short_memory_texts_with_ids=all_docs_with_ids, # 간소화를 위해 모든 정보를 전달
            mid_long_memory_texts_with_ids=all_docs_with_ids,
        )

        self.reflection_result_series_dict[cur_date] = reflection_result
        self.logger.info(f"분석 완료. 최종 투자 계획: {reflection_result.get('final_actionable_plan', '생성 실패')}")

    def _process_action(self, cur_date: date) -> Dict[str, Union[str, float]]:
        reflection_result = self.reflection_result_series_dict.get(cur_date, {})
        final_plan = reflection_result.get("final_actionable_plan")

        if not final_plan or "error" in reflection_result:
            self.logger.warning(f"{cur_date}: 분석 실패 또는 오류로 '보유(hold)' 결정")
            return {"investment_decision": "hold", "position_sizing": 0.0}
        
        return {
            "investment_decision": final_plan.get("investment_decision", "hold"),
            "position_sizing": final_plan.get("position_sizing", 0.0)
        }

    def _update_access_counter(self):
        feedback = self.portfolio.get_feedback_response()
        if not feedback or feedback["feedback"] == 0:
            return

        cur_date = feedback["date"]
        cur_reflection = self.reflection_result_series_dict.get(cur_date, {})
        final_plan = cur_reflection.get("final_actionable_plan")

        if final_plan and (doc_ids := final_plan.get("supporting_document_ids")):
            if not doc_ids: return
            self.logger.info(f"피드백({feedback['feedback']})에 따라 문서 ID {doc_ids}의 중요도를 업데이트합니다.")
            self.brain.update_access_count_with_feed_back(
                symbol=self.trading_symbol,
                ids=doc_ids,
                feedback=feedback["feedback"],
            )

    def step(self, market_info: market_info_type) -> None:
        cur_date, cur_price, cur_news, cur_filing_k, cur_filing_q, is_done = market_info
        self.logger.info(f"=== 스텝 {self.counter} 시작 (날짜: {cur_date}) ===")
        
        self._handling_news(cur_date, cur_news)
        self._handling_filings(cur_date, cur_filing_q, cur_filing_k);
        self.portfolio.update_market_info(new_market_price=cur_price, cur_date=cur_date)
        self._reflect(cur_date=cur_date)
        
        cur_action = self._process_action(cur_date=cur_date)
        self.portfolio.record_action(action=cur_action)
        
        self._update_access_counter()
        self.brain.step()
        
        self.counter += 1
        self.logger.info(f"=== 스텝 {self.counter-1} 완료 ===\n")

    def save_checkpoint(self, path: str, force: bool = False) -> None:
        if os.path.exists(path) and not force:
            raise FileExistsError(f"Path {path} already exists")
        if os.path.exists(path): shutil.rmtree(path)
        os.makedirs(os.path.join(path, "brain"), exist_ok=True)
        
        state_dict = {
            "agent_name": self.agent_name, "top_k": self.top_k, 
            "counter": self.counter, "trading_symbol": self.trading_symbol,
            "portfolio": self.portfolio, "chat_config": self.chat_config_save,
            "reflection_result_series_dict": self.reflection_result_series_dict,
        }
        with open(os.path.join(path, "state_dict.pkl"), "wb") as f:
            pickle.dump(state_dict, f)
        self.brain.save_checkpoint(path=os.path.join(path, "brain"), force=force)

    @classmethod
    def load_checkpoint(cls, path: str) -> "LLMAgent":
        with open(os.path.join(path, "state_dict.pkl"), "rb") as f:
            state_dict = pickle.load(f)
        brain = BrainDB.load_checkpoint(path=os.path.join(path, "brain"))
        
        # config에서 portfolio 설정을 로드해야 하지만, 여기서는 기본값으로 생성
        portfolio_config = state_dict.get("portfolio_config", {})
        
        class_obj = cls(
            agent_name=state_dict["agent_name"], trading_symbol=state_dict["trading_symbol"],
            brain_db=brain, top_k=state_dict["top_k"], chat_config=state_dict["chat_config"],
            portfolio_config=portfolio_config
        )
        class_obj.portfolio = state_dict["portfolio"]
        class_obj.reflection_result_series_dict = state_dict["reflection_result_series_dict"]
        class_obj.counter = state_dict["counter"]
        return class_obj