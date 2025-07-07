# # sourcery skip: dont-import-test-modules
# from rich import print
# import logging
# import guardrails as gd
# from datetime import date
# from .run_type import RunMode
# from pydantic import BaseModel, Field
# from httpx import HTTPStatusError
# from guardrails.validators import ValidChoices
# from typing import List, Callable, Dict, Union, Any, Tuple
# from .chat import LongerThanContextError
# from .prompts import (
#     short_memory_id_desc,
#     mid_memory_id_desc,
#     long_memory_id_desc,
#     reflection_memory_id_desc,
#     train_prompt,
#     train_memory_id_extract_prompt,
#     train_trade_reason_summary,
#     train_investment_info_prefix,
#     test_prompt,
#     test_trade_reason_summary,
#     test_memory_id_extract_prompt,
#     test_invest_action_choice,
#     test_investment_info_prefix,
#     test_sentiment_explanation,
#     test_momentum_explanation,
# )


# def _train_memory_factory(memory_layer: str, id_list: List[int]):
#     """
#     [학습용] 특정 메모리 계층(short, mid, long 등)에서 LLM이 선택한 메모리 ID를 검증하기 위한 Pydantic 모델을 동적으로 생성
#     Args:
#         memory_layer (str): 메모리 계층의 이름 (예: "long-level").
#         id_list (List[int]): LLM이 선택할 수 있는 유효한 메모리 ID 목록.

#     Returns:
#         Pydantic BaseModel: 선택된 메모리 ID를 검증하는 `Memory` 모델.
#     """
#     class Memory(BaseModel):
#         memory_index: int = Field(
#             ...,
#             description=train_memory_id_extract_prompt.format(
#                 memory_layer=memory_layer
#             ),
#             validators=[ValidChoices(id_list, on_fail="reask")],  # type: ignore
#         )

#     return Memory


# def _test_memory_factory(memory_layer: str, id_list: List[int]):
#     class Memory(BaseModel):
#         memory_index: int = Field(
#             ...,
#             description=test_memory_id_extract_prompt.format(memory_layer=memory_layer),
#             validators=[ValidChoices(id_list)],  # type: ignore
#         )

#     return Memory


# # train + test reflection model
# def _train_reflection_factory(
#     short_id_list: List[int],
#     mid_id_list: List[int],
#     long_id_list: List[int],
#     reflection_id_list: List[int],
# ):
#     """
#     [학습용] LLM의 최종 출력 구조를 정의하고 검증하는 메인 Pydantic 모델(`InvestInfo`)을 생성
#     각 메모리 계층의 ID 리스트를 받아, 해당 메모리가 존재할 경우에만 필드를 동적으로 추가

#     Args:
#         short_id_list (List[int]): 단기 메모리 ID 목록.
#         mid_id_list (List[int]): 중기 메모리 ID 목록.
#         long_id_list (List[int]): 장기 메모리 ID 목록.
#         reflection_id_list (List[int]): 리플렉션 메모리 ID 목록.

#     Returns:
#         Pydantic BaseModel: LLM의 학습용 출력 전체를 검증하는 `InvestInfo` 모델
#     """
#     LongMem = _train_memory_factory("long-level", long_id_list)
#     MidMem = _train_memory_factory("mid-level", mid_id_list)
#     ShortMem = _train_memory_factory("short-level", short_id_list)
#     ReflectionMem = _train_memory_factory("reflection-level", reflection_id_list)

#     class InvestInfo(BaseModel):
#         # 각 메모리 계층의 ID 리스트가 비어있지 않은 경우
#         if reflection_id_list:
#             reflection_memory_index: List[ReflectionMem] = Field(
#                 ...,
#                 description=reflection_memory_id_desc,
#             )
#         if long_id_list:
#             long_memory_index: List[LongMem] = Field(
#                 ...,
#                 description=long_memory_id_desc,
#             )
#         if mid_id_list:
#             middle_memory_index: List[MidMem] = Field(
#                 ...,
#                 description=mid_memory_id_desc,
#             )
#         if short_id_list:
#             short_memory_index: List[ShortMem] = Field(
#                 ...,
#                 description=short_memory_id_desc,
#             )
#         summary_reason: str = Field(
#             ...,
#             description=train_trade_reason_summary,
#         )

#     return InvestInfo


# def _test_reflection_factory(
#     short_id_list: List[int],
#     mid_id_list: List[int],
#     long_id_list: List[int],
#     reflection_id_list: List[int],
# ):
#     """
#     [테스트용] LLM의 최종 출력 구조를 정의하고 검증하는 메인 Pydantic 모델(`InvestInfo`)을 생성
#     학습용 모델과 유사하지만, 실제 투자 결정('buy', 'sell', 'hold')을 하는 필드가 추가

#     Args:
#         short_id_list (List[int]): 단기 메모리 ID 목록.
#         mid_id_list (List[int]): 중기 메모리 ID 목록.
#         long_id_list (List[int]): 장기 메모리 ID 목록.
#         reflection_id_list (List[int]): 리플렉션 메모리 ID 목록.

#     Returns:
#         Pydantic BaseModel: LLM의 테스트용 출력 전체를 검증하는 `InvestInfo` 모델.
#     """
#     LongMem = _test_memory_factory("long-level", long_id_list)
#     MidMem = _test_memory_factory("mid-level", mid_id_list)
#     ShortMem = _test_memory_factory("short-level", short_id_list)
#     ReflectionMem = _test_memory_factory("reflection-level", reflection_id_list)

#     class InvestInfo(BaseModel):
#         # 'buy', 'sell', 'hold' 중 하나의 투자 결정을 내리는 필드
#         investment_decision: str = Field(
#             ...,
#             description=test_invest_action_choice,
#             validators=[ValidChoices(choices=["buy", "sell", "hold"])],  # type: ignore
#         )
#         # 투자 결정 이유를 요약
#         summary_reason: str = Field(
#             ...,
#             description=test_trade_reason_summary,
#         )
#         if short_id_list:
#             short_memory_index: List[ShortMem] = Field(
#                 ...,
#                 description=short_memory_id_desc,
#             )
#         if mid_id_list:
#             middle_memory_index: List[MidMem] = Field(
#                 ...,
#                 description=mid_memory_id_desc,
#             )
#         if long_id_list:
#             long_memory_index: List[LongMem] = Field(
#                 ...,
#                 description=long_memory_id_desc,
#             )
#         if reflection_id_list:
#             reflection_memory_index: List[ReflectionMem] = Field(
#                 ...,
#                 description=reflection_memory_id_desc,
#             )

#     return InvestInfo


# def _format_memories(
#     short_memory: Union[List[str], None] = None,
#     short_memory_id: Union[List[int], None] = None,
#     mid_memory: Union[List[str], None] = None,
#     mid_memory_id: Union[List[int], None] = None,
#     long_memory: Union[List[str], None] = None,
#     long_memory_id: Union[List[int], None] = None,
#     reflection_memory: Union[List[str], None] = None,
#     reflection_memory_id: Union[List[int], None] = None,
# ) -> Tuple[
#     List[str],
#     List[int],
#     List[str],
#     List[int],
#     List[str],
#     List[int],
#     List[str],
#     List[int],
# ]:
#     """
#     메모리 목록을 포맷팅. Guardrails의 ValidChoices 검증기가 최소 2개의 선택지를 요구하기 때문에,
#     메모리가 없거나 1개만 있을 경우 플레이스홀더나 중복 데이터를 추가하여 항상 2개 이상의 항목을 갖도록 한다.

#     Args:
#         (각 메모리 및 ID 목록들): 각 계층별 메모리 텍스트와 ID의 리스트.

#     Returns:
#         Tuple: 포맷팅이 완료된 각 계층별 메모리 텍스트와 ID의 튜플.
#     """

#     # 각 메모리 목록의 길이가 1 이하일 경우, 플레이스홀더나 중복 데이터를 추가.
#     if (short_memory is None) or len(short_memory) == 0:
#         short_memory = ["No short-term information.", "No short-term information."]
#         short_memory_id = [-1, -1]
#     elif len(short_memory) == 1:
#         short_memory = [short_memory[0], short_memory[0]]
#         short_memory_id = [short_memory_id[0], short_memory_id[0]]  # type: ignore
#     if (mid_memory is None) or len(mid_memory) == 0:
#         mid_memory = ["No mid-term information.", "No mid-term information."]
#         mid_memory_id = [-1, -1]
#     elif len(mid_memory) == 1:
#         mid_memory = [mid_memory[0], mid_memory[0]]
#         mid_memory_id = [mid_memory_id[0], mid_memory_id[0]]  # type: ignore
#     if (long_memory is None) or len(long_memory) == 0:
#         long_memory = ["No long-term information.", "No long-term information."]
#         long_memory_id = [-1, -1]
#     elif len(long_memory) == 1:
#         long_memory = [long_memory[0], long_memory[0]]
#         long_memory_id = [long_memory_id[0], long_memory_id[0]]  # type: ignore
#     if (reflection_memory is None) or len(reflection_memory) == 0:
#         reflection_memory = [
#             "No reflection-term information.",
#             "No reflection-term information.",
#         ]
#         reflection_memory_id = [-1, -1]
#     elif len(reflection_memory) == 1:
#         reflection_memory = [reflection_memory[0], reflection_memory[0]]
#         reflection_memory_id = [reflection_memory_id[0], reflection_memory_id[0]]  # type: ignore

#     return (
#         short_memory,
#         short_memory_id,
#         mid_memory,
#         mid_memory_id,
#         long_memory,
#         long_memory_id,
#         reflection_memory,
#         reflection_memory_id,
#     )


# def _delete_placeholder_info(validated_output: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Guardrails를 통해 검증된 LLM 출력에서 플레이스홀더(-1) 정보를 제거
#     LLM이 메모리가 없음을 나타내는 -1을 선택했을 경우, 해당 메모리 필드를 최종 결과에서 삭제

#     Args:
#         validated_output (Dict[str, Any]): Guardrails가 검증한 LLM의 출력 딕셔너리.

#     Returns:
#         Dict[str, Any]: 플레이스홀더 정보가 제거된 최종 출력 딕셔너리.
#     """

#     if "reflection_memory_index" in validated_output and (
#         (validated_output["reflection_memory_index"])
#         and (validated_output["reflection_memory_index"][0]["memory_index"] == -1)
#     ):
#         del validated_output["reflection_memory_index"]
#     if "long_memory_index" in validated_output and (
#         (validated_output["long_memory_index"])
#         and (validated_output["long_memory_index"][0]["memory_index"] == -1)
#     ):
#         del validated_output["long_memory_index"]
#     if "middle_memory_index" in validated_output and (
#         (validated_output["middle_memory_index"])
#         and (validated_output["middle_memory_index"][0]["memory_index"] == -1)
#     ):
#         del validated_output["middle_memory_index"]
#     if "short_memory_index" in validated_output and (
#         (validated_output["short_memory_index"])
#         and (validated_output["short_memory_index"][0]["memory_index"] == -1)
#     ):
#         del validated_output["short_memory_index"]

#     return validated_output


# def _add_momentum_info(momentum: int, investment_info: str) -> str:
#     """
#     주식의 모멘텀 정보(과거 3일 누적 수익률)를 텍스트로 변환하여 프롬프트에 추가

#     Args:
#         momentum (int): 모멘텀 값 (1: 긍정, 0: 중립, -1: 부정).
#         investment_info (str): LLM에게 전달될 프롬프트 문자열.

#     Returns:
#         str: 모멘텀 설명이 추가된 프롬프트 문자열.
#     """
#     if momentum == -1:
#         investment_info += (
#             "The cumulative return of past 3 days for this stock is negative."
#         )

#     elif momentum == 0:
#         investment_info += (
#             "The cumulative return of past 3 days for this stock is zero."
#         )

#     elif momentum == 1:
#         investment_info += (
#             "The cumulative return of past 3 days for this stock is positive."
#         )

#     return investment_info


# def _train_response_model_invest_info(
#     cur_date: date,
#     symbol: str,
#     future_record: Dict[str, float | str],
#     short_memory: List[str],
#     short_memory_id: List[int],
#     mid_memory: List[str],
#     mid_memory_id: List[int],
#     long_memory: List[str],
#     long_memory_id: List[int],
#     reflection_memory: List[str],
#     reflection_memory_id: List[int],
# ):
#     """
#     [학습용] LLM에게 전달할 전체 프롬프트(`investment_info`)와 출력 검증 모델(`response_model`)을 생성

#     Args:
#         (다양한 인자들): 현재 날짜, 주식 심볼, 미래 수익률, 각 메모리 계층의 정보 및 ID.

#     Returns:
#         Tuple:
#             - response_model: LLM 출력을 검증할 Pydantic 모델.
#             - investment_info: LLM에게 전달될 최종 프롬프트 텍스트.
#     """
#     # 학습용 출력 검증 모델을 생성
#     response_model = _train_reflection_factory(
#         short_id_list=short_memory_id,
#         mid_id_list=mid_memory_id,
#         long_id_list=long_memory_id,
#         reflection_id_list=reflection_memory_id,
#     )

#     # 프롬프트의 기본 접두사를 설정
#     investment_info = train_investment_info_prefix.format(
#         cur_date=cur_date, symbol=symbol, future_record=future_record
#     )

#     # 각 메모리 정보를 프롬프트에 추가
#     if short_memory:
#         investment_info += "The short-term information:\n"
#         investment_info += "\n".join(
#             [f"{i[0]}. {i[1].strip()}" for i in zip(short_memory_id, short_memory)]
#         )
#         investment_info += "\n\n"
#     if mid_memory:
#         investment_info += "The mid-term information:\n"
#         investment_info += "\n".join(
#             [f"{i[0]}. {i[1].strip()}" for i in zip(mid_memory_id, mid_memory)]
#         )
#         investment_info += "\n\n"
#     if long_memory:
#         investment_info += "The long-term information:\n"
#         investment_info += "\n".join(
#             [f"{i[0]}. {i[1].strip()}" for i in zip(long_memory_id, long_memory)]
#         )
#         investment_info += "\n\n"
#     if reflection_memory:
#         investment_info += "The reflection-term information:\n"
#         investment_info += "\n".join(
#             [f"{i[0]}. {i[1].strip()}" for i in zip(reflection_memory_id, reflection_memory)]
#         )
#         investment_info += "\n\n"

#     return response_model, investment_info


# def _test_response_model_invest_info(
#     cur_date: date,
#     symbol: str,
#     short_memory: List[str],
#     short_memory_id: List[int],
#     mid_memory: List[str],
#     mid_memory_id: List[int],
#     long_memory: List[str],
#     long_memory_id: List[int],
#     reflection_memory: List[str],
#     reflection_memory_id: List[int],
#     momentum: Union[int, None] = None,
# ):
#     # pydantic reflection model
#     response_model = _test_reflection_factory(
#         short_id_list=short_memory_id,
#         mid_id_list=mid_memory_id,
#         long_id_list=long_memory_id,
#         reflection_id_list=reflection_memory_id,
#     )
#     # investment info + memories
#     investment_info = test_investment_info_prefix.format(
#         symbol=symbol, cur_date=cur_date
#     )
#     if short_memory:
#         investment_info += "The short-term information:\n"
#         investment_info += "\n".join(
#             [f"{i[0]}. {i[1].strip()}" for i in zip(short_memory_id, short_memory)]
#         )
#         investment_info += test_sentiment_explanation
#         investment_info += "\n\n"
#     if mid_memory:
#         investment_info += "The mid-term information:\n"
#         investment_info += "\n".join(
#             [f"{i[0]}. {i[1].strip()}" for i in zip(mid_memory_id, mid_memory)]
#         )
#         investment_info += "\n\n"
#     if long_memory:
#         investment_info += "The long-term information:\n"
#         investment_info += "\n".join(
#             [f"{i[0]}. {i[1].strip()}" for i in zip(long_memory_id, long_memory)]
#         )
#         investment_info += "\n\n"
#     if reflection_memory:
#         investment_info += "The reflection-term information:\n"
#         investment_info += "\n".join(
#             [f"{i[0]}. {i[1].strip()}" for i in zip(reflection_memory_id, reflection_memory)]
#         )
#         investment_info += "\n\n"
#     if momentum:
#         investment_info += test_momentum_explanation
#         investment_info = _add_momentum_info(momentum, investment_info)

#     return response_model, investment_info


# def trading_reflection(
#     cur_date: date,
#     endpoint_func: Callable[[str], str],
#     symbol: str,
#     run_mode: RunMode,
#     logger: logging.Logger,
#     momentum: Union[int, None] = None,
#     future_record: Union[Dict[str, float | str], None] = None,
#     short_memory: Union[List[str], None] = None,
#     short_memory_id: Union[List[int], None] = None,
#     mid_memory: Union[List[str], None] = None,
#     mid_memory_id: Union[List[int], None] = None,
#     long_memory: Union[List[str], None] = None,
#     long_memory_id: Union[List[int], None] = None,
#     reflection_memory: Union[List[str], None] = None,
#     reflection_memory_id: Union[List[int], None] = None,
# ) -> Dict[str, Any]:
#     """
#     메인 리플렉션 함수. LLM을 호출하여 투자 결정을 추론하고, 그 결과를 구조화된 형식으로 반환
#     전체 추론 과정을 오케스트레이션

#     Args:
#         (다양한 인자들): 추론에 필요한 모든 정보(날짜, LLM 엔드포인트, 심볼, 모드, 메모리 등).

#     Returns:
#         Dict[str, Any]: LLM의 추론 결과. (예: {"investment_decision": "buy", "summary_reason": "...", ...})
#                          오류 발생 시 빈 딕셔너리나 기본값을 반환
#     """
#     # 1. 메모리 포맷팅: Guardrails 요구사항에 맞게 메모리 목록을 준비합
#     (
#         short_memory,
#         short_memory_id,
#         mid_memory,
#         mid_memory_id,
#         long_memory,
#         long_memory_id,
#         reflection_memory,
#         reflection_memory_id,
#     ) = _format_memories(
#         short_memory=short_memory,
#         short_memory_id=short_memory_id,
#         mid_memory=mid_memory,
#         mid_memory_id=mid_memory_id,
#         long_memory=long_memory,
#         long_memory_id=long_memory_id,
#         reflection_memory=reflection_memory,
#         reflection_memory_id=reflection_memory_id,
#     )

#     # 2. 실행 모드(학습/테스트)에 따라 적절한 프롬프트와 검증 모델을 선택하고 생성
#     if run_mode == RunMode.Train:
#         response_model, investment_info = _train_response_model_invest_info(
#             cur_date=cur_date,
#             symbol=symbol,
#             future_record=future_record,  # type: ignore
#             short_memory=short_memory,
#             short_memory_id=short_memory_id,
#             mid_memory=mid_memory,
#             mid_memory_id=mid_memory_id,
#             long_memory=long_memory,
#             long_memory_id=long_memory_id,
#             reflection_memory=reflection_memory,
#             reflection_memory_id=reflection_memory_id,
#         )
#         cur_prompt = train_prompt
#     else:
#         response_model, investment_info = _test_response_model_invest_info(
#             cur_date=cur_date,
#             symbol=symbol,
#             short_memory=short_memory,
#             short_memory_id=short_memory_id,
#             mid_memory=mid_memory,
#             mid_memory_id=mid_memory_id,
#             long_memory=long_memory,
#             long_memory_id=long_memory_id,
#             reflection_memory=reflection_memory,
#             reflection_memory_id=reflection_memory_id,
#             momentum=momentum,
#         )
#         cur_prompt = test_prompt

#     # 3. Guardrails 객체를 생성. Pydantic 모델과 프롬프트를 사용하여 LLM의 출력을 제어
#     guard = gd.Guard.from_pydantic(
#         output_class=response_model, prompt=cur_prompt, num_reasks=1
#     )

#     try:
#         # 4. Guardrails를 통해 LLM 엔드포인트를 호출
#         # 프롬프트 템플릿의 `{investment_info}` 부분에 생성된 프롬프트 텍스트를 주입
#         validated_outcomes = guard(
#             endpoint_func,
#             prompt_params={"investment_info": investment_info},
#         )

#         logger.info("Guardrails Raw LLM Outputs")
#         for i, o in enumerate(guard.history[0].raw_outputs):
#             logger.info(f"Reask {i}")
#             logger.info(o)
#             logger.info("\n\n")
        
#         # 5. 검증된 출력이 유효하지 않은 경우, 기본값을 반환
#         if (validated_outcomes.validated_output is None) or (
#             not isinstance(validated_outcomes.validated_output, dict)
#         ):
#             logger.info(f"reflection failed for {symbol}")
#             if run_mode == RunMode.Train:
#                 return {"summary_reason": validated_outcomes.__dict__['reask'].__dict__['fail_results'][0].__dict__['error_message'], "short_memory_index": None, "middle_memory_index": None, "long_memory_index": None, "reflection_memory_index": None}
#             else:
#                 return {"investment_decision" : "hold", "summary_reason": validated_outcomes.__dict__['reask'].__dict__['fail_results'][0].__dict__['error_message'], "short_memory_index": None, "middle_memory_index": None, "long_memory_index": None, "reflection_memory_index": None}
        
#         # 6. 최종적으로 검증되고 플레이스홀더가 제거된 결과를 반환
#         return _delete_placeholder_info(validated_outcomes.validated_output)

#     except Exception as e:
#         if isinstance(e.__context__, LongerThanContextError):
#             raise LongerThanContextError from e
#         logger.info("Wrong again!!!!!")
#         logger.error(e)
#         return _delete_placeholder_info({})


# (Multi-Persona Investment Committee Version)

import logging
import json
from typing import List, Callable, Dict, Any, Tuple
from enum import Enum 

import guardrails as gd
from pydantic import BaseModel, Field, condecimal, ValidationError

# --------------------------------------------------------------------------
# 헬퍼 함수: LLM 호출 및 검증 로직 (최종 수정)
# --------------------------------------------------------------------------
def call_llm_and_validate(
    endpoint_func: Callable,
    output_class: type[BaseModel],
    prompt: str, # 이제 완성된 프롬프트를 직접 받음
    num_reasks: int = 1
) -> Dict:
    """
    LLM을 호출하고, 응답에서 JSON을 직접 추출하여 Pydantic으로 검증합니다.
    """
    logger = logging.getLogger(__name__)
    
    # 1. Pydantic 모델로부터 Guard 객체 생성
    guard = gd.Guard.from_pydantic(output_class=output_class)
    
    # 2. LLM에게 JSON 출력을 지시하는 명확한 지시문을 프롬프트에 추가합니다.
    final_prompt = (
        f"{prompt}\n\n"
        "Provide your response strictly in the following JSON format. Do not include any other text or explanations. "
        "The JSON object must be enclosed in a markdown JSON block.\n"
        f"JSON Schema:\n```json\n{output_class.schema_json(indent=2)}\n```"
    )
    
    try:
        # 3. LLM 호출
        raw_llm_response = endpoint_func(final_prompt)
        
        # 4. LLM 응답에서 JSON 블록만 추출
        try:
            json_str = raw_llm_response.split("```json")[1].split("```")[0].strip()
        except IndexError:
            logger.warning(f"LLM 응답에서 Markdown JSON 블록을 찾지 못했습니다. Raw Response: {raw_llm_response}")
            json_str = raw_llm_response

        # 5. JSON 문자열을 딕셔너리로 파싱하고 Pydantic으로 검증
        parsed_json = json.loads(json_str)
        validated_data = output_class.parse_obj(parsed_json)
        
        return validated_data.dict()

    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"파싱 또는 검증 오류: {e}. Raw Response: {raw_llm_response}")
        # Guardrails의 재질의 기능 활용 시도
        try:
            raw_llm_response, validated_output = guard(endpoint_func, prompt=final_prompt, num_reasks=num_reasks)
            return validated_output if validated_output else {"error": "Re-ask failed."}
        except Exception as reask_e:
            logger.error(f"재질의 중 오류 발생: {reask_e}")
            return {"error": str(reask_e)}
    except Exception as e:
        logger.error(f"LLM 호출 또는 데이터 처리 중 예상치 못한 오류 발생: {e}", exc_info=True)
        return {"error": str(e)}


# --------------------------------------------------------------------------
# Pydantic 모델 정의
# --------------------------------------------------------------------------
class QuantitativeAnalysis(BaseModel):
    three_year_cagr: float
    pe_ratio: float
    ps_ratio: float
    debt_to_equity: float
    rd_investment_ratio: float

class QualitativeAnalysis(BaseModel):
    competitive_moat: str
    short_term_catalysts: List[str]
    short_term_headwinds: List[str]

def _run_foundational_analysis(
    endpoint_func: Callable, short_mem_texts: str, mid_long_mem_texts: str
) -> Tuple[Dict, Dict]:
    prompt_quant = f"""
Analyze the provided financial reports (10-K, 10-Q) and extract or calculate the following metrics for the company.
- 3-Year revenue Compound Annual Growth Rate (CAGR).
- Current Price-to-Earnings (P/E) Ratio.
- Current Price-to-Sales (P/S) Ratio.
- Current Debt-to-Equity Ratio.
- R&D spending as a percentage of revenue.

Financial Reports:
{mid_long_mem_texts}
"""
    quant_analysis = call_llm_and_validate(endpoint_func, QuantitativeAnalysis, prompt_quant)

    prompt_qual = f"""
Analyze the recent news and filings (e.g., 8-K) and summarize the key qualitative factors.
- What is the company's primary competitive moat?
- What are the most significant short-term catalysts (positive news)?
- What are the most significant short-term headwinds (negative news)?

Recent News & Filings:
{short_mem_texts}
"""
    qual_analysis = call_llm_and_validate(endpoint_func, QualitativeAnalysis, prompt_qual)
    return quant_analysis, qual_analysis

class ValueInvestorAnalysis(BaseModel):
    intrinsic_value_assessment: str
    is_undervalued: bool
    long_term_risks: List[str]

class GrowthInvestorAnalysis(BaseModel):
    disruptive_potential: str
    tam_expansion: str
    key_growth_catalysts: List[str]

class TechnicalTraderAnalysis(BaseModel):
    market_sentiment: str
    key_support_level: float
    key_resistance_level: float
    
def _run_persona_analysis(
    endpoint_func: Callable, quant_analysis: Dict, qual_analysis: Dict
) -> Dict[str, Dict]:
    personas = {
        "Value Investor": {"philosophy": "Buy wonderful companies at a fair price.", "model": ValueInvestorAnalysis},
        "Growth Investor": {"philosophy": "Invest in innovative companies that can change the world.", "model": GrowthInvestorAnalysis},
        "Technical & Momentum Trader": {"philosophy": "The trend is your friend. Analyze sentiment and price action.", "model": TechnicalTraderAnalysis}
    }
    results = {}
    for name, config in personas.items():
        prompt = f"""
You are a {name}. Your investment philosophy is: "{config['philosophy']}"
Based on the foundational analysis below, provide your specific insights for the stock.

Foundational Analysis:
- Quantitative: {quant_analysis}
- Qualitative: {qual_analysis}

Your task is to provide a detailed analysis from your unique perspective."""
        results[name] = call_llm_and_validate(endpoint_func, config["model"], prompt)
    return results

class DebateSummary(BaseModel):
    main_contention: str
    strongest_argument: str
    simulated_qa: str

def _run_committee_debate(endpoint_func: Callable, persona_analyses: Dict) -> Dict:
    prompt = f"""
As the Chief Investment Officer (CIO), your job is to moderate a debate between your top analysts.
Here are their reports on the stock:
1.  **Value Investor's Report**: {persona_analyses.get("Value Investor", {})}
2.  **Growth Investor's Report**: {persona_analyses.get("Growth Investor", {})}
3.  **Technical Trader's Report**: {persona_analyses.get("Technical & Momentum Trader", {})}

Your tasks:
1.  Identify and summarize the single biggest point of disagreement (main contention) between these views.
2.  Based on all three reports, determine which argument is the most compelling right now and explain why.
3.  Simulate a brief Q&A: What is the one critical question the Value Investor would ask the Growth Investor, and what would be the likely response?

Synthesize this debate into a structured summary."""
    return call_llm_and_validate(endpoint_func, DebateSummary, prompt)

class InvestmentDecisionEnum(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class ActionableTradePlan(BaseModel):
    investment_decision: InvestmentDecisionEnum
    primary_reason: str
    supporting_document_ids: List[int]
    confidence_score: condecimal(ge=0.0, le=1.0)
    position_sizing: condecimal(ge=0.0, le=1.0)
    strategy_definition: str
    time_horizon: str
    entry_price_range: str
    stop_loss_rule: str

def _formulate_final_plan(
    endpoint_func: Callable, quant_analysis: Dict, qual_analysis: Dict,
    persona_analyses: Dict, debate_summary: Dict
) -> Dict:
    prompt = f"""
As the Chief Investment Officer (CIO), you must make the final decision.
Synthesize all the prior analysis and the committee debate to formulate a concrete, actionable trade plan for the stock.
- **Foundational Analysis**: {quant_analysis}
- **Persona Analyses**: {persona_analyses}
- **Committee Debate Summary**: {debate_summary}
Based on everything, construct the final plan. Be decisive."""
    
    validated_output = call_llm_and_validate(endpoint_func, ActionableTradePlan, prompt)
    
    if validated_output and 'investment_decision' in validated_output and isinstance(validated_output.get('investment_decision'), Enum):
        validated_output['investment_decision'] = validated_output['investment_decision'].value
    return validated_output

def run_investment_committee_analysis(
    endpoint_func: Callable[[str], str],
    short_memory_texts_with_ids: List[Tuple[int, str]],
    mid_long_memory_texts_with_ids: List[Tuple[int, str]],
) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    full_analysis_report = {}
    
    def format_mem_for_prompt(mem_with_ids: List[Tuple[int, str]]) -> str:
        return "\n".join([f"[ID: {id}] {text}" for id, text in mem_with_ids])

    short_mem_for_prompt = format_mem_for_prompt(short_memory_texts_with_ids)
    mid_long_mem_for_prompt = format_mem_for_prompt(mid_long_memory_texts_with_ids)
    
    try:
        logger.info("단계 1: 기초 자산 분석 실행 중...")
        quant_analysis, qual_analysis = _run_foundational_analysis(endpoint_func, short_mem_for_prompt, mid_long_mem_for_prompt)
        full_analysis_report["foundational_analysis"] = {"quantitative": quant_analysis, "qualitative": qual_analysis}
        if "error" in quant_analysis or "error" in qual_analysis: raise ValueError(f"기초 분석 실패: {quant_analysis.get('error') or qual_analysis.get('error')}")

        logger.info("단계 2: 페르소나별 심층 분석 실행 중...")
        persona_analyses = _run_persona_analysis(endpoint_func, quant_analysis, qual_analysis)
        full_analysis_report["persona_analyses"] = persona_analyses
        if any("error" in r for r in persona_analyses.values()): raise ValueError("페르소나 분석 중 하나 이상이 실패했습니다.")

        logger.info("단계 3: 투자 위원회 토론 시뮬레이션 중...")
        debate_summary = _run_committee_debate(endpoint_func, persona_analyses)
        full_analysis_report["committee_debate"] = debate_summary
        if "error" in debate_summary: raise ValueError(f"위원회 토론 실패: {debate_summary.get('error')}")

        logger.info("단계 4: 최종 투자 집행 계획 수립 중...")
        final_plan = _formulate_final_plan(endpoint_func, quant_analysis, qual_analysis, persona_analyses, debate_summary)
        full_analysis_report["final_actionable_plan"] = final_plan
        if "error" in final_plan: raise ValueError(f"최종 계획 수립 실패: {final_plan.get('error')}")
        
        logger.info("모든 분석 단계가 성공적으로 완료되었습니다.")
        return full_analysis_report

    except Exception as e:
        logger.error(f"투자 위원회 분석 중 최종 오류 발생: {e}", exc_info=True)
        return {"error": str(e), "partial_report": full_analysis_report}