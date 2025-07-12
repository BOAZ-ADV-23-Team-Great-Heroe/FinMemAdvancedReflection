import logging
import json
from typing import List, Callable, Dict, Any, Tuple, Optional
from enum import Enum 

import guardrails as gd
from pydantic import BaseModel, Field, condecimal


FAILED_TRADE_ANALYSIS_PROMPT = """
You are a Chief Risk Officer. The following trade plan resulted in a significant loss. 
Analyze the original plan and the provided context. Identify the single most likely reason for the failure.
Was it an overestimation of a catalyst? An underestimation of a risk? A flawed valuation model? 
Provide a concise one-sentence explanation for the failure.

Original Trade Plan:
{trade_plan}

Analysis Context Provided at the Time:
{analysis_context}
"""

class FailureAnalysis(BaseModel):
    failure_reason: str = Field(description="A concise, one-sentence explanation for why the trade failed.")

def analyze_failed_trade(endpoint_func: Callable, trade_plan: Dict, analysis_context: str) -> Dict:
    """실패한 거래의 원인을 분석합니다."""
    prompt = FAILED_TRADE_ANALYSIS_PROMPT.format(
        trade_plan=str(trade_plan),
        analysis_context=analysis_context
    )
    return call_llm_and_validate(endpoint_func, FailureAnalysis, prompt)

def call_llm_and_validate(
    endpoint_func: Callable, output_class: type[BaseModel], prompt: str, num_reasks: int = 1
) -> Dict:
    logger = logging.getLogger(__name__)
    guard = gd.Guard.from_pydantic(output_class=output_class)
    final_prompt = (
        f"{prompt}\n\n"
        "Provide your response strictly in the following JSON format. Do not include any other text or explanations. "
        "The JSON object must be enclosed in a markdown JSON block.\n"
        f"JSON Schema:\n```json\n{output_class.schema_json(indent=2)}\n```"
    )
    try:
        raw_llm_response = endpoint_func(final_prompt)
        try:
            json_str = raw_llm_response.split("```json")[1].split("```")[0].strip()
        except IndexError:
            logger.warning(f"LLM 응답에서 Markdown JSON 블록을 찾지 못했습니다. Raw Response: {raw_llm_response}")
            json_str = raw_llm_response
        parsed_json = json.loads(json_str)
        validated_data = output_class.parse_obj(parsed_json)
        return validated_data.dict()
    except Exception as e:
        logger.error(f"LLM 호출 또는 파싱 중 오류 발생: {e}", exc_info=True)
        return {"error": str(e)}


class QuantitativeAnalysis(BaseModel):
    three_year_cagr: Optional[float] = Field(None, description="3-Year revenue Compound Annual Growth Rate (CAGR).")
    pe_ratio: Optional[float] = Field(None, description="Current Price-to-Earnings (P/E) Ratio.")
    ps_ratio: Optional[float] = Field(None, description="Current Price-to-Sales (P/S) Ratio.")
    debt_to_equity: Optional[float] = Field(None, description="Current Debt-to-Equity Ratio.")
    rd_investment_ratio: Optional[float] = Field(None, description="R&D spending as a percentage of revenue.")

class QualitativeAnalysis(BaseModel):
    competitive_moat: str
    short_term_catalysts: List[str]
    short_term_headwinds: List[str]

def _run_foundational_analysis(
    endpoint_func: Callable, short_mem_texts: str, mid_long_mem_texts: str, initial_context: str
) -> Tuple[Dict, Dict]:
    prompt_quant = f"""{initial_context}
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

    prompt_qual = f"""{initial_context}
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
    endpoint_func: Callable, quant_analysis: Dict, qual_analysis: Dict, initial_context: str
) -> Dict[str, Dict]:
    personas = {
        "Value Investor": {"philosophy": "Buy wonderful companies at a fair price.", "model": ValueInvestorAnalysis},
        "Growth Investor": {"philosophy": "Invest in innovative companies that can change the world.", "model": GrowthInvestorAnalysis},
        "Technical & Momentum Trader": {"philosophy": "The trend is your friend. Analyze sentiment and price action.", "model": TechnicalTraderAnalysis}
    }
    results = {}
    for name, config in personas.items():
        prompt = f"""{initial_context}
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

def _run_committee_debate(endpoint_func: Callable, persona_analyses: Dict, initial_context: str) -> Dict:
    # ... (내부 로직은 동일, prompt에 initial_context 추가) ...
    prompt = f"""{initial_context}
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


FINAL_PLAN_PROMPT = """
As the Chief Investment Officer (CIO), you must make the final decision.
Synthesize all the prior analysis and the committee debate to formulate a concrete, actionable trade plan for the stock.

- **Foundational Analysis**: {quant_analysis}
- **Persona Analyses**: {persona_analyses}
- **Committee Debate Summary**: {debate_summary}
- **List of Available Document IDs for Reference**: {available_doc_ids}

Based on all available information, construct the final plan.
**Crucially, for the 'supporting_document_ids' field, you MUST select up to 3 of the most relevant IDs ONLY from the 'List of Available Document IDs for Reference' provided above.** Be decisive. This is a strict requirement.
"""

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
    persona_analyses: Dict, debate_summary: Dict,
    available_doc_ids: List[int], initial_context: str
) -> Dict:
    prompt = f"""{initial_context}
As the Chief Investment Officer (CIO), you must make the final decision.
Synthesize all the prior analysis and the committee debate to formulate a concrete, actionable trade plan for the stock.

- **Foundational Analysis**: {quant_analysis}
- **Persona Analyses**: {persona_analyses}
- **Committee Debate Summary**: {debate_summary}
- **List of Available Document IDs for Reference**: {available_doc_ids}

Based on all available information, construct the final plan.
**Crucially, for the 'supporting_document_ids' field, you MUST select up to 3 of the most relevant IDs ONLY from the 'List of Available Document IDs for Reference' provided above.** Be decisive.
"""
    
    validated_output = call_llm_and_validate(endpoint_func, ActionableTradePlan, prompt)
    
    if validated_output and 'investment_decision' in validated_output and isinstance(validated_output.get('investment_decision'), Enum):
        validated_output['investment_decision'] = validated_output['investment_decision'].value
    return validated_output

def run_investment_committee_analysis(
    endpoint_func: Callable[[str], str],
    short_memory_texts_with_ids: List[Tuple[int, str]],
    mid_long_memory_texts_with_ids: List[Tuple[int, str]],
    initial_context: str, # "오답 노트"를 전달받는 인자
) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    full_analysis_report = {}
    
    def format_mem_for_prompt(mem_with_ids: List[Tuple[int, str]]) -> str:
        return "\n".join([f"[ID: {id}] {text}" for id, text in mem_with_ids])

    short_mem_for_prompt = format_mem_for_prompt(short_memory_texts_with_ids)
    mid_long_mem_for_prompt = format_mem_for_prompt(mid_long_memory_texts_with_ids)
    
    try:
        logger.info("단계 1: 기초 자산 분석 실행 중...")
        quant_analysis, qual_analysis = _run_foundational_analysis(endpoint_func, short_mem_for_prompt, mid_long_mem_for_prompt, initial_context)
        full_analysis_report["foundational_analysis"] = {"quantitative": quant_analysis, "qualitative": qual_analysis}
        if "error" in quant_analysis or "error" in qual_analysis: raise ValueError("기초 분석 실패")

        logger.info("단계 2: 페르소나별 심층 분석 실행 중...")
        persona_analyses = _run_persona_analysis(endpoint_func, quant_analysis, qual_analysis, initial_context)
        full_analysis_report["persona_analyses"] = persona_analyses
        if any("error" in r for r in persona_analyses.values()): raise ValueError("페르소나 분석 실패")

        logger.info("단계 3: 투자 위원회 토론 시뮬레이션 중...")
        debate_summary = _run_committee_debate(endpoint_func, persona_analyses, initial_context)
        full_analysis_report["committee_debate"] = debate_summary
        if "error" in debate_summary: raise ValueError("위원회 토론 실패")

        logger.info("단계 4: 최종 투자 집행 계획 수립 중...")
        all_doc_ids = [doc_id for doc_id, text in short_memory_texts_with_ids + mid_long_memory_texts_with_ids]
        final_plan = _formulate_final_plan(
            endpoint_func, quant_analysis, qual_analysis, 
            persona_analyses, debate_summary,
            available_doc_ids=all_doc_ids,
            initial_context=initial_context
        )
        full_analysis_report["final_actionable_plan"] = final_plan
        if "error" in final_plan: raise ValueError("최종 계획 수립 실패")
        
        logger.info("모든 분석 단계가 성공적으로 완료되었습니다.")
        return full_analysis_report

    except Exception as e:
        logger.error(f"투자 위원회 분석 중 최종 오류 발생: {e}", exc_info=True)
        return {"error": str(e), "partial_report": full_analysis_report}