# puppy/prompts.py (수정 불필요)

# --------------------------------------------------------------------------
# 기초 자산 분석 (Foundational Asset Analysis)
FOUNDATIONAL_QUANT_PROMPT = """
Analyze the provided financial reports (10-K, 10-Q) and extract or calculate the following metrics for the company.
- 3-Year revenue Compound Annual Growth Rate (CAGR).
- Current Price-to-Earnings (P/E) Ratio.
- Current Price-to-Sales (P/S) Ratio.
- Current Debt-to-Equity Ratio.
- R&D spending as a percentage of revenue.

Financial Reports:
${mid_long_mem_texts}

${gr.complete_json_suffix_v2}
"""

FOUNDATIONAL_QUAL_PROMPT = """
Analyze the recent news and filings (e.g., 8-K) and summarize the key qualitative factors.
- What is the company's primary competitive moat?
- What are the most significant short-term catalysts (positive news)?
- What are the most significant short-term headwinds (negative news)?

Recent News & Filings:
${short_mem_texts}

${gr.complete_json_suffix_v2}
"""

# --------------------------------------------------------------------------
# 페르소나별 심층 분석 (Persona-Driven Deep Dive)
PERSONA_PROMPT_TEMPLATE = """
You are a {persona_name}. Your investment philosophy is: "{philosophy}"
Based on the foundational analysis below, provide your specific insights for the stock.

Foundational Analysis:
- Quantitative: ${quant_analysis}
- Qualitative: ${qual_analysis}

Your task is to provide a detailed analysis from your unique perspective.

${gr.complete_json_suffix_v2}
"""

# --------------------------------------------------------------------------
# 투자 위원회 토론 (Investment Committee Debate)
DEBATE_PROMPT = """
As the Chief Investment Officer (CIO), your job is to moderate a debate between your top analysts.
Here are their reports on the stock:
1.  **Value Investor's Report**: ${value_analysis}
2.  **Growth Investor's Report**: ${growth_analysis}
3.  **Technical Trader's Report**: ${technical_analysis}

Your tasks:
1.  Identify and summarize the single biggest point of disagreement (main contention) between these views.
2.  Based on all three reports, determine which argument is the most compelling right now and explain why.
3.  Simulate a brief Q&A: What is the one critical question the Value Investor would ask the Growth Investor, and what would be the likely response?

Synthesize this debate into a structured summary.

${gr.complete_json_suffix_v2}
"""

# --------------------------------------------------------------------------
# 최종 투자 집행 계획 수립 (Actionable Trade Plan)
FINAL_PLAN_PROMPT = """
As the Chief Investment Officer (CIO), you must make the final decision.
Synthesize all the prior analysis and the committee debate to formulate a concrete, actionable trade plan for the stock.

- **Foundational Analysis**: ${quant_analysis}, ${qual_analysis}
- **Persona Analyses**: ${persona_analyses}
- **Committee Debate Summary**: ${debate_summary}

Based on everything, construct the final plan. Be decisive. Your plan must include the final investment decision, the primary reason, a list of crucial document IDs that support your decision, and your confidence score.

${gr.complete_json_suffix_v2}
"""