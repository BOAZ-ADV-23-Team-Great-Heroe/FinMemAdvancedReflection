import os
import pickle
import faiss
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- 1. 설정 (Configuration) ---
DB_FAISS_PATH = "./vector_db/"
EMBEDDING_MODEL = "LegalInsight/PretrainedModel-base"
LLM_MODEL = "gpt-4-turbo" # 또는 gpt-3.5-turbo 등

# --- 2. 벡터 DB 및 리트리버 로드 ---
def load_retriever():
    """저장된 FAISS 인덱스와 문서를 로드하여 리트리버를 생성합니다."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    index = faiss.read_index(os.path.join(DB_FAISS_PATH, "legal.index", "index.faiss"))
    
    with open(os.path.join(DB_FAISS_PATH, "legal.index", "index.pkl"), "rb") as f:
        pkl = pickle.load(f)
        
    vectorstore = faiss.FAISS(embeddings.embed_query, index, pkl, {})
    return vectorstore.as_retriever(search_kwargs={'k': 3})

# --- 3. Step-back Prompting 체인 정의 ---
def create_step_back_chain(llm):
    """Step-back 질문을 생성하는 체인을 만듭니다."""
    few_shot_examples = [
        {
            "input": "제 친구가 A 보험사에서 변액보험을 가입했는데, 투자 손실이 크다고 합니다. 이거 불완전판매 아닌가요?",
            "output": "변액보험과 같은 투자성 상품 판매 시 금융회사가 준수해야 할 법적 의무는 무엇이며, 불완전판매의 법적 기준은 무엇인가?"
        },
        {
            "input": "은행에서 대출 상담을 받았는데, 상담사가 자꾸 다른 적금 상품 가입을 권유해요. 이거 괜찮은 건가요?",
            "output": "금융회사가 대출성 상품과 다른 금융상품을 연계하여 판매하는 행위(꺾기)에 대한 법적 규제는 무엇인가?"
        },
    ]
    example_prompt = ChatPromptTemplate.from_messages(
        [("human", "{input}"), ("ai", "{output}")]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=few_shot_examples,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 유능한 법률 전문가입니다. 사용자의 구체적인 질문을 바탕으로, 문제의 핵심을 파악할 수 있는 더 근본적이고 일반적인 법률 질문을 1개 생성해주세요."),
        few_shot_prompt,
        ("user", "{question}"),
    ])
    return prompt | llm | StrOutputParser()

# --- 4. 최종 답변 생성 프롬프트 (CoT 적용) ---
FINAL_RESPONSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 대한민국 금융소비자보호법에 정통한 AI 법률 전문가입니다. 
    아래에 제공된 [관련 법률 조항]과 [핵심 원칙]을 바탕으로 사용자의 질문에 답변하세요.

    **답변 생성 규칙:**
    1.  **생각의 흐름(Chain-of-Thought):** 단계별로 논리적인 추론 과정을 반드시 보여주세요.
    2.  **근거 제시:** 각 추론 단계마다 어떤 법률 조항을 근거로 판단했는지 명확하게 인용하세요. (예: '금소법 제21조 부당권유행위 금지에 따라...')
    3.  **종합 결론:** 모든 추론을 종합하여 사용자의 질문에 대한 명확하고 이해하기 쉬운 결론을 내리세요.
    4.  **정보 부족 시:** 만약 제공된 정보만으로 답변이 불가능하다면, '제공된 법률 정보만으로는 명확한 판단이 어렵습니다'라고 솔직하게 답변하세요. 추측하여 답변하지 마세요.

    ---
    [관련 법률 조항]
    {normal_context}
    ---
    [핵심 원칙 및 관련 규정]
    {step_back_context}
    ---
    """),
    ("user", "사용자 질문: {question}"),
    ("ai", "답변:\n\n**1. 문제 분석 및 쟁점 정리**\n\n"), # CoT 시작점 유도
])

if __name__ == '__main__':
    # 모델 및 리트리버 초기화
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    retriever = load_retriever()
    
    # 체인 구성
    question_gen_chain = create_step_back_chain(llm)
    
    # 전체 RAG 파이프라인 정의
    full_rag_chain = (
        {
            "normal_context": RunnableLambda(lambda x: x['question']) | retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
            "step_back_context": question_gen_chain | retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
            "question": RunnablePassthrough(),
        }

| FINAL_RESPONSE_PROMPT
| llm
| StrOutputParser()
    )
    
    # 실행
    user_question = "은행에서 신용대출을 받으려고 하는데, 자꾸만 월 10만원짜리 펀드 상품을 같이 가입해야 대출이 가능하다고 합니다. 이거 법적으로 문제 없는 건가요?"
    
    print("--- 질문 ---")
    print(user_question)
    print("\n--- AI 법률 전문가 답변 (RAG + CoT + Step-back) ---")
    
    # 스트리밍 출력을 위해 stream 사용
    for chunk in full_rag_chain.stream({"question": user_question}):
        print(chunk, end="", flush=True)