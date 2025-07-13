# test_embedding.py

import os
# ⛔️ 중요: 아래 export 명령어를 터미널에서 실행했다면, 코드에서도 한 번 더 설정해줍니다.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("테스트 시작: HuggingFaceEmbeddings 라이브러리를 로드합니다.")
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings

    print("라이브러리 로드 성공. 이제 모델을 메모리에 올립니다...")

    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'}
    )

    print("모델 로드 성공. 테스트 문장을 임베딩합니다...")

    test_sentence = "이것은 임베딩 기능의 정상 동작을 확인하기 위한 테스트 문장입니다."
    vector = embeddings.embed_query(test_sentence)

    print("\n✅ 테스트 성공! 임베딩이 정상적으로 완료되었습니다.")
    print(f"생성된 벡터의 일부: {vector[:5]}")

except Exception as e:
    print(f"\n❌ 테스트 실패: 오류가 발생했습니다.")
    print(e)