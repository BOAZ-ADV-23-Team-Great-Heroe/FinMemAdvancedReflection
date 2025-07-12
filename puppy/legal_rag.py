# puppy/legal_rag.py (최종 수정본)

import faiss
import pickle
import numpy as np
from typing import List, Dict, Any
from .embedding import OpenAILongerThanContextEmb # 👈 에이전트의 메인 임베딩 모델 재사용
from langchain.schema import Document

class LegalVectorDB:
    """
    미리 구축된 법률 FAISS 인덱스와 원문 데이터를 로드하여
    관련 법률 조항을 검색하는 역할을 담당하는 클래스.
    """
    def __init__(self, index_path: str, docs_path: str, emb_config: Dict[str, Any]):
        """
        법률 FAISS 인덱스와 원문 데이터를 로드합니다.

        Args:
            index_path (str): 미리 생성된 FAISS 인덱스 파일 경로.
            docs_path (str): 원본 문서 청크와 ID가 저장된 pickle 파일 경로.
            emb_config (dict): 에이전트가 사용하는 메인 임베딩 모델 설정.
        """
        print(f"Loading legal FAISS index from {index_path}...")
        self.index = faiss.read_index(index_path)
        
        print(f"Loading legal documents from {docs_path}...")
        with open(docs_path, 'rb') as f:
            # 👈 [개선] build_legal_db.py에서 저장한 Dict[int, Document] 형식으로 로드
            self.documents: Dict[int, Document] = pickle.load(f)
        
        # 👈 [수정] 벡터 DB 생성 시 사용된 모델과 동일한 임베딩 함수를 사용하도록 통일
        emb_only_config = {k: v for k, v in emb_config.items() if k in ['embedding_model', 'chunk_size', 'verbose', 'openai_api_key']}
        self.emb_func = OpenAILongerThanContextEmb(**emb_only_config)
        print("LegalVectorDB initialized successfully.")

    def query(self, query_text: str, top_k: int = 3) -> List[str]:
        """
        입력된 텍스트와 가장 유사한 법률 조항 텍스트를 검색하여 반환합니다.
        """
        if self.index.ntotal == 0:
            return []
            
        query_vector = self.emb_func(query_text)
        faiss.normalize_L2(query_vector)
        
        distances, ids = self.index.search(query_vector, top_k)
        
        retrieved_texts: List[str] = [] # 👈 [수정] 빈 리스트로 초기화
        
        # faiss는 결과로 2차원 numpy 배열을 반환하므로 첫 번째 원소를 사용
        if ids.size > 0:
            for doc_id in ids[0]:
                if doc_id != -1 and doc_id in self.documents:
                    # Document 객체의 page_content 속성에서 텍스트를 가져옴
                    retrieved_texts.append(self.documents[doc_id].page_content)
                    
        return retrieved_texts