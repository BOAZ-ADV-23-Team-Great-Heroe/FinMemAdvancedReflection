# puppy/legal_rag.py
# [참고] 이 파일은 이번 성능 개선 단계에서 수정이 필요하지 않습니다.
# agent.py에서 호출하는 방식만 개선되었습니다. 완전한 코드 제공을 위해 포함합니다.

import faiss
import pickle
import numpy as np
from typing import List, Dict, Any
from .embedding import OpenAILongerThanContextEmb
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
            self.documents: Dict[int, Document] = pickle.load(f)
        
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
        
        retrieved_texts: List[str] = []
        
        if ids.size > 0:
            for doc_id in ids[0]:
                if doc_id != -1 and doc_id in self.documents:
                    retrieved_texts.append(self.documents[doc_id].page_content)
                    
        return retrieved_texts