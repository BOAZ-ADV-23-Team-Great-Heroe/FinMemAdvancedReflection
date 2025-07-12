# puppy/build_legal_db.py

"""
법률 문서(txt, pdf)를 로드하여 '구조적 청킹(structural chunking)'을 수행하고,
HuggingFace 임베딩 모델을 사용해 FAISS 벡터 데이터베이스를 생성 및 저장합니다.

실행 방법:
python -m puppy.build_legal_db
"""

import os
import re
import pickle
from typing import List

import faiss
import numpy as np
from tqdm import tqdm

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from .embedding import OpenAILongerThanContextEmb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv(override=True)

# --- 1. 설정 (Configuration) ---
DATA_PATH = './data/legal_documents/'
DB_FAISS_PATH = "./vector_db/"
EMBEDDING_MODEL = "text-embedding-ada-002"

def load_documents(path: str) -> list[Document]:
    """지정된 경로에서 텍스트 및 PDF 문서를 로드합니다."""
    print("문서 로딩 시작...")
    all_docs = []
    
    # 지정된 경로에 디렉토리가 없으면 오류 메시지와 함께 빈 리스트 반환
    if not os.path.isdir(path):
        print(f"오류: '{path}' 디렉토리를 찾을 수 없습니다. 경로를 확인해주세요.")
        return all_docs

    for filename in tqdm(os.listdir(path), desc="문서 로딩 중"):
        full_path = os.path.join(path, filename)
        try:
            if filename.endswith('.pdf'):
                loader = PyPDFLoader(full_path)
            elif filename.endswith('.txt'):
                loader = TextLoader(full_path, encoding='utf-8')
            else:
                continue # 지원하지 않는 파일 형식은 건너뜁니다.
            
            loaded_docs = loader.load()
            # 각 문서에 파일명을 메타데이터로 추가
            for doc in loaded_docs:
                doc.metadata['source'] = filename
            all_docs.extend(loaded_docs)
        except Exception as e:
            print(f"'{filename}' 파일 로딩 중 오류 발생: {e}")
            
    print(f"총 {len(all_docs)}개의 페이지/파일 로드 완료.")
    return all_docs

def clean_text(text: str) -> str:
    """법률 텍스트에서 불필요한 공백, 페이지 번호 등을 정제합니다."""
    text = re.sub(r'-\s*\d+\s*-', '', text)  # 페이지 번호 형식 (- 1 -) 제거
    text = re.sub(r'\s{3,}', '\n\n', text).strip()  # 과도한 공백은 문단으로 변환
    return text

def chunk_legal_documents(documents: list[Document]) -> list[Document]:
    """법률 문서에 최적화된 '구조적 청킹'을 수행합니다."""
    print("구조적 청킹 시작...")
    
    # 법률 조항 구조에 기반한 분리자 설정 (조/항/호)
    separators = [
        r'\n\n제[0-9]+조(?:의[0-9]+)?\s*\(.+?\)',  # "제1조(목적)" 형태
        r'\n\n제[0-9]+조(?:의[0-9]+)?',             # "제2조" 형태
        "\n\n", "\n", " ", ""
    ]
    
    # 모든 문서를 하나의 텍스트로 결합
    full_text = "\n\n".join([clean_text(doc.page_content) for doc in documents])
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=1000,
        chunk_overlap=150,
        is_separator_regex=True,
        length_function=len,
    )
    
    chunks = text_splitter.split_text(full_text)
    
    doc_chunks = []
    for i, chunk_text in enumerate(tqdm(chunks, desc="청크 생성 중")):
        # 각 청크의 시작 부분에서 '제N조' 형태의 조항 번호를 추출하여 메타데이터로 활용
        match = re.search(r'^(제[0-9]+조(?:의[0-9]+)?)', chunk_text)
        article = match.group(1).strip() if match else f"기타 조항 Chunk {i+1}"
        
        # 검색 시 활용할 수 있도록 고유 ID와 조항 정보를 메타데이터에 포함
        metadata = {"source": "종합 법률 문서", "article": article, "id": i}
        doc_chunks.append(Document(page_content=chunk_text, metadata=metadata))
        
    print(f"총 {len(doc_chunks)}개의 법률 조항 청크 생성 완료.")
    return doc_chunks

# ⛔️ 기존 함수를 지우고, 아래의 새로운 함수 코드로 완전히 교체해주세요. ⛔️

def create_and_save_vector_db(chunks: list[Document]):
    """임베딩 모델을 사용하여 벡터 DB를 생성하고 파일로 저장합니다. (메모리 최적화 버전)"""
    print("임베딩 및 벡터 DB 생성 시작...")

    embeddings_model = OpenAILongerThanContextEmb(
        embedding_model=EMBEDDING_MODEL,
        chunk_size=1000,
        verbose=False
    )

    # --- 👇 여기가 핵심적인 수정 부분입니다! ---

    all_vectors = []
    batch_size = 8  # 한 번에 처리할 청크 개수 (PC 사양에 따라 조절 가능)

    print(f"총 {len(chunks)}개의 청크를 {batch_size}개씩 나누어 임베딩합니다.")

    # tqdm을 사용하여 배치 처리 진행 상황을 시각적으로 표시
    for i in tqdm(range(0, len(chunks), batch_size), desc="문서 임베딩 중"):
        batch_chunks = chunks[i:i + batch_size]
        batch_texts = [doc.page_content for doc in batch_chunks]

        # 실제 임베딩 처리
        batch_vectors = []
        for text in batch_texts:
            vector = embeddings_model(text)
            if hasattr(vector, 'shape') and len(vector.shape) == 2:
                vector = vector.flatten()
            batch_vectors.append(vector)
        all_vectors.extend(batch_vectors)

    print("모든 문서의 임베딩이 완료되었습니다.")
    # --- 👆 여기까지가 핵심적인 수정 부분입니다! ---

    embedding_dim = len(all_vectors[0])
    index = faiss.IndexFlatIP(embedding_dim)
    index = faiss.IndexIDMap2(index)

    ids = np.array([doc.metadata['id'] for doc in chunks])
    index.add_with_ids(np.array(all_vectors, dtype=np.float32), ids)

    os.makedirs(DB_FAISS_PATH, exist_ok=True)

    faiss_index_path = os.path.join(DB_FAISS_PATH, "legal_faiss.index")
    faiss.write_index(index, faiss_index_path)
    print(f"FAISS 인덱스가 '{faiss_index_path}'에 저장되었습니다.")

    doc_map = {doc.metadata['id']: doc for doc in chunks}
    docs_pkl_path = os.path.join(DB_FAISS_PATH, "legal_docs.pkl")
    with open(docs_pkl_path, "wb") as f:
        pickle.dump(doc_map, f)

    print(f"문서 데이터(ID-Chunk Map)가 '{docs_pkl_path}'에 저장되었습니다.")
    print("✅ 벡터 DB 생성이 성공적으로 완료되었습니다.")

if __name__ == '__main__':
    # 1. 문서 로드
    raw_documents = load_documents(DATA_PATH)
    
    # 2. 문서가 성공적으로 로드된 경우에만 다음 단계 진행
    if raw_documents:
        # 3. 법률 문서 청킹
        legal_chunks = chunk_legal_documents(raw_documents)
        # 4. 벡터 DB 생성 및 저장
        create_and_save_vector_db(legal_chunks)
    else:
        print("❌ 로드할 문서가 없습니다. 스크립트를 종료합니다.")