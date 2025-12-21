# =============================
# 설치 / 실행 방법 (로컬)
# =============================
# 1) 패키지 설치
#    pip install fastapi uvicorn langchain langchain-community chromadb sentence-transformers pymupdf
#    pip install langchain-experimental kiwipiepy
#    pip install rank-bm25
#
# 2) 가상환경 활성화
#    source .venv/bin/activate
#
# 3) Ollama 서버 실행 (LLM 호출용)
#    ollama serve
#
# 4) 모델 다운로드 (최초 1회)
#    ollama pull gemma2:2b
#
# 5) FastAPI 서버 실행
#    uvicorn main:app --reload
#
# 6) Swagger UI
#    http://127.0.0.1:8000/docs


from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

import os
import re
import shutil
import uuid
import time
from typing import List, Dict

# -----------------------------
# LangChain / VectorStore
# -----------------------------
from langchain.schema import Document
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# -----------------------------
# (선택) 검색 품질 개선 도구
# -----------------------------
SEMANTIC_CHUNK_AVAILABLE = True
try:
    from langchain_experimental.text_splitter import SemanticChunker
except Exception:
    SEMANTIC_CHUNK_AVAILABLE = False

KIWI_AVAILABLE = True
try:
    from kiwipiepy import Kiwi
except Exception:
    KIWI_AVAILABLE = False

BM25_AVAILABLE = True
try:
    from langchain_community.retrievers import BM25Retriever
except Exception:
    BM25_AVAILABLE = False

ENSEMBLE_AVAILABLE = True
try:
    from langchain.retrievers import EnsembleRetriever
except Exception:
    ENSEMBLE_AVAILABLE = False


# =============================
# FastAPI App
# =============================
app = FastAPI(title="Ollama RAG Demo")


# =============================
# 경로 설정
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 벡터 DB가 디스크에 저장되는 위치
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")

# 업로드 파일 임시 저장 위치
TEMP_DIR = os.path.join(BASE_DIR, "temp_uploads")
os.makedirs(TEMP_DIR, exist_ok=True)


# =============================
# Embedding / LLM 설정
# =============================
# 문서와 질문을 벡터로 변환하는 임베딩 모델
embedding = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large-instruct",
    encode_kwargs={"normalize_embeddings": True},
)

# Ollama로 로컬 LLM 호출
llm = Ollama(model="gemma2:2b")


# =============================
# Chroma Vector DB 로딩
# =============================
def get_vectorstore() -> Chroma:
    """
    - chroma_db 폴더가 있으면 기존 DB를 그대로 사용
    - 서버 재시작해도 문서가 유지됨
    """
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding,
    )


# =============================
# 요청 바디 모델
# =============================
class Question(BaseModel):
    question: str


# =============================
# Prompt Template
# =============================
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Using the context below, answer the user's question.
Use only the information from the context.
If the answer is not in the context, say:
"The document does not contain information about this topic."

Context:
{context}

Question:
{question}

Answer:
"""
)


# =============================
# 텍스트 전처리
# =============================
def preprocess_text(text: str) -> str:
    """
    - 문서에서 불필요한 공백 제거
    - 노이즈 단어 간단 정리
    """
    text = text.replace("ft", "처")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_documents(docs: List[Document]) -> List[Document]:
    """
    - loader가 읽어온 모든 문서에 전처리 적용
    """
    for d in docs:
        d.page_content = preprocess_text(d.page_content)
    return docs


# =============================
# 문서 청킹
# =============================
def split_documents_semantic_first(
    docs: List[Document],
    use_semantic: bool = True,
    chunk_size_fallback: int = 500,
    chunk_overlap_fallback: int = 50,
) -> List[Document]:
    """
    문서를 LLM이 처리 가능한 크기의 chunk로 분할
    1) SemanticChunker 가능하면 의미 단위 분할
    2) 아니면 일반 문자 기반 분할
    """
    if use_semantic and SEMANTIC_CHUNK_AVAILABLE:
        try:
            return SemanticChunker(embedding).split_documents(docs)
        except Exception:
            pass

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_fallback,
        chunk_overlap=chunk_overlap_fallback,
    )
    return splitter.split_documents(docs)


# =============================
# BM25 Retriever 생성
# =============================
def build_bm25_retriever_from_chroma(vectordb: Chroma, k: int = 3):
    """
    - Chroma에 저장된 문서를 기반으로 키워드 검색(BM25) 인덱스 생성
    - 벡터 검색의 약점을 보완하기 위한 용도
    """
    if not BM25_AVAILABLE:
        return None

    data = vectordb._collection.get(include=["documents", "metadatas"])
    docs = []

    for text, meta in zip(data.get("documents", []), data.get("metadatas", [])):
        if text:
            docs.append(Document(page_content=text, metadata=meta or {}))

    if not docs:
        return None

    if KIWI_AVAILABLE:
        kiwi = Kiwi()
        preprocess_func = lambda t: [x.form for x in kiwi.tokenize(t)]
    else:
        preprocess_func = lambda t: t.split()

    bm25 = BM25Retriever.from_documents(docs, preprocess_func=preprocess_func)
    bm25.k = k
    return bm25


# =============================
# 파일 업로드 API
# =============================
@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    """
    흐름:
    1) 파일 임시 저장
    2) PDF/TXT 로더로 문서 읽기
    3) 전처리
    4) 청킹
    5) Chroma DB에 벡터로 저장
    """

    temp_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}_{file.filename}")

    with open(temp_path, "wb") as f:
        f.write(await file.read())

    if file.filename.endswith(".pdf"):
        loader = PyMuPDFLoader(temp_path)
    elif file.filename.endswith(".txt"):
        loader = TextLoader(temp_path, encoding="utf-8")
    else:
        raise HTTPException(status_code=400, detail="Only pdf or txt allowed")

    docs = preprocess_documents(loader.load())
    chunks = split_documents_semantic_first(docs)

    doc_id = str(uuid.uuid4())
    for i, d in enumerate(chunks):
        d.metadata["doc_id"] = doc_id
        d.metadata["source"] = file.filename
        d.metadata["chunk_index"] = i

    vectordb = get_vectorstore()
    vectordb.add_documents(chunks)
    vectordb.persist()

    os.remove(temp_path)

    return {
        "status": "ok",
        "filename": file.filename,
        "chunks": len(chunks),
        "doc_id": doc_id,
    }


# =============================
# 질문 API (RAG) 
# =============================
@app.post("/ask")
async def ask_question(item: Question):
    """
    전체 RAG 흐름:
    질문 →
    (Vector 검색 + BM25 검색) →
    관련 문서 context 수집 →
    LLM 답변 생성
    """

    vectordb = get_vectorstore()

    # Vector 기반 검색 , 의미는 잘 잡지만 근거가 흐릴 수 있음
    vector_retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # BM25 기반 검색, 근거는 정확하지만 표현 변화에 약함
    bm25_retriever = build_bm25_retriever_from_chroma(vectordb, k=3)

    # 두 검색 결과를 앙상블로 결합
    retriever = vector_retriever
    if bm25_retriever and ENSEMBLE_AVAILABLE:
        retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.6, 0.4],
        )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    # 응답 시간 측정
    start = time.time()
    result = chain.invoke({"query": item.question})
    elapsed = round(time.time() - start, 3)

    return {
        "question": item.question,
        "answer": result["result"],
        "elapsed_time_sec": elapsed,
        "sources": [
            {
                "source": d.metadata.get("source"),
                "page": d.metadata.get("page", -1),
                "chunk_index": d.metadata.get("chunk_index"),
                "doc_id": d.metadata.get("doc_id"),
            }
            for d in result["source_documents"]
        ],
    }


# =============================
# 문서 목록 조회
# =============================
@app.get("/list_documents")
async def list_documents():
    """
    - DB에 들어있는 문서(doc_id 기준) 목록 확인
    """
    vectordb = get_vectorstore()
    data = vectordb._collection.get(include=["metadatas"])

    docs: Dict[str, str] = {}
    for m in data["metadatas"]:
        docs[m["doc_id"]] = m["source"]

    return {"documents": docs}


# =============================
# 문서 삭제
# =============================
@app.delete("/delete_document")
async def delete_document(doc_id: str):
    """
    - 특정 문서(doc_id)에 해당하는 모든 chunk 삭제
    """
    vectordb = get_vectorstore()
    data = vectordb._collection.get(where={"doc_id": doc_id})

    vectordb._collection.delete(data["ids"])
    return {"deleted_chunks": len(data["ids"])}


# =============================
# 전체 초기화
# =============================
@app.delete("/clear_all")
async def clear_all():
    """
    - chroma_db 폴더 삭제 → 모든 문서/임베딩 제거
    """
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
    return {"status": "ok"}
