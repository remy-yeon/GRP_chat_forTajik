"""
main.py - Single-file RAG API (Ollama Embeddings + Chroma + Hybrid Retriever)

Stack:
- Chunking: SemanticChunker (fallback: RecursiveCharacterTextSplitter)
- Vector DB: Chroma (persist_directory)
- Embedding: OllamaEmbeddings("snowflake-arctic-embed2")
- LLM: ChatOllama("gemma2:2b")
- Hybrid Retrieval: Dense(MMR) + Sparse(BM25) merged by RRF
- Performance: VectorDB + BM25 cached in app.state, BM25 rebuilt only on ingest/delete/clear
"""

from __future__ import annotations

import os
import re
import shutil
import time
import uuid
import asyncio
import math
from typing import List, Optional, Dict, Tuple
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata


# =========================
# 1) Path 설정
# =========================
UPLOAD_DIR = "data_storage"
DB_DIR = "chroma_db"


# =========================
# 2) Retrieval 튜닝 파라미터
# =========================
# Dense 검색(MMR)
K_DENSE = 20 #8               # 최종 dense 반환 개수
FETCH_K = 40              # MMR이 후보로 더 많이 뽑아 다양성 고려
LAMBDA_MULT = 0.35        # 1에 가까울수록 유사성, 0에 가까울수록 다양성

# Sparse 검색(BM25)
K_SPARSE = 20 #8

# Hybrid merge
# K_FINAL = 4               # LLM에 넣을 최종 문서 개수
K_FINAL = 8
RRF_K = 60                # RRF 안정 상수(크면 랭크 차이 완만)

# Context 길이 제한(너무 길면 속도/품질 흔들림 방지)
MAX_CONTEXT_CHARS = 6500


# =========================
# 3) Prompt
# =========================
PROMPT_TEMPLATE = """
You are a helpful travel assistant for tourists interested in visiting Tajikistan.

Use ONLY the information provided in [Context].

Guidelines:
1. Answer as if you are helping a traveler understand the destination,
   not as if you are analyzing or describing a document.
2. Do not mention documents, reports, figures, pages, or sources explicitly.
3. Avoid generic or textbook-style explanations.
4. Focus on practical, concrete information that would be useful to travelers,
   such as real examples, regions, activities, projects, or situations
   mentioned in the context.
5. If the question asks about problems or challenges, explain them in a way
   that helps travelers understand what to expect.
6. Do not infer or add information that is not clearly supported by the context.
7. Answer in the same language as the question.
8. Write in clear, natural sentences suitable for a travel guide or tourism app.

[Context]:
{context}

[Question]:
{question}

[Answer]:
""".strip()

PROMPT_TEMPLATE_RU = """
Вы полезный туристический помощник для путешественников,
интересующихся Таджикистаном.

Используйте ТОЛЬКО информацию из [Context].

Правила:
1. Отвечайте так, как будто вы помогаете путешественнику,
   а не анализируете документ.
2. Не упоминайте документы, отчёты, страницы или источники.
3. Избегайте общих, учебниковых объяснений.
4. Сосредотачивайтесь на практической информации:
   регионы, маршруты, условия, примеры из контекста.
5. Если вопрос о трудностях или проблемах — объясняйте,
   чего ожидать туристу.
6. Не добавляйте информацию, которой нет в контексте.
7. Отвечайте ТОЛЬКО на русском языке.
8. Пишите естественным, разговорным русским языком.

[Context]:
{context}

[Question]:
{question}

[Answer]:
""".strip()



# =========================
# 4) LLM / Embedding (Ollama)
# =========================
# 필요하면 base_url="http://127.0.0.1:11434" 명시 가능
llm = ChatOllama(model="gemma2:2b", temperature=0.2)
embedding_model = OllamaEmbeddings(model="snowflake-arctic-embed2")


# =========================
# 5) Vector Store
# =========================
def get_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )


# =========================
# 6) API Models
# =========================
class SourceInfo(BaseModel):
    file: str
    page: Optional[int] = None
    snippet: str


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    time_taken: float
    sources: List[SourceInfo]


# =========================
# 7) Utils
# =========================
def preprocess_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def is_russian(text: str) -> bool:
    return any("\u0400" <= c <= "\u04FF" for c in text)


def sanitize_metadata(meta: dict) -> dict:
    """
    Chroma metadata는 str/int/float/bool만 허용.
    None/리스트/딕셔너리 등은 제거 또는 문자열화.
    """
    clean = {}
    for k, v in (meta or {}).items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean


def split_semantic_then_fallback(docs: List[Document]) -> List[Document]:
    """
    SemanticChunker는 embedding이 필요합니다.
    OllamaEmbeddings로도 동작하며, 실패 시 안전하게 char splitter로 fallback.
    """
    try:
        return SemanticChunker(
            embedding=embedding_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=90,
        ).split_documents(docs)
    except Exception:
        splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
        return splitter.split_documents(docs)


def build_context(docs: List[Document], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    parts = []
    total = 0
    for d in docs:
        text = d.page_content.strip()
        if not text:
            continue
        if total + len(text) > max_chars:
            remain = max_chars - total
            if remain > 200:
                parts.append(text[:remain])
            break
        parts.append(text)
        total += len(text)
    return "\n\n---\n\n".join(parts)


def doc_key(d: Document) -> str:
    """
    중복 제거 키: (doc_id, chunk_index, source) 우선, 없으면 내용 기반
    """
    m = d.metadata or {}
    if "doc_id" in m and "chunk_index" in m and "source" in m:
        return f'{m["doc_id"]}:{m["chunk_index"]}:{m["source"]}'
    return (m.get("source", "unknown") + ":" + str(hash(d.page_content)))


def rrf_merge(
    dense_docs: List[Document],
    sparse_docs: List[Document],
    k_final: int = K_FINAL,
    rrf_k: int = RRF_K,
    dense_weight: float = 0.3,
    sparse_weight: float = 0.7,
) -> List[Document]:
    """
    Reciprocal Rank Fusion: score = Σ 1/(rrf_k + rank)
    (rank는 1부터)
    """
    scores: Dict[str, float] = {}
    by_key: Dict[str, Document] = {}

    def add(docs: List[Document], weight: float):
        for rank, d in enumerate(docs, start=1):
            key = doc_key(d)
            by_key[key] = d
            scores[key] = scores.get(key, 0.0) + weight * (1.0 / (rrf_k + rank))

    # 가중치(원하면 조절): dense를 조금 더 믿는 편
    
#     추천 기본값(관광 챗봇 운영 관점)
# ✅ 기본(가장 무난)
# dense 0.45 / sparse 0.55
# ✅ 대화형·추천형이 더 많다(“어디 가야 해?”, “분위기 좋은 곳?”, “아이랑 가기 좋나?”)
# dense 0.6 / sparse 0.4
# ✅ 팩트 검증/정보성 QA가 더 많다(고도, 위치, 이름, 입장, 운영 등)
# dense 0.3~0.4 / sparse 0.6~0.7
    
    add(dense_docs, weight=dense_weight)
    add(sparse_docs, weight=sparse_weight)

    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [by_key[k] for k, _ in merged[:k_final]]


async def rebuild_bm25(app: FastAPI) -> None:
    """
    BM25는 매 요청마다 만들지 말고, 데이터 변경 때만 갱신.
    """
    vectordb: Chroma = app.state.vectordb
    raw = vectordb._collection.get(include=["documents", "metadatas"])
    docs = [
        Document(page_content=t, metadata=(m or {}))
        for t, m in zip(raw.get("documents", []), raw.get("metadatas", []))
        if t and t.strip()
    ]
    if docs:
        bm25 = BM25Retriever.from_documents(docs)
        bm25.k = K_SPARSE
        app.state.bm25 = bm25
    else:
        app.state.bm25 = None

def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    return dot / (na*nb + 1e-12)

def rerank_by_embedding(query: str, docs: List[Document], top_k: int) -> List[Document]:
    q = embedding_model.embed_query(query)
    doc_vecs = embedding_model.embed_documents([d.page_content for d in docs])
    scored = [(cosine(q, v), d) for v, d in zip(doc_vecs, docs)]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_k]]

def is_fact_question(q: str) -> bool:
    """
    팩트/정확 근거가 중요한 질문 감지:
    - 숫자/단위/연도
    - name/located/which/where/how many 등
    """
    ql = q.lower()

    # 숫자 포함(1911, 5000m 등)
    if any(ch.isdigit() for ch in q):
        return True

    patterns = [
        r"\bhow many\b",
        r"\bhow high\b",
        r"\bwhat(?:'s| is) the name\b",
        r"\bwhich\b",
        r"\bwhere\b",
        r"\blocated\b",
        r"\byear\b",
        r"\bcentury\b",
        r"\bmeters?\b",
        r"\bmetres?\b",
        r"\bkm\b",
        r"\baltitude\b",
        r"\bheight\b",
    ]
    return any(re.search(p, ql) for p in patterns)


# =========================
# 8) FastAPI Lifespan (캐시/락 준비)
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(DB_DIR, exist_ok=True)

    app.state.vectordb = get_vectorstore()
    app.state.bm25 = None

    # 쓰기/리빌드 보호용 락
    app.state.write_lock = asyncio.Lock()
    app.state.rebuild_lock = asyncio.Lock()

    # 초기 BM25 build(기존 DB 있을 때)
    async with app.state.rebuild_lock:
        await rebuild_bm25(app)

    print(">> Server Started")
    yield
    print(">> Server Shutdown")


app = FastAPI(
    title="Tajikistan RAG API (Ollama Embeddings + Gemma2)",
    lifespan=lifespan,
)


# =========================
# 9) Ingest API
# =========================
@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    doc_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{file.filename}")

    async with app.state.write_lock:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        filename_lower = file.filename.lower()
        if filename_lower.endswith(".pdf"):
            loader = PyMuPDFLoader(save_path)
        elif filename_lower.endswith(".txt"):
            loader = TextLoader(save_path, encoding="utf-8")
        else:
            raise HTTPException(status_code=400, detail="Only pdf or txt supported")

        docs = loader.load()
        if not docs:
            raise HTTPException(status_code=400, detail="No content extracted from file")

        for d in docs:
            d.page_content = preprocess_text(d.page_content)

        chunks = split_semantic_then_fallback(docs)
        if not chunks:
            raise HTTPException(status_code=400, detail="Chunking produced no chunks")

        # 메타데이터 정리(중요: Chroma는 None/복잡 타입 싫어함)
        for i, d in enumerate(chunks):
            d.metadata = d.metadata or {}
            d.metadata.update(
                {
                    "doc_id": doc_id,
                    "source": file.filename,
                    "chunk_index": i,
                }
            )

            # page는 없을 수 있으니 None이면 제거
            page = d.metadata.get("page")
            if page is None:
                d.metadata.pop("page", None)
            else:
                # 혹시 문자열인 경우도 있어 int로 고정
                try:
                    d.metadata["page"] = int(page)
                except Exception:
                    d.metadata.pop("page", None)

        # 1) LangChain helper로 복잡 메타 1차 정리
        chunks = filter_complex_metadata(chunks)

        # 2) 최종적으로 Chroma 허용 타입으로 강제
        for d in chunks:
            d.metadata = sanitize_metadata(d.metadata)

        vectordb: Chroma = app.state.vectordb
        vectordb.add_documents(chunks)

    # BM25는 데이터 변경 때만 rebuild
    async with app.state.rebuild_lock:
        await rebuild_bm25(app)

    return {"message": f"Ingested {len(chunks)} chunks", "doc_id": doc_id}


# =========================
# 10) Chat API (Hybrid RAG)
# =========================
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    start = time.time()
    vectordb: Chroma = app.state.vectordb
    bm25: Optional[BM25Retriever] = app.state.bm25

    # Dense: MMR (중복 덜 가져오고 다양성 확보)
    # dense_retriever = vectordb.as_retriever(
    #     search_type="mmr",
    #     search_kwargs={
    #         "k": K_DENSE,
    #         "fetch_k": FETCH_K,
    #         "lambda_mult": LAMBDA_MULT,
    #     },
    # )
    fact = is_fact_question(req.question)

    # 질문 타입별 전략 설정
    if fact:
        # 팩트형: 정확한 근거 우선
        dense_search_type = "similarity"
        dense_kwargs = {"k": K_DENSE}
        dense_weight, sparse_weight = 0.3, 0.7

        # (선택) rerank는 유지 추천 (근거 누락 줄이는 데 도움)
        rerank = True
    else:
        # 추천/설명형: 의미/다양성 우선
        dense_search_type = "mmr"
        dense_kwargs = {
            "k": K_DENSE,
            "fetch_k": FETCH_K,
            "lambda_mult": 0.7,  # 추천/설명형은 유사성 더 주는 편이 안정적
        }
        dense_weight, sparse_weight = 0.6, 0.4
        rerank = True  # 추천형도 rerank가 종종 도움됨(원치 않으면 False)

    dense_retriever = vectordb.as_retriever(
        search_type=dense_search_type,
        search_kwargs=dense_kwargs,
    )
    dense_docs = dense_retriever.invoke(req.question)


    # Sparse: BM25 (없으면 빈 리스트)
    sparse_docs: List[Document] = []
    if bm25 is not None:
        bm25.k = K_SPARSE
        sparse_docs = bm25.invoke(req.question)

    # Hybrid merge (RRF)
    # final_docs = rrf_merge(dense_docs, sparse_docs, k_final=K_FINAL, rrf_k=RRF_K)
    final_candidates = rrf_merge(
        dense_docs,
        sparse_docs,
        k_final=20,
        rrf_k=RRF_K,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
    )
    if rerank:
        final_docs = rerank_by_embedding(req.question, final_candidates, top_k=K_FINAL)
    else:
        final_docs = final_candidates[:K_FINAL]



    if not final_docs:
        answer = (
            "У меня нет информации об этом в моих документах."
            if is_russian(req.question)
            else "I don't have information about that in my documents."
        )
        return ChatResponse(answer=answer, time_taken=time.time() - start, sources=[])

    context = build_context(final_docs, max_chars=MAX_CONTEXT_CHARS)
    sources = [
        SourceInfo(
            file=(d.metadata or {}).get("source", "unknown"),
            page=(d.metadata or {}).get("page"),
            snippet=d.page_content[:300],
        )
        for d in final_docs
    ]
    # 🔑 질문 언어에 따라 프롬프트 선택
    if is_russian(req.question):
        selected_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_RU)
    else:
        selected_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    chain = selected_prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": req.question})

    return ChatResponse(answer=answer, time_taken=time.time() - start, sources=sources)


# =========================
# 11) List Documents API
# =========================
@app.get("/documents")
async def list_documents():
    vectordb: Chroma = app.state.vectordb
    data = vectordb._collection.get(include=["metadatas"])

    documents: Dict[str, str] = {}
    for meta in data.get("metadatas", []):
        if meta and "doc_id" in meta and "source" in meta:
            documents[meta["doc_id"]] = meta["source"]

    return {"documents": documents}


# =========================
# 12) Delete Document API
# =========================
@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    async with app.state.write_lock:
        vectordb: Chroma = app.state.vectordb
        data = vectordb._collection.get(where={"doc_id": doc_id})
        ids = data.get("ids", [])
        if not ids:
            raise HTTPException(status_code=404, detail="Document not found")

        vectordb._collection.delete(ids=ids)

    async with app.state.rebuild_lock:
        await rebuild_bm25(app)

    return {"deleted_chunks": len(ids), "doc_id": doc_id}


# =========================
# 13) Clear All Documents API
# =========================
@app.delete("/documents")
async def clear_all_documents():
    try:
        async with app.state.write_lock:
            if os.path.exists(DB_DIR):
                shutil.rmtree(DB_DIR)
            os.makedirs(DB_DIR, exist_ok=True)

            if os.path.exists(UPLOAD_DIR):
                shutil.rmtree(UPLOAD_DIR)
            os.makedirs(UPLOAD_DIR, exist_ok=True)

            # VectorDB 핸들 재생성
            app.state.vectordb = get_vectorstore()
            app.state.bm25 = None

        return {"message": "All documents cleared (DB + uploads)."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# 14) Run
# =========================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)