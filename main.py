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
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
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
STATIC_DIR = "static"


# =========================
# 2) Retrieval 튜닝 파라미터
# =========================
# Dense 검색(MMR)
K_DENSE = 20  #8               # 최종 dense 반환 개수
FETCH_K = 40                   # MMR이 후보로 더 많이 뽑아 다양성 고려
LAMBDA_MULT = 0.35             # 1에 가까울수록 유사성, 0에 가까울수록 다양성

# Sparse 검색(BM25)
K_SPARSE = 20  #8

# Hybrid merge
# K_FINAL = 4               # LLM에 넣을 최종 문서 개수
K_FINAL = 8
RRF_K = 60                # RRF 안정 상수(크면 랭크 차이 완만)

# Context 길이 제한(너무 길면 속도/품질 흔들림 방지)
MAX_CONTEXT_CHARS = 6500

# History 길이 제한(프롬프트 과부하 방지)
MAX_HISTORY_CHARS = 2500
MAX_HISTORY_TURNS = 12


# =========================
# 3) Prompt
# =========================
PROMPT_TEMPLATE = """
You are a helpful travel assistant for tourists interested in visiting Tajikistan.

Use ONLY the information provided in [Context] for factual claims.

You may use [History] only to understand conversational references
(e.g., "that place", "the previous one", "what you said earlier"),
but do NOT introduce new facts from [History] that are not supported by [Context].

Language rules (STRICT):
- If the question is in Russian, answer in Russian.
- Otherwise, answer in English.
- Do not use any other language.

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
7. Write in clear, natural sentences suitable for a travel guide or tourism app.

[History]:
{history}

[Context]:
{context}

[Question]:
{question}

[Answer]:
""".strip()

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)


# =========================
# 3.1) Embassy Prompt (긴급/대사관 전용)
# =========================
EMBASSY_PROMPT_TEMPLATE = """
You are helping a traveler in an urgent situation in Tajikistan (e.g., lost or stolen passport).

Use ONLY the information provided in [Context] for factual claims.
You may use [History] only to resolve references, but never add new facts not supported by [Context].

Language rules (STRICT):
- If the question is in Russian, answer in Russian.
- Otherwise, answer in English.
- Do not use any other language.

Response requirements:
1. Provide the embassy/consulate contact information clearly and directly.
2. If the information exists in the context, you MUST include ALL of the following fields:
   - Embassy/Consulate name
   - City
   - Address/location
   - Phone number(s) (including emergency number if present)
3. If a phone number is present in the context, you MUST explicitly include it in the answer.
   Do NOT omit phone numbers under any circumstances.
4. If the context does not contain the specific country’s embassy info, explicitly say you do not have it in your documents.
5. Do NOT mention documents, pages, sources, or citations.


[History]:
{history}

[Context]:
{context}

[Question]:
{question}

[Answer]:
""".strip()

embassy_prompt = ChatPromptTemplate.from_template(EMBASSY_PROMPT_TEMPLATE)


# =========================
# 3.2) Hospital Prompt (추가)
# =========================
HOSPITAL_PROMPT_TEMPLATE = """
You are helping a traveler who needs hospital information in Tajikistan.

Use ONLY the information provided in [Context] for factual claims.
You may use [History] only to resolve references, but never add new facts not supported by [Context].

Language rules (STRICT):
- If the question is in Russian, answer in Russian.
- Otherwise, answer in English.
- Do not use any other language.

Response requirements:
1. Provide 1–2 hospitals in the requested city.
2. If available in the context, include:
   - Hospital name
   - City
   - Address/location
   - Phone number(s)
3. If the context does not contain hospitals for that city, say you do not have it in your documents.
4. Do NOT mention documents, pages, sources, or citations.
5. Do NOT give medical advice or diagnosis; only provide contact info.

[History]:
{history}

[Context]:
{context}

[Question]:
{question}

[Answer]:
""".strip()

hospital_prompt = ChatPromptTemplate.from_template(HOSPITAL_PROMPT_TEMPLATE)


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


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history: Optional[List[ChatMessage]] = None


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


def build_history(history: Optional[List[ChatMessage]]) -> str:
    """
    프론트에서 넘어온 history를 프롬프트에 넣기 좋게 문자열로 변환.
    - 너무 길면 최근 메시지 위주로 자름
    """
    if not history:
        return ""

    recent = history[-MAX_HISTORY_TURNS:]
    lines: List[str] = []
    for m in recent:
        role = (m.role or "").strip().lower()
        content = (m.content or "").strip()
        if not content:
            continue
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        else:
            lines.append(f"{role.capitalize() if role else 'Message'}: {content}")

    text = "\n".join(lines).strip()
    if len(text) <= MAX_HISTORY_CHARS:
        return text
    return text[-MAX_HISTORY_CHARS:]


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
# 7.1) Embassy / Emergency intent & nationality extraction (추가)
# =========================
def needs_embassy_help(q: str) -> bool:
    """
    여권 분실/도난/긴급 상황에서 대사관 안내가 필요한지 감지 (rule-based 1차)
    """
    ql = (q or "").lower()
    keywords = [
        "lost passport",
        "lost my passport",
        "passport lost",
        "stolen passport",
        "passport stolen",
        "my passport was stolen",
        "i lost my passport",
        "i have lost my passport",
        "i lost passport",
        "embassy",
        "consulate",
        "emergency",
        "urgent",
        "robbed",
        "theft",
        "stolen",
        "visa problem",
        "need help",
        "lost documents",
        "lost my id",
    ]
    if any(k in ql for k in keywords):
        return True

    # 러시아어 키워드(간단 버전)
    ru_keywords = [
        "потерял паспорт",
        "потеряла паспорт",
        "украли паспорт",
        "посольство",
        "консульство",
        "срочно",
        "экстренно",
    ]
    if any(k in ql for k in ru_keywords):
        return True

    return False


def extract_nationality(text: str) -> Optional[str]:
    """
    사용자가 말한 국적(또는 국가)을 명시적 표현에서만 추출.
    예:
    - "I am American"
    - "I'm Korean"
    - "My nationality is German"
    - "Citizen of France"

    ⚠️ 단순 문장("i lost my passport")을
    국적으로 오인하지 않도록 단답 fallback 제거
    """
    if not text:
        return None

    t = text.strip()

    patterns = [
        r"\bI am\s+([A-Za-z][A-Za-z \-]{1,40})\b",
        r"\bI'm\s+([A-Za-z][A-Za-z \-]{1,40})\b",
        r"\bI’m\s+([A-Za-z][A-Za-z \-]{1,40})\b",
        r"\bmy nationality is\s+([A-Za-z][A-Za-z \-]{1,40})\b",
        r"\bmy country is\s+([A-Za-z][A-Za-z \-]{1,40})\b",
        r"\bnationality:\s*([A-Za-z][A-Za-z \-]{1,40})\b",
        r"\bcitizen of\s+([A-Za-z][A-Za-z \-]{1,40})\b",
    ]

    for p in patterns:
        m = re.search(p, t, flags=re.IGNORECASE)
        if m:
            cand = m.group(1).strip(" .,!?:;\"'")
            if cand:
                return cand

    return None



def was_nationality_requested(history: Optional[List[ChatMessage]]) -> bool:
    """
    history만 보고 '직전에 국적을 물어본 상태인지' 판단
    - 세션 저장 안 하고, stateless로 구현하기 위해 사용
    """
    if not history:
        return False
    # 최근 assistant 메시지 중 nationality 요청이 있었는지 확인
    recent = history[-MAX_HISTORY_TURNS:]
    for m in reversed(recent):
        if (m.role or "").strip().lower() != "assistant":
            continue
        c = (m.content or "").lower()
        if "nationality" in c or "what is your nationality" in c or "tell me your nationality" in c:
            return True
        # 러시아어
        if "гражданство" in c or "какое у вас гражданство" in c:
            return True
        # 국적 문의를 만나면 거기서 멈춤(그 이후 더 과거는 안 봐도 됨)
        break
    return False


def find_nationality_from_history(history: Optional[List[ChatMessage]]) -> Optional[str]:
    """
    history에서 user가 말한 국적을 찾아봄 (최근 user 발화 우선)
    """
    if not history:
        return None
    recent = history[-MAX_HISTORY_TURNS:]
    for m in reversed(recent):
        if (m.role or "").strip().lower() != "user":
            continue
        nat = extract_nationality(m.content or "")
        if nat:
            return nat
    return None


# =========================
# 7.2) Hospital intent & city extraction (추가)
# =========================
def needs_hospital_help(q: str) -> bool:
    """
    병원/진료/아픔/의료 도움 질문 감지 (rule-based 1차)
    """
    ql = (q or "").lower()
    keywords = [
        "hospital",
        "clinic",
        "doctor",
        "medical",
        "i am sick",
        "i'm sick",
        "i feel sick",
        "i am ill",
        "i'm ill",
        "fever",
        "pain",
        "injury",
        "injured",
        "need a doctor",
        "need hospital",
        "where is a hospital",
        "where can i see a doctor",
        "emergency room",
        "er",
    ]
    if any(k in ql for k in keywords):
        return True

    ru_keywords = [
        "больница",
        "клиника",
        "врач",
        "мне плохо",
        "я болен",
        "температура",
        "боль",
        "травма",
        "скорая",
    ]
    if any(k in ql for k in ru_keywords):
        return True

    return False


def extract_city(text: str) -> Optional[str]:
    """
    도시를 명시적으로 말한 경우만 추출.
    예:
    - "I am in Dushanbe"
    - "I'm in Khujand"
    - "in Dushanbe"
    - "at Dushanbe"
    """
    if not text:
        return None
    t = text.strip()

    patterns = [
        r"\bI am in\s+([A-Za-z][A-Za-z \-]{1,40})\b",
        r"\bI'm in\s+([A-Za-z][A-Za-z \-]{1,40})\b",
        r"\bI’m in\s+([A-Za-z][A-Za-z \-]{1,40})\b",
        r"\bin\s+([A-Za-z][A-Za-z \-]{1,40})\b",
        r"\bat\s+([A-Za-z][A-Za-z \-]{1,40})\b",
        r"\bcity:\s*([A-Za-z][A-Za-z \-]{1,40})\b",
    ]
    for p in patterns:
        m = re.search(p, t, flags=re.IGNORECASE)
        if m:
            cand = m.group(1).strip(" .,!?:;\"'")
            if cand:
                return cand
    return None


def was_city_requested(history: Optional[List[ChatMessage]]) -> bool:
    """
    직전에 '도시를 물어본 상태인지' 판단 (stateless)
    """
    if not history:
        return False
    recent = history[-MAX_HISTORY_TURNS:]
    for m in reversed(recent):
        if (m.role or "").strip().lower() != "assistant":
            continue
        c = (m.content or "").lower()
        if "which city" in c or "what city" in c or "city are you in" in c or "currently in" in c:
            return True
        if "в каком городе" in c:
            return True
        break
    return False


def find_city_from_history(history: Optional[List[ChatMessage]]) -> Optional[str]:
    """
    history에서 user가 말한 도시를 찾아봄 (최근 user 발화 우선)
    """
    if not history:
        return None
    recent = history[-MAX_HISTORY_TURNS:]
    for m in reversed(recent):
        if (m.role or "").strip().lower() != "user":
            continue
        city = extract_city(m.content or "")
        if city:
            return city
    return None


def extract_city_fallback_if_awaiting(history: Optional[List[ChatMessage]], text: str) -> Optional[str]:
    """
    직전에 도시를 물어본 상태라면,
    user가 'Dushanbe' 처럼 단답으로만 도시를 말했을 때도 도시로 인정.
    """
    if not text:
        return None
    if not was_city_requested(history):
        return None

    t = text.strip()
    # 너무 긴 문장/숫자/특수문자 많은 경우 제외
    if len(t) > 40:
        return None
    if any(ch.isdigit() for ch in t):
        return None
    # 영문/공백/하이픈만 허용
    if not re.fullmatch(r"[A-Za-z \-]+", t):
        return None
    return t.strip(" -")


# =========================
# 8) FastAPI Lifespan (캐시/락 준비)
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(DB_DIR, exist_ok=True)
    os.makedirs(STATIC_DIR, exist_ok=True)

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
# Frontend (Static)
# =========================
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


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
# 10) Chat API (Hybrid RAG + Embassy flow)
# =========================
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    start = time.time()
    vectordb: Chroma = app.state.vectordb
    bm25: Optional[BM25Retriever] = app.state.bm25

    history_text = build_history(req.history)

    # -------------------------
    # Hospital flow (추가)
    # - "아파요/병원" 류 질문이면 도시 먼저 확인 후 병원 1~2개 제공
    # - Embassy와 충돌 방지: 여권/대사관 키워드가 명확하면 embassy 우선
    # -------------------------
    hospital_intent = needs_hospital_help(req.question)
    if hospital_intent:
        ql = (req.question or "").lower()
        passport_like = any(k in ql for k in ["passport", "embassy", "consulate", "visa"])

        if not passport_like:
            city_in_question = extract_city(req.question)
            city_in_history = find_city_from_history(req.history)
            city_fallback = extract_city_fallback_if_awaiting(req.history, req.question)

            city = city_in_question or city_in_history or city_fallback

            if city is None:
                if is_russian(req.question):
                    answer = (
                        "Понимаю. Чтобы подсказать больницу, скажите, пожалуйста, в каком городе вы находитесь "
                        "(например: Dushanbe, Khujand, Bokhtar)."
                    )
                else:
                    answer = (
                        "I understand. To suggest a hospital, which city are you currently in "
                        "(e.g., Dushanbe, Khujand, Bokhtar)?"
                    )
                return ChatResponse(answer=answer, time_taken=time.time() - start, sources=[])

            # 병원은 팩트형(연락처/주소)
            dense_search_type = "similarity"
            dense_kwargs = {"k": K_DENSE}
            dense_weight, sparse_weight = 0.3, 0.7
            rerank = True

            # hospital.txt 레코드가 "City: xxx", "Type: Hospital" 형태라서 이 쿼리가 잘 맞음
            if is_russian(req.question):
                search_query = f"Type: Hospital City: {city} телефон адрес"
            else:
                search_query = f"Type: Hospital City: {city} phone address"

            dense_retriever = vectordb.as_retriever(
                search_type=dense_search_type,
                search_kwargs=dense_kwargs,
            )
            dense_docs = dense_retriever.invoke(search_query)

            sparse_docs: List[Document] = []
            if bm25 is not None:
                bm25.k = K_SPARSE
                sparse_docs = bm25.invoke(search_query)

            final_candidates = rrf_merge(
                dense_docs,
                sparse_docs,
                k_final=20,
                rrf_k=RRF_K,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
            )
            if rerank:
                final_docs = rerank_by_embedding(search_query, final_candidates, top_k=K_FINAL)
            else:
                final_docs = final_candidates[:K_FINAL]

            # 도시 병원만 강하게 걸러내기(가끔 다른 도시가 섞여 들어올 때 방지)
            # (컨텍스트가 "City: X"로 명시되는 txt 구조에 최적)
            city_filtered = []
            for d in final_docs:
                txt = (d.page_content or "").lower()
                if f"city: {city.lower()}" in txt:
                    city_filtered.append(d)
            if city_filtered:
                final_docs = city_filtered

            # 1~2개만
            final_docs = final_docs[:2]

            if not final_docs:
                answer = (
                    f"У меня нет информации о больницах в городе {city} в моих документах."
                    if is_russian(req.question)
                    else f"I don’t have hospital information for {city} in my documents."
                )
                return ChatResponse(answer=answer, time_taken=time.time() - start, sources=[])

            context = build_context(final_docs, max_chars=MAX_CONTEXT_CHARS)

            sources = [
                SourceInfo(
                    file=(d.metadata or {}).get("source", "unknown"),
                    page=(d.metadata or {}).get("page"),
                    snippet=d.page_content,
                )
                for d in final_docs
            ]

            chain = hospital_prompt | llm | StrOutputParser()
            answer = chain.invoke({"context": context, "question": req.question, "history": history_text})

            return ChatResponse(answer=answer, time_taken=time.time() - start, sources=sources)

    # -------------------------
    # Embassy flow (추가)
    # -------------------------
    embassy_intent = needs_embassy_help(req.question)

    if embassy_intent:
        nat_in_question = extract_nationality(req.question)
        nat_in_history = find_nationality_from_history(req.history)

        nationality = nat_in_question or nat_in_history

        awaiting_nat = was_nationality_requested(req.history)

        if nationality is None:
            if is_russian(req.question):
                answer = (
                    "Понимаю — это срочная ситуация.\n"
                    "Чтобы подсказать правильное посольство/консульство в Таджикистане, "
                    "скажите, пожалуйста, ваше гражданство (например: USA, Korea, Germany)."
                )
            else:
                answer = (
                    "I’m sorry you’re dealing with this.\n"
                    "To point you to the correct embassy/consulate in Tajikistan, "
                    "what is your nationality (e.g., USA, Korea, Germany)?"
                )
            return ChatResponse(answer=answer, time_taken=time.time() - start, sources=[])

        fact = True
        dense_search_type = "similarity"
        dense_kwargs = {"k": K_DENSE}
        dense_weight, sparse_weight = 0.3, 0.7
        rerank = False

        if is_russian(req.question):
            search_query = f"посольство {nationality} Таджикистан Душанбе телефон адрес"
        else:
            search_query = f"embassy consulate {nationality} Tajikistan Dushanbe phone address"

        dense_retriever = vectordb.as_retriever(
            search_type=dense_search_type,
            search_kwargs=dense_kwargs,
        )
        dense_docs = dense_retriever.invoke(search_query)

        sparse_docs: List[Document] = []
        if bm25 is not None:
            bm25.k = K_SPARSE
            sparse_docs = bm25.invoke(search_query)

        final_candidates = rrf_merge(
            dense_docs,
            sparse_docs,
            k_final=20,
            rrf_k=RRF_K,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )
        if rerank:
            final_docs = rerank_by_embedding(search_query, final_candidates, top_k=K_FINAL)
        else:
            final_docs = final_candidates[:K_FINAL]

        if not final_docs:
            answer = (
                "У меня нет информации о посольстве этой страны в моих документах."
                if is_russian(req.question)
                else "I don’t have embassy/consulate information for that country in my documents."
            )
            return ChatResponse(answer=answer, time_taken=time.time() - start, sources=[])

        context = build_context(final_docs, max_chars=MAX_CONTEXT_CHARS)

        sources = [
            SourceInfo(
                file=(d.metadata or {}).get("source", "unknown"),
                page=(d.metadata or {}).get("page"),
                snippet=d.page_content,
            )
            for d in final_docs
        ]

        chain = embassy_prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": req.question, "history": history_text})

        return ChatResponse(answer=answer, time_taken=time.time() - start, sources=sources)

    # -------------------------
    # 기존 Travel RAG (그대로)
    # -------------------------
    fact = is_fact_question(req.question)

    if fact:
        dense_search_type = "similarity"
        dense_kwargs = {"k": K_DENSE}
        dense_weight, sparse_weight = 0.3, 0.7
        rerank = True
    else:
        dense_search_type = "mmr"
        dense_kwargs = {
            "k": K_DENSE,
            "fetch_k": FETCH_K,
            "lambda_mult": 0.7,
        }
        dense_weight, sparse_weight = 0.6, 0.4
        rerank = True

    dense_retriever = vectordb.as_retriever(
        search_type=dense_search_type,
        search_kwargs=dense_kwargs,
    )
    dense_docs = dense_retriever.invoke(req.question)

    sparse_docs: List[Document] = []
    if bm25 is not None:
        bm25.k = K_SPARSE
        sparse_docs = bm25.invoke(req.question)

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
            snippet=d.page_content,
        )
        for d in final_docs
    ]

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": req.question, "history": history_text})

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
