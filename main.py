"""
main.py - Single-file RAG API (Ollama Embeddings + Chroma + Hybrid Retriever)
Retrieval-first + Prompt-selection (NO intent state machine)

Stack:
- Chunking: SemanticChunker (fallback: RecursiveCharacterTextSplitter)
- Vector DB: Chroma (persist_directory)
- Embedding: OllamaEmbeddings("snowflake-arctic-embed2")
- LLM: ChatOllama("gemma2:2b")
- Hybrid Retrieval: Dense(similarity/MMR) + Sparse(BM25) merged by RRF
- Rerank: cosine(embedding)
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
from typing import List, Optional, Dict
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

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


# =========================
# 1) Path 설정
# =========================
UPLOAD_DIR = "data_storage"
DB_DIR = "chroma_db"
STATIC_DIR = "static"


# =========================
# 2) Retrieval 튜닝 파라미터
# =========================
# Dense
K_DENSE = 20
FETCH_K = 40
LAMBDA_MULT = 0.35

# Sparse (BM25)
K_SPARSE = 20

# Hybrid merge / final
K_FINAL = 8
RRF_K = 60

# Context 제한
MAX_CONTEXT_CHARS = 6500


# =========================
# 3) Prompts
# =========================
TRAVEL_PROMPT_TEMPLATE = """
You are a helpful travel assistant for tourists interested in visiting Tajikistan.

Use ONLY the information provided in [Context] for factual claims.
If the answer is not in the context, say you don't have that information.

Language rules (STRICT):
- If the question is in Russian, answer in Russian.
- Otherwise, answer in English.
- Do not use any other language.

Guidelines:
1. Answer as if you are helping a traveler understand the destination,
   not as if you are analyzing or describing a document.
2. Do not mention documents, reports, figures, pages, or sources explicitly.
3. Avoid generic or textbook-style explanations.
4. Focus on practical, concrete information useful to travelers.
5. Do not infer or add information not clearly supported by the context.
6. Write in clear, natural sentences suitable for a travel guide / tourism app.

[Context]:
{context}

[Question]:
{question}

[Answer]:
""".strip()


HOSPITAL_PROMPT_TEMPLATE = """
You are helping a traveler who may need medical assistance in Tajikistan.

Use ONLY the information provided in [Context].
If hospital names, addresses, or phone numbers are present, present them clearly.
Do NOT provide medical diagnosis or treatment advice.
Do NOT mention documents or sources.

Language rules (STRICT):
- If the question is in Russian, answer in Russian.
- Otherwise, answer in English.
- Do not use any other language.

Output style:
- Prefer 2–4 short sentences.
- If multiple hospitals appear, list up to 2 hospitals with:
  name, city, address, phone (if available).
- If phone is N/A, say it is not available.

[Context]:
{context}

[Question]:
{question}

[Answer]:
""".strip()


EMBASSY_PROMPT_TEMPLATE = """
You are a traveler assistance bot.

CRITICAL OUTPUT RULE:
- You MUST answer ONLY in {answer_language}.
- Do NOT use any other language.
- Even if the context is written in a different language, translate it into {answer_language}.

STRICT RULES:
- Use ONLY the information in [Context].
- Mention ONLY ONE embassy.
- Include the embassy name, city, address, and phone number.
- Do NOT add or guess information.
- Plain text only.

[Context]:
{context}

[Answer]:
""".strip()




travel_prompt = ChatPromptTemplate.from_template(TRAVEL_PROMPT_TEMPLATE)
hospital_prompt = hospital_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "STRICT RULES (violation = incorrect answer):\n"
     "- Answer ONLY in {answer_language}.\n"
     "- If Russian, use Cyrillic letters only.\n"
     "- Output MUST be plain text.\n"
     "- DO NOT use lists, bullets, asterisks, or markdown.\n"
     "- Mention NO MORE THAN TWO hospitals.\n"
     "- Use ONLY the context.\n"
     "- Include: name, city, address, phone (if available).\n"
     "- Do NOT add introductions like 'there are several hospitals'.\n"
     "- Do NOT mention documents, files, or sources.\n"
    ),
    ("human",
     "[Context]\n{context}\n\n"
     "[Question]\n{question}\n\n"
     "Answer following the rules exactly.\n"
     "[Answer]\n")
])

embassy_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You MUST follow these rules:\n"
     "- Answer ONLY in {answer_language}.\n"
     "- If Russian, use Cyrillic letters.\n"
     "- Plain text only. NO markdown, NO bullets.\n"
     "- Use ONLY the context.\n"
     "- Mention ONLY ONE embassy.\n"
     "- Include name, city, address, phone.\n"
    ),
    ("human",
     "[Context]\n{context}\n\n"
     "[Question]\n{question}\n\n"
     "[Answer]\n")
])


# =========================
# 4) LLM / Embedding (Ollama)
# =========================
llm = ChatOllama(
    model="gemma2:2b",
    temperature=0.2,
    num_predict=220,
    timeout=25,
)
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
def split_record_txt(text: str, source: str) -> List[Document]:
    records = []
    pattern = r"=== (HOSPITAL|EMBASSY)_RECORD_START ===(.*?)=== \1_RECORD_END ==="
    matches = re.findall(pattern, text, flags=re.DOTALL)

    for i, (rtype, body) in enumerate(matches):
        meta = {
            "source": source,
            "record_type": rtype.lower(),  # hospital | embassy
            "chunk_index": i,
        }
        records.append(
            Document(
                page_content=preprocess_text(body),
                metadata=meta,
            )
        )
    return records

def preprocess_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def is_russian(text: str) -> bool:
    return any("\u0400" <= c <= "\u04FF" for c in (text or ""))


def sanitize_metadata(meta: dict) -> dict:
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
    parts: List[str] = []
    total = 0
    for d in docs:
        text = (d.page_content or "").strip()
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
    m = d.metadata or {}
    if "doc_id" in m and "chunk_index" in m and "source" in m:
        return f'{m["doc_id"]}:{m["chunk_index"]}:{m["source"]}'
    return (m.get("source", "unknown") + ":" + str(hash(d.page_content)))


def rrf_merge(
    dense_docs: List[Document],
    sparse_docs: List[Document],
    k_final: int,
    rrf_k: int,
    dense_weight: float,
    sparse_weight: float,
) -> List[Document]:
    scores: Dict[str, float] = {}
    by_key: Dict[str, Document] = {}

    def add(docs: List[Document], weight: float):
        for rank, d in enumerate(docs, start=1):
            key = doc_key(d)
            by_key[key] = d
            scores[key] = scores.get(key, 0.0) + weight * (1.0 / (rrf_k + rank))

    add(dense_docs, dense_weight)
    add(sparse_docs, sparse_weight)

    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [by_key[k] for k, _ in merged[:k_final]]


async def rebuild_bm25(app: FastAPI) -> None:
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
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-12)


def rerank_by_embedding(query: str, docs: List[Document], top_k: int) -> List[Document]:
    if not docs:
        return []
    q = embedding_model.embed_query(query)
    doc_vecs = embedding_model.embed_documents([(d.page_content or "") for d in docs])
    scored = [(cosine(q, v), d) for v, d in zip(doc_vecs, docs)]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_k]]


def is_fact_question(q: str) -> bool:
    q = q or ""
    ql = q.lower()

    if any(ch.isdigit() for ch in q):
        return True

    patterns = [
        r"\bhow many\b",
        r"\bhow high\b",
        r"\bwhat(?:'s| is)\b",
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


def select_prompt(question: str, docs: List[Document]):
    ql = (question or "").lower()
    types = { (d.metadata or {}).get("record_type") for d in docs }

    has_hospital = "hospital" in types
    has_embassy = "embassy" in types

    q_hospital_hint = any(k in ql for k in ["hospital", "clinic", "doctor", "medical", "sick", "fever", "injury"])
    q_embassy_hint = any(k in ql for k in ["passport", "visa", "embassy", "consulate", "stolen", "lost passport"])

    if has_hospital and q_hospital_hint:
        return hospital_prompt
    if has_embassy and q_embassy_hint:
        return embassy_prompt
    if has_hospital and not has_embassy:
        return hospital_prompt
    if has_embassy and not has_hospital:
        return embassy_prompt

    return travel_prompt


# =========================
# 8) FastAPI Lifespan
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(DB_DIR, exist_ok=True)

    app.state.vectordb = get_vectorstore()
    app.state.bm25 = None

    app.state.write_lock = asyncio.Lock()
    app.state.rebuild_lock = asyncio.Lock()

    async with app.state.rebuild_lock:
        await rebuild_bm25(app)

    print(">> Server Started")
    yield
    print(">> Server Shutdown")


app = FastAPI(
    title="Tajikistan RAG API (Retrieval-first Prompt Selection)",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    doc_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{file.filename}")

    async with app.state.write_lock:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        filename_lower = (file.filename or "").lower()

        # ======================
        # 1️⃣ PDF
        # ======================
        if filename_lower.endswith(".pdf"):
            loader = PyMuPDFLoader(save_path)
            docs = loader.load()

            for d in docs:
                d.page_content = preprocess_text(d.page_content)

            chunks = split_semantic_then_fallback(docs)

        # ======================
        # 2️⃣ TXT
        # ======================
        elif filename_lower.endswith(".txt"):
            raw_text = TextLoader(save_path, encoding="utf-8").load()[0].page_content
            raw_text = preprocess_text(raw_text)

            # ✅ record 기반 txt
            if "HOSPITAL_RECORD_START" in raw_text or "EMBASSY_RECORD_START" in raw_text:
                chunks = split_record_txt(raw_text, file.filename)
            else:
                docs = [Document(page_content=raw_text, metadata={"source": file.filename})]
                chunks = split_semantic_then_fallback(docs)

        else:
            raise HTTPException(status_code=400, detail="Only pdf or txt supported")

        if not chunks:
            raise HTTPException(status_code=400, detail="Chunking produced no chunks")

        # ======================
        # 3️⃣ metadata 공통 처리
        # ======================
        for i, d in enumerate(chunks):
            d.metadata = d.metadata or {}
            d.metadata.update(
                {
                    "doc_id": doc_id,
                    "source": file.filename,
                    "chunk_index": i,
                }
            )

        chunks = filter_complex_metadata(chunks)
        for d in chunks:
            d.metadata = sanitize_metadata(d.metadata)

        vectordb: Chroma = app.state.vectordb
        vectordb.add_documents(chunks)

    async with app.state.rebuild_lock:
        await rebuild_bm25(app)

    return {"message": f"Ingested {len(chunks)} chunks", "doc_id": doc_id}


# =========================
# 10) Chat API (Hybrid RAG + Prompt Selection)
# =========================
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    start = time.time()
    vectordb: Chroma = app.state.vectordb
    bm25: Optional[BM25Retriever] = app.state.bm25

    q = (req.question or "").strip()
    if not q:
        return ChatResponse(
            answer="Please ask a question.",
            time_taken=time.time() - start,
            sources=[]
        )

    # ✅ 출력 언어를 코드 레벨에서 강제
    answer_language = "Russian" if is_russian(q) else "English"

    fact = is_fact_question(q)

    if fact:
        dense_search_type = "similarity"
        dense_kwargs = {"k": K_DENSE}
        dense_weight, sparse_weight = 0.3, 0.7
        rerank = True
    else:
        dense_search_type = "mmr"
        dense_kwargs = {"k": K_DENSE, "fetch_k": FETCH_K, "lambda_mult": 0.7}
        dense_weight, sparse_weight = 0.6, 0.4
        rerank = True

    dense_docs = vectordb.as_retriever(
        search_type=dense_search_type,
        search_kwargs=dense_kwargs,
    ).invoke(q)

    sparse_docs: List[Document] = []
    if bm25 is not None:
        bm25.k = K_SPARSE
        sparse_docs = bm25.invoke(q)

    final_candidates = rrf_merge(
        dense_docs=dense_docs,
        sparse_docs=sparse_docs,
        k_final=20,
        rrf_k=RRF_K,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
    )

    final_docs = (
        rerank_by_embedding(q, final_candidates, top_k=K_FINAL)
        if rerank
        else final_candidates[:K_FINAL]
    )

    if not final_docs:
        answer = (
            "У меня нет информации об этом."
            if answer_language == "Russian"
            else "I don't have information about that."
        )
        return ChatResponse(
            answer=answer,
            time_taken=time.time() - start,
            sources=[]
        )

    context = build_context(final_docs, max_chars=MAX_CONTEXT_CHARS)

    # ✅ retrieval 결과(metadata) 기반 prompt 선택
    chosen_prompt = select_prompt(q, final_docs)

    if chosen_prompt == hospital_prompt:
        final_docs = [
            d for d in final_docs
            if (d.metadata or {}).get("record_type") == "hospital"
        ]

    # ✅ 언어 강제 변수를 프롬프트에 전달
    chain = chosen_prompt | llm | StrOutputParser()
    answer = (
        chain.invoke({
            "context": context,
            "question": q,
            "answer_language": answer_language,
        }) or ""
    ).strip()

    if answer_language == "Russian":
        answer = (
            ChatPromptTemplate.from_messages([
                ("system",
                "Translate the following text into natural Russian.\n"
                "Use Cyrillic letters only.\n"
                "Plain text only. No markdown."
                ),
                ("human", "{text}")
            ])
            | llm
            | StrOutputParser()
        ).invoke({"text": answer}).strip()
        answer = re.sub(r"^[\*\-\•]\s*", "", answer, flags=re.MULTILINE)

    sources = [
        SourceInfo(
            file=(d.metadata or {}).get("source", "unknown"),
            page=(d.metadata or {}).get("page"),
            snippet=(d.page_content or "")[:300],
        )
        for d in final_docs
    ]

    return ChatResponse(
        answer=answer,
        time_taken=time.time() - start,
        sources=sources,
    )


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
