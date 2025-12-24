from __future__ import annotations

import os
import json
import time
import shutil
import asyncio
import hashlib
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Loaders
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader

# Docs / splitters
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Semantic chunker (experimental)
from langchain_experimental.text_splitter import SemanticChunker

# Vector stores
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS

# Retrievers
try:
    from langchain_community.retrievers import BM25Retriever
except Exception:
    from langchain.retrievers import BM25Retriever  # type: ignore

# Ollama (newer)
try:
    from langchain_ollama import ChatOllama, OllamaEmbeddings
except Exception:
    # fallback (older)
    from langchain_community.chat_models import ChatOllama  # type: ignore
    from langchain_community.embeddings import OllamaEmbeddings  # type: ignore

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# -----------------------------
# 1) Settings
# -----------------------------
BASE_DIR = Path(__file__).parent.resolve()

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(BASE_DIR / "data_storage")))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(BASE_DIR / "chroma_db")))
FAISS_DIR = Path(os.getenv("FAISS_DIR", str(BASE_DIR / "faiss_index")))
INDEX_DIR = Path(os.getenv("INDEX_DIR", str(BASE_DIR / "index")))
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"

CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "rag_docs")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "snowflake-arctic-embed2")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma2:2b")

# retrieval knobs
TOP_K_BM25 = int(os.getenv("TOP_K_BM25", "8"))
TOP_K_FAISS = int(os.getenv("TOP_K_FAISS", "8"))
TOP_K_ENSEMBLE = int(os.getenv("TOP_K_ENSEMBLE", "6"))
W_BM25 = float(os.getenv("W_BM25", "0.45"))
W_FAISS = float(os.getenv("W_FAISS", "0.55"))

# chunk knobs
COARSE_CHUNK_SIZE = int(os.getenv("COARSE_CHUNK_SIZE", "2000"))
COARSE_CHUNK_OVERLAP = int(os.getenv("COARSE_CHUNK_OVERLAP", "200"))
SEMANTIC_THRESHOLD_TYPE = os.getenv("SEMANTIC_THRESHOLD_TYPE", "percentile")

# context limit (Gemma 2B 대비)
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))


# -----------------------------
# 2) API Models
# -----------------------------
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


# -----------------------------
# 3) Helpers
# -----------------------------
def _ensure_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)


def _is_supported(filename: str) -> bool:
    fn = filename.lower()
    return fn.endswith(".pdf") or fn.endswith(".txt") or fn.endswith(".md")


def _load_file_as_docs(file_path: Path) -> List[Document]:
    lower = file_path.name.lower()
    if lower.endswith(".pdf"):
        loader = PyMuPDFLoader(str(file_path))
    elif lower.endswith(".txt") or lower.endswith(".md"):
        loader = TextLoader(str(file_path), encoding="utf-8")
    else:
        raise ValueError("Only .pdf, .txt, .md are supported.")

    docs = loader.load()
    for d in docs:
        md = dict(d.metadata or {})
        md["source"] = str(file_path)
        d.metadata = md
    return docs


def _semantic_chunk(docs: List[Document], embeddings: OllamaEmbeddings) -> List[Document]:
    coarse_splitter = RecursiveCharacterTextSplitter(
        chunk_size=COARSE_CHUNK_SIZE,
        chunk_overlap=COARSE_CHUNK_OVERLAP,
    )
    coarse_docs = coarse_splitter.split_documents(docs)

    semantic_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type=SEMANTIC_THRESHOLD_TYPE,
    )
    sem_docs = semantic_splitter.split_documents(coarse_docs)
    return [d for d in sem_docs if d.page_content and d.page_content.strip()]


def _append_chunks_jsonl(chunks: List[Document]) -> int:
    new_lines = 0
    with CHUNKS_PATH.open("a", encoding="utf-8") as f:
        for d in chunks:
            md = {}
            for k, v in (d.metadata or {}).items():
                try:
                    json.dumps(v)
                    md[k] = v
                except Exception:
                    md[k] = str(v)
            f.write(json.dumps({"page_content": d.page_content, "metadata": md}, ensure_ascii=False) + "\n")
            new_lines += 1
    return new_lines


def _load_all_chunks_jsonl() -> List[Document]:
    if not CHUNKS_PATH.exists():
        return []
    docs: List[Document] = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            docs.append(Document(page_content=obj["page_content"], metadata=obj.get("metadata") or {}))
    return docs


def _build_bm25_retriever(all_chunks: List[Document]) -> BM25Retriever:
    bm25 = BM25Retriever.from_documents(all_chunks)
    bm25.k = TOP_K_BM25
    return bm25


def _load_faiss_if_exists(embeddings: OllamaEmbeddings) -> Optional[FAISS]:
    if (FAISS_DIR / "index.faiss").exists() or (FAISS_DIR / "index.pkl").exists():
        return FAISS.load_local(
            str(FAISS_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    return None


def _save_faiss(vs: FAISS) -> None:
    vs.save_local(str(FAISS_DIR))


def _format_context(docs: List[Document], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    parts = []
    total = 0
    for d in docs:
        text = (d.page_content or "").strip()
        if not text:
            continue
        if total + len(text) > max_chars:
            break
        parts.append(text)
        total += len(text)
    return "\n\n---\n\n".join(parts)


def _sources_from_docs(docs: List[Document], snippet_len: int = 300) -> List[SourceInfo]:
    out: List[SourceInfo] = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        filename = os.path.basename(str(src))
        page = d.metadata.get("page", d.metadata.get("page_number", None))
        snippet = (d.page_content or "")[:snippet_len]
        out.append(SourceInfo(file=filename, page=page, snippet=snippet))
    return out


def _is_russian(text: str) -> bool:
    return any("\u0400" <= ch <= "\u04FF" for ch in text)


def _fallback_no_info(text: str) -> str:
    if _is_russian(text):
        return "У меня нет информации об этом в моих документах."
    return "I don't have information about that in my documents."


def _safe_invoke(retriever, query: str) -> List[Document]:
    # retriever.invoke 우선, 없으면 get_relevant_documents
    if hasattr(retriever, "invoke"):
        return retriever.invoke(query)
    if hasattr(retriever, "get_relevant_documents"):
        return retriever.get_relevant_documents(query)
    raise RuntimeError("Retriever has no invoke/get_relevant_documents method.")


def _doc_key(d: Document) -> str:
    src = str(d.metadata.get("source", ""))
    page = str(d.metadata.get("page", d.metadata.get("page_number", "")))
    content_hash = hashlib.md5((d.page_content or "").encode("utf-8", errors="ignore")).hexdigest()
    return f"{src}|{page}|{content_hash}"


def _rrf_fuse(
    a: List[Document],
    b: List[Document],
    w_a: float,
    w_b: float,
    rrf_k: int = 60,
    top_k: int = TOP_K_ENSEMBLE,
) -> List[Document]:
    """
    Weighted Reciprocal Rank Fusion
    score += weight / (rrf_k + rank)
    """
    scores = {}
    docs_map = {}

    for rank, d in enumerate(a, start=1):
        key = _doc_key(d)
        docs_map[key] = d
        scores[key] = scores.get(key, 0.0) + (w_a / (rrf_k + rank))

    for rank, d in enumerate(b, start=1):
        key = _doc_key(d)
        docs_map[key] = d
        scores[key] = scores.get(key, 0.0) + (w_b / (rrf_k + rank))

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [docs_map[k] for k, _ in ranked[:top_k]]


def _build_rag_chain(llm: ChatOllama):
    """English/Russian guide persona (NO outside knowledge)."""
    template = """
You are an AI assistant specialized in Tajikistan tourism.

You MUST follow these rules strictly:
- Use ONLY the information provided in [Context].
- Do NOT add any facts, names, numbers, or locations that are not explicitly stated in [Context].
- If [Context] is empty OR the answer cannot be clearly found in [Context], you MUST say:
  * English: "I don't have information about that in my documents."
  * Russian: "У меня нет информации об этом в моих документах."
- If the user asks in English, answer in English.
- If the user asks in Russian (Cyrillic), answer in Russian.
- Do NOT use Korean.
- Do NOT write sentences in Tajik (proper nouns from the documents are allowed).
- Keep your answer concise and helpful.

[Context]:
{context}

[User Question]:
{question}

[Answer]:
"""
    prompt = ChatPromptTemplate.from_template(template)
    return prompt | llm | StrOutputParser()


# -----------------------------
# 4) FastAPI app + lifespan
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_dirs()
    print(">> [System] Server Started. Storage Ready.")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.2)

    all_chunks = _load_all_chunks_jsonl()
    bm25 = _build_bm25_retriever(all_chunks) if all_chunks else None

    chroma = Chroma(
        collection_name=CHROMA_COLLECTION,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    faiss_vs = None
    try:
        faiss_vs = _load_faiss_if_exists(embeddings)
    except Exception as e:
        print(f">> [WARN] FAISS load failed: {e}. Will create on first ingest.")
        faiss_vs = None

    app.state.lock = asyncio.Lock()
    app.state.embeddings = embeddings
    app.state.llm = llm
    app.state.chain = _build_rag_chain(llm)

    app.state.chroma = chroma
    app.state.faiss_vs = faiss_vs
    app.state.bm25_docs = all_chunks
    app.state.bm25 = bm25

    yield
    print(">> [System] Server Shutdown.")


app = FastAPI(title="RAG Chatbot API (SemanticChunk + Chroma + BM25/FAISS RRF Fusion)", lifespan=lifespan)


# -----------------------------
# 5) Endpoints
# -----------------------------
@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/ingest", summary="Document Upload & Indexing")
async def ingest_document(file: UploadFile = File(...)):
    if not file.filename or not _is_supported(file.filename):
        raise HTTPException(status_code=400, detail="Only .pdf, .txt, .md files are supported.")

    file_path = UPLOAD_DIR / file.filename
    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    async with app.state.lock:
        try:
            docs = _load_file_as_docs(file_path)
            chunks = _semantic_chunk(docs, app.state.embeddings)

            if not chunks:
                raise HTTPException(status_code=400, detail="No chunks produced. Check the document content.")

            # Chroma store (optional for now)
            app.state.chroma.add_documents(chunks)
            if hasattr(app.state.chroma, "persist"):
                try:
                    app.state.chroma.persist()
                except Exception:
                    pass

            # BM25 corpus
            _append_chunks_jsonl(chunks)
            app.state.bm25_docs.extend(chunks)
            app.state.bm25 = _build_bm25_retriever(app.state.bm25_docs)

            # FAISS
            if app.state.faiss_vs is None:
                app.state.faiss_vs = FAISS.from_documents(chunks, app.state.embeddings)
            else:
                app.state.faiss_vs.add_documents(chunks)
            _save_faiss(app.state.faiss_vs)

            return {
                "message": f"Successfully ingested {len(chunks)} chunks from {file.filename}.",
                "chunks_added": len(chunks),
                "bm25_corpus_size": len(app.state.bm25_docs),
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse, summary="Ask the RAG Bot")
async def chat(req: ChatRequest):
    start = time.time()
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="question is required.")

    async with app.state.lock:
        try:
            bm25 = app.state.bm25
            faiss_vs = app.state.faiss_vs

            # no index yet
            if bm25 is None or faiss_vs is None:
                end = time.time()
                return ChatResponse(
                    answer=_fallback_no_info(q),
                    time_taken=end - start,
                    sources=[],
                )

            faiss_ret = faiss_vs.as_retriever(search_kwargs={"k": TOP_K_FAISS})

            bm25_docs = _safe_invoke(bm25, q)[:TOP_K_BM25]
            faiss_docs = _safe_invoke(faiss_ret, q)[:TOP_K_FAISS]

            docs = _rrf_fuse(bm25_docs, faiss_docs, W_BM25, W_FAISS, top_k=TOP_K_ENSEMBLE)

            if not docs:
                end = time.time()
                return ChatResponse(
                    answer=_fallback_no_info(q),
                    time_taken=end - start,
                    sources=[],
                )

            context = _format_context(docs)
            sources = _sources_from_docs(docs)

            answer = app.state.chain.invoke({"context": context, "question": q})
            end = time.time()

            return ChatResponse(
                answer=answer,
                time_taken=end - start,
                sources=sources,
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
