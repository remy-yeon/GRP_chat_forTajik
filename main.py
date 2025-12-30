# =====================================================
# Tajikistan RAG API – Ver 1.0
# Scope: Attractions / Culture-Etiquette / Emergency
# =====================================================

# 0. Python Standard Library
import os
import re
import shutil
import time
import uuid
from typing import List, Optional, Dict
from contextlib import asynccontextmanager

# 1. PyTorch
import torch
import torch.nn.functional as F

# 2. FastAPI & Pydantic
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

# 3. LangChain Core
from langchain.schema import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 4. Loaders & Splitters
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 5. Vector Store
from langchain_chroma import Chroma

# 6. Retriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# 7. LLM
from langchain_ollama import ChatOllama

# 8. Embedding Model
from transformers import AutoTokenizer, AutoModel


# =========================
# Path 설정
# =========================
UPLOAD_DIR = "data_storage"
DB_DIR = "chroma_db"


# =========================
# FastAPI Lifespan
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(DB_DIR, exist_ok=True)
    print(">> Server Started")
    yield
    print(">> Server Shutdown")


app = FastAPI(
    title="Tajikistan RAG API (Ver 1.0)",
    lifespan=lifespan
)


# =========================
# Device 선택
# =========================
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f">> Embedding device: {DEVICE}")


# =========================
# Arctic Embedding
# =========================
class ArcticEmbedEmbeddings(Embeddings):
    def __init__(
        self,
        model_name: str = "Snowflake/snowflake-arctic-embed-l",
        device: torch.device = DEVICE,
        batch_size: int = 8,
        max_length: int = 512,
    ):
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32
        ).to(self.device)
        self.model.eval()

    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    @torch.no_grad()
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        pooled = self._mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
        pooled = F.normalize(pooled, p=2, dim=1)
        return pooled.cpu().tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = []
        for i in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_batch(texts[i:i + self.batch_size]))
        return vectors

    def embed_query(self, text: str) -> List[float]:
        return self._embed_batch([text])[0]


# =========================
# LLM / Embedding Init
# =========================
llm = ChatOllama(model="gemma2:2b", temperature=0.2)
embedding_model = ArcticEmbedEmbeddings()


# =========================
# Prompt (Ver 1.0)
# =========================
PROMPT_TEMPLATE = """
You are a travel assistant using retrieved documents.

Rules:
- Base your answer ONLY on the information in the Context.
- You MAY combine multiple related facts from the Context
  to provide a clearer explanation.
- Do NOT add facts that are not present in the Context.
- Do NOT speculate.

Write 3–5 sentences in a natural explanatory tone.
Avoid general tourism or promotional language.
Use factual descriptions grounded in the Context.
Use neutral, report-style language and avoid promotional or descriptive adjectives not explicitly stated in the Context.
Do not state direct outcomes or impacts unless they are explicitly mentioned in the Context.
Use cautious expressions such as "can", "is considered", or "has potential".
If the question is about how to reach a place,
describe the route and travel conditions ONLY if they are mentioned in the context

[Context]:
{context}

[Question]:
{question}

[Answer]:
"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)


# =========================
# Vector Store
# =========================
def get_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )


# =========================
# API Models
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
# Utils
# =========================
def preprocess_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_semantic_then_fallback(docs: List[Document]) -> List[Document]:
    try:
        return SemanticChunker(
            embedding=embedding_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=90,
        ).split_documents(docs)
    except Exception:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        return splitter.split_documents(docs)


def is_russian(text: str) -> bool:
    return any("\u0400" <= c <= "\u04FF" for c in text)


def is_garbled(text: str) -> bool:
    return any(ord(c) < 32 and c not in "\n\t" for c in text)


def is_low_value(text: str) -> bool:
    t = text.lower()
    return len(t) < 50 or "thank you" in t or "www." in t or "@" in t


# =========================
# Category Detection (Ver 1.0)
# =========================
def detect_question_category(q: str) -> Optional[str]:
    ql = q.lower()

    emergency = ["embassy", "police", "hospital", "emergency", "посоль", "полици", "больниц"]
    culture = ["culture", "etiquette", "custom", "religion", "этикет", "обыча", "религ"]
    attraction = ["visit", "see", "attraction", "city", "mountain", "lake", "посет", "достопримеч"]

    if any(k in ql for k in emergency):
        return "emergency"
    if any(k in ql for k in culture):
        return "culture"
    if any(k in ql for k in attraction):
        return "attraction"
    return None


def infer_doc_category(filename: str, text: str) -> Optional[str]:
    f = filename.lower()
    t = text.lower()
    if "embassy" in f or "emergency" in f or "police" in t:
        return "emergency"
    if "culture" in f or "etiquette" in t or "custom" in t:
        return "culture"
    if "tour" in f or "attraction" in t or "visit" in t:
        return "attraction"
    return None


# =========================
# Ingest API
# =========================
@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    doc_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{file.filename}")

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    if file.filename.lower().endswith(".pdf"):
        loader = PyMuPDFLoader(save_path)
    elif file.filename.lower().endswith(".txt"):
        loader = TextLoader(save_path, encoding="utf-8")
    else:
        raise HTTPException(status_code=400, detail="Only pdf or txt supported")

    docs = loader.load()
    for d in docs:
        d.page_content = preprocess_text(d.page_content)

    chunks = split_semantic_then_fallback(docs)
    sample = " ".join(c.page_content[:200] for c in chunks[:2])
    category = infer_doc_category(file.filename, sample)

    for i, d in enumerate(chunks):
        d.metadata.update({
            "doc_id": doc_id,
            "source": file.filename,
            "chunk_index": i,
            "page": d.metadata.get("page"),
            "category": category
        })

    vectordb = get_vectorstore()
    vectordb.add_documents(chunks)

    return {"message": f"Ingested {len(chunks)} chunks", "doc_id": doc_id}


# =========================
# Chat API (Ver 1.0)
# =========================
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    start = time.time()
    vectordb = get_vectorstore()

    language = "Russian" if is_russian(req.question) else "English"
    category = detect_question_category(req.question)

    # if category is None:
    #     msg = (
    #         "Эта функция не поддерживается в версии 1.0."
    #         if language == "Russian"
    #         else "This question is not supported in version 1.0."
    #     )
    #     return ChatResponse(answer=msg, time_taken=time.time() - start, sources=[])

    search_kwargs = {"k": 3}
    if category:
        search_kwargs["filter"] = {"category": category}

    vector_retriever = vectordb.as_retriever(
        search_kwargs=search_kwargs
    )

    raw = vectordb._collection.get(include=["documents", "metadatas"])
    bm25_docs = [
        Document(page_content=t, metadata=m)
        for t, m in zip(raw["documents"], raw["metadatas"])
        if (category is None or m.get("category") == category)
    ]

    retriever = vector_retriever
    if bm25_docs:
        bm25 = BM25Retriever.from_documents(bm25_docs)
        bm25.k = 3
        retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25],
            weights=[0.7, 0.3]
        )

    docs = [
        d for d in retriever.invoke(req.question)
        if not is_garbled(d.page_content) and not is_low_value(d.page_content)
    ]

    if not docs:
        msg = (
            "В предоставленных документах нет информации."
            if language == "Russian"
            else "The provided documents do not contain this information."
        )
        return ChatResponse(answer=msg, time_taken=time.time() - start, sources=[])

    context = "\n\n---\n\n".join(d.page_content for d in docs)
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "question": req.question,
        "language": language
    })

    sources = [
        SourceInfo(
            file=d.metadata.get("source", "unknown"),
            page=d.metadata.get("page"),
            snippet=d.page_content[:300]
        )
        for d in docs
    ]

    return ChatResponse(answer=answer, time_taken=time.time() - start, sources=sources)

@app.get("/documents")
async def list_documents():
    vectordb = get_vectorstore()
    data = vectordb._collection.get(include=["metadatas"])

    documents: Dict[str, Dict[str, str]] = {}

    for meta in data.get("metadatas", []):
        if not meta:
            continue
        doc_id = meta.get("doc_id")
        source = meta.get("source")
        category = meta.get("category")
        if doc_id and source:
            documents[doc_id] = {
                "source": source,
                "category": category
            }

    return {"documents": documents}

@app.delete("/documents")
async def clear_all_documents():
    """
    Vector DB 전체 초기화 + 업로드 파일 삭제
    """
    try:
        if os.path.exists(DB_DIR):
            shutil.rmtree(DB_DIR)
        os.makedirs(DB_DIR, exist_ok=True)

        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        return {"message": "All documents cleared (DB + uploads)."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =========================
# Run
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
