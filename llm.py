#!/usr/bin/env python3
# 0. Python Standard Library
import os
import re
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

# 1. PyTorch
import torch
import torch.nn.functional as F

# 2. FastAPI & Pydantic
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# 3. LangChain Core
try:
    from langchain.schema import Document
except Exception:
    from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 4. Document Loaders & Text Splitters
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 5. Vector Store
from langchain_chroma import Chroma

# 6. Retriever
from langchain_community.retrievers import BM25Retriever

ENSEMBLE_AVAILABLE = True
try:
    from langchain.retrievers import EnsembleRetriever
except Exception:
    try:
        from langchain_classic.retrievers import EnsembleRetriever
    except Exception:
        ENSEMBLE_AVAILABLE = False
        EnsembleRetriever = None

# 7. LLM
from langchain_ollama import ChatOllama

# 8. HuggingFace Transformers
from transformers import AutoConfig, AutoTokenizer, AutoModel


# =============================
# 1. Paths
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "data_storage")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")


# =============================
# 2. FastAPI Lifespan
# =============================
@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(DB_DIR, exist_ok=True)
    print(">> Server Started")
    yield
    print(">> Server Shutdown")


app = FastAPI(
    title="Tajikistan RAG API (Arctic Embed v2 + Gemma2)",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))


# =============================
# 3. Device Selection
# =============================
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f">> Embedding device: {DEVICE}")


# =============================
# 4. Arctic Embeddings (Custom)
# =============================
class ArcticEmbedEmbeddings(Embeddings):
    def __init__(
        self,
        model_name: str = "Snowflake/snowflake-arctic-embed-m-v2.0",
        device: torch.device = DEVICE,
        batch_size: int = 8,
        max_length: int = 512,
    ):
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        if getattr(config, "use_memory_efficient_attention", None) is not None:
            config.use_memory_efficient_attention = False
        if getattr(config, "attn_implementation", None) is not None:
            config.attn_implementation = "eager"

        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            config=config,
            torch_dtype=torch.float32,
            attn_implementation="eager",
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
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        pooled = self._mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
        pooled = F.normalize(pooled, p=2, dim=1)

        return pooled.cpu().tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_batch(texts[i : i + self.batch_size]))
        return vectors

    def embed_query(self, text: str) -> List[float]:
        return self._embed_batch([text])[0]


# =============================
# 5. LLM / Embedding Init
# =============================
llm = ChatOllama(
    model="gemma2:2b",
    temperature=0.2,
)

embedding_model = ArcticEmbedEmbeddings()


# =============================
# 6. Prompt Template
# =============================
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
"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)


# =============================
# 7. Vector Store
# =============================
def get_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )


# =============================
# 8. API Models
# =============================
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


# =============================
# 9. Utils
# =============================
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
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        return splitter.split_documents(docs)


def is_russian(text: str) -> bool:
    return any("\u0400" <= c <= "\u04FF" for c in text)


# =============================
# 10. Ingest API
# =============================
@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    doc_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{file.filename}")

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

    for i, d in enumerate(chunks):
        d.metadata.update(
            {
                "doc_id": doc_id,
                "source": file.filename,
                "chunk_index": i,
                "page": d.metadata.get("page"),
            }
        )

    vectordb = get_vectorstore()
    vectordb.add_documents(chunks)

    return {
        "message": f"Ingested {len(chunks)} chunks",
        "doc_id": doc_id,
    }


# =============================
# 11. Chat API
# =============================
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    start = time.time()
    vectordb = get_vectorstore()

    vector_retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    raw = vectordb._collection.get(include=["documents", "metadatas"])
    bm25_docs = [
        Document(page_content=t, metadata=m or {})
        for t, m in zip(raw.get("documents", []), raw.get("metadatas", []))
        if t
    ]

    retriever = vector_retriever
    if bm25_docs:
        bm25 = BM25Retriever.from_documents(bm25_docs)
        bm25.k = 3
        if ENSEMBLE_AVAILABLE:
            retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25],
                weights=[0.7, 0.3],
            )

    docs = retriever.invoke(req.question)

    if not docs:
        answer = (
            "У меня нет информации об этом в моих документах."
            if is_russian(req.question)
            else "I don't have information about that in my documents."
        )
        return ChatResponse(answer=answer, time_taken=time.time() - start, sources=[])

    context = "\n\n---\n\n".join(d.page_content for d in docs)
    sources = [
        SourceInfo(
            file=d.metadata.get("source", "unknown"),
            page=d.metadata.get("page"),
            snippet=d.page_content[:300],
        )
        for d in docs
    ]

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": req.question})

    return ChatResponse(
        answer=answer,
        time_taken=time.time() - start,
        sources=sources,
    )


# =============================
# 12. List Documents API
# =============================
@app.get("/documents")
async def list_documents():
    vectordb = get_vectorstore()
    data = vectordb._collection.get(include=["metadatas"])

    documents: Dict[str, str] = {}
    for meta in data.get("metadatas", []):
        if meta and "doc_id" in meta and "source" in meta:
            documents[meta["doc_id"]] = meta["source"]

    return {"documents": documents}


# =============================
# 13. Delete Document API
# =============================
@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    vectordb = get_vectorstore()
    data = vectordb._collection.get(where={"doc_id": doc_id})
    ids = data.get("ids", [])

    if not ids:
        raise HTTPException(status_code=404, detail="Document not found")

    vectordb._collection.delete(ids=ids)

    return {"deleted_chunks": len(ids), "doc_id": doc_id}


# =============================
# 14. Clear All Documents API
# =============================
@app.delete("/documents")
async def clear_all_documents():
    """
    Clear vector DB + uploaded files.
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


# =============================
# 15. Run
# =============================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("llm:app", host="0.0.0.0", port=8000, reload=True)
