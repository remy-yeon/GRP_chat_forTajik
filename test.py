# 0. Python Standard Library (기본 내장 라이브러리)
import os            # 파일/디렉토리 경로 처리
import re            # 정규표현식 (텍스트 전처리)
import shutil        # 파일 복사/삭제
import time          # 실행 시간 측정
import uuid          # 문서 고유 ID 생성
from typing import List, Optional, Dict  # 타입 힌트
from contextlib import asynccontextmanager  # FastAPI lifespan 관리

# 1. PyTorch (모델 연산 / 임베딩 계산)
import torch
import torch.nn.functional as F

# 2. FastAPI & Pydantic (API 서버 / 요청·응답 모델)
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

# 3. LangChain Core (문서, 프롬프트, 출력 파싱)
from langchain.schema import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 4. LangChain Document Loaders & Text Splitters
from langchain_community.document_loaders import (
    PyMuPDFLoader,   # PDF 로더
    TextLoader       # TXT 로더
)

from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 5. Vector Store (ChromaDB)
from langchain_chroma import Chroma

# 6. Retriever (검색 로직)
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# 7. LLM (Ollama 기반 Gemma 2)
from langchain_ollama import ChatOllama

# 8. HuggingFace Transformers (Arctic Embedding 모델)
from transformers import AutoTokenizer, AutoModel



# 1. Path 설정

# 업로드된 원본 파일이 저장되는 디렉토리
UPLOAD_DIR = "data_storage"

# ChromaDB 벡터 데이터가 저장되는 디렉토리
DB_DIR = "chroma_db"


# 2. FastAPI Lifespan (서버 시작 / 종료 훅)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 서버 시작 시:
    - 필요한 디렉토리 생성
    서버 종료 시:
    - 별도 작업은 없음 (DB는 유지)
    """
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(DB_DIR, exist_ok=True)
    print(">> Server Started")
    yield
    print(">> Server Shutdown")


app = FastAPI(
    title="Tajikistan RAG API (Arctic Embed + Gemma2)",
    lifespan=lifespan
)


# 3. Device 자동 선택 (Mac / CUDA / CPU)
"""
- Mac: MPS (Apple Silicon GPU)
- NVIDIA: CUDA
- 그 외: CPU

임베딩/추론 속도에 직접적인 영향
"""
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f">> Embedding device: {DEVICE}")


# 4. Snowflake Arctic Embedding (Custom Embeddings)
class ArcticEmbedEmbeddings(Embeddings):
    # - Arctic은 LangChain 기본 임베딩으로 제공되지 않음
    # - 직접 embed_documents / embed_query 구현 필요

    #  임베딩 방식
    # - Transformer 출력 (last_hidden_state)
    # - Attention mask 기반 mean pooling
    # - L2 normalize → cosine similarity 전제

    def __init__( # 모델/토크나이저를 메모리에 올리는 준비 단계
        self,
        model_name: str = "Snowflake/snowflake-arctic-embed-l",
        device: torch.device = DEVICE,
        batch_size: int = 8, # 한 번에 몇 문장을 처리할지, 크면 빠르지만 메모리 많이 씀
        max_length: int = 512, # 문서 길이 제한, 너무 크면 느려짐
    ):

        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained( # 텍스트 → 토큰 ID로 바꾸는 애
            model_name,
            trust_remote_code=True
        )

        self.model = AutoModel.from_pretrained( # 토큰 → 벡터(숨겨진 표현) 만들어주는 애
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32  # MPS 안정화하기 위해서 명시
        ).to(self.device)

        self.model.eval() # 학습 모드가 아니라 추론 모드로 바꿔서 더 안정적으로/빠르게

    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask): # 토큰 벡터들을 문장 벡터 1개로 합치는 과정
        # 모델 출력 last_hidden_state는 보통 아래 형태
        # (B, T, H)

        # B: 배치 크기(문장 개수)
        # T: 토큰 개수 (max_length까지)
        # H: 임베딩 차원(예: 768 같은 값)

        # 즉 문장 하나가 토큰 여러 개로 쪼개져 있고, 토큰마다 벡터가 있음.
        # 그런데 벡터DB에는 “문장당 벡터 1개”가 필요하니까
        # 토큰 벡터들을 평균 내서 1개로 만드는 게 _mean_pool.

        # attention_mask:
        # - padding 토큰 제외,토크나이저는 길이를 맞추려고 padding(빈칸 토큰)을 넣는데 padding까지 평균 내면 의미가 깨짐.

        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    @torch.no_grad()
    def _embed_batch(self, texts: List[str]) -> List[List[float]]: # 실제 텍스트 → 벡터 계산 핵심
        """
        실제 임베딩 계산이 수행되는 내부 함수
        """
        inputs = self.tokenizer( # 문장을 토큰으로 쪼갬
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()} # GPU/CPU로 이동해서 계산할 준비함

        outputs = self.model(**inputs) # 모델에 넣어서 토큰 벡터 얻음(last_hidden_state)
        pooled = self._mean_pool(outputs.last_hidden_state, inputs["attention_mask"]) # 토큰 벡터 → 문장 벡터 1개
        pooled = F.normalize(pooled, p=2, dim=1) # 정규화 (L2 normalize), 벡터 길이를 1로 맞춰서 코사인 유사도 계산이 깔끔해짐

        return pooled.cpu().tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]: # 여러 문장을 배치로 쪼개서 처리
        # 문서 chunk가 수천개면 한 번에 다 넣으면 메모리 터짐
        # 그래서 batch_size만큼씩 끊어서 돌림

        """
        문서 임베딩 (Vector DB 저장용)
        """
        vectors = []
        for i in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_batch(texts[i:i + self.batch_size]))
        return vectors

    def embed_query(self, text: str) -> List[float]: # 질문은 1개니까 배치 1개로 처리
        """
        쿼리 임베딩 (검색용)
        """
        return self._embed_batch([text])[0]


# 5. LLM (Gemma 2:2B via Ollama)

llm = ChatOllama(
    model="gemma2:2b",
    temperature=0.2  # 낮을수록 사실 위주
)

embedding_model = ArcticEmbedEmbeddings()


# 6. Prompt Template
PROMPT_TEMPLATE = """
You are an AI assistant specialized in Tajikistan tourism.

Rules:
- Use ONLY the information in [Context].
- If the answer is not in the context, say:
  - English: "I don't have information about that in my documents."
  - Russian: "У меня нет информации об этом в моих документах."
- Answer in the same language as the question.
- Be concise and factual.

[Context]:
{context}

[Question]:
{question}

[Answer]:
"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)


# 7. Vector Store (ChromaDB)
def get_vectorstore():

    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )


# 8. API Models
class SourceInfo(BaseModel):
    file: str
    page: Optional[int]
    snippet: str


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    time_taken: float
    sources: List[SourceInfo]


# 9. Utils
def preprocess_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_semantic_then_fallback(docs: List[Document]) -> List[Document]:
    # 1. SemanticChunker
    #     - 의미 단위 분할
    #     - Embedding 기반

    # 2. 실패 시 RecursiveCharacterTextSplitter
    #     - 길이 기반 안전망
    try:
        return SemanticChunker(
            embedding=embedding_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=90,  # ↑ 키우면 chunk 커짐
        ).split_documents(docs)
    except Exception:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,     # ↑ 키우면 문맥 ↑
            chunk_overlap=50    # ↑ 키우면 recall ↑
        )
        return splitter.split_documents(docs)


def is_russian(text: str) -> bool:
    """
    질문 언어 판별 (러시아어 여부)
    """
    return any("\u0400" <= c <= "\u04FF" for c in text)


# 10. Ingest API
@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):

    # PDF / TXT 업로드 → 전처리 → Chunk → Vector DB 저장

    doc_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{file.filename}")

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    if file.filename.lower().endswith(".pdf"):
        loader = PyMuPDFLoader(save_path)
    elif file.filename.lower().endswith(".txt"):
        loader = TextLoader(save_path, encoding="utf-8")
    else:
        raise HTTPException(400, "Only pdf or txt supported")

    docs = loader.load()
    for d in docs:
        d.page_content = preprocess_text(d.page_content)

    chunks = split_semantic_then_fallback(docs)

    for i, d in enumerate(chunks):
        d.metadata.update({
            "doc_id": doc_id,
            "source": file.filename,
            "chunk_index": i,
            "page": d.metadata.get("page")
        })

    vectordb = get_vectorstore()
    vectordb.add_documents(chunks)
    vectordb.persist()

    return {
        "message": f"Ingested {len(chunks)} chunks",
        "doc_id": doc_id
    }


# 11. Chat API (RAG)
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    start = time.time()
    vectordb = get_vectorstore()

    # Dense Retrieval
    vector_retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # Sparse Retrieval (BM25)
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
        retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25],
            weights=[0.7, 0.3]  # Vector 우선
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
            snippet=d.page_content[:300]
        )
        for d in docs
    ]

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": req.question})

    return ChatResponse(
        answer=answer,
        time_taken=time.time() - start,
        sources=sources
    )

# 13. List Documents API
@app.get("/documents")
async def list_documents():
    """
    현재 Vector DB에 저장된 문서 목록 조회

    반환 형식:
    {
        "documents": {
            "doc_id_1": "filename1.pdf",
            "doc_id_2": "filename2.txt"
        }
    }
    """
    vectordb = get_vectorstore()

    data = vectordb._collection.get(include=["metadatas"])

    documents: Dict[str, str] = {}
    for meta in data.get("metadatas", []):
        if meta and "doc_id" in meta and "source" in meta:
            documents[meta["doc_id"]] = meta["source"]

    return {"documents": documents}


# 14. Delete Document API
@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """
    doc_id에 해당하는 모든 chunk 삭제
    """
    vectordb = get_vectorstore()

    data = vectordb._collection.get(where={"doc_id": doc_id})
    ids = data.get("ids", [])

    if not ids:
        raise HTTPException(status_code=404, detail="Document not found")

    vectordb._collection.delete(ids=ids)
    vectordb.persist()

    return {
        "deleted_chunks": len(ids),
        "doc_id": doc_id
    }


# 12. Run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

