import os
import shutil
import time
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# LangChain & AI 관련 임포트
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from typing import List, Optional

# 1. 설정 및 디렉토리 관리
UPLOAD_DIR = "data_storage"  # PDF 원본 저장소
DB_DIR = "chroma_db"         # 벡터 DB 저장소

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시 저장소 폴더 생성
    if not os.path.exists(UPLOAD_DIR): os.makedirs(UPLOAD_DIR)
    if not os.path.exists(DB_DIR): os.makedirs(DB_DIR)
    print(">> [System] Server Started. Storage Ready.")
    yield
    print(">> [System] Server Shutdown.")

app = FastAPI(title="Tajikistan Tour Guide API (Eng/Rus)", lifespan=lifespan)

# 2. AI 모델 설정 (Mac M2 최적화)
# GPU 사용 가능 여부 확인 (MPS: Mac Metal Performance Shaders)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f">> Embedding Model Device: {device}")

# 임베딩 모델 (다국어 지원 모델 유지)
embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large-instruct",
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': True}
)

# LLM 모델 (Gemma 2 2B)
llm = ChatOllama(model="gemma2:2b", temperature=0.1) # 0.1로 낮춰서 팩트 기반 답변 강화

# 3. 데이터 모델 (요청/응답 형식)
class SourceInfo(BaseModel):
    file: str                # 파일 이름 (예: tajikistan_guide_en.txt)
    page: Optional[int] = None  # PDF면 page, txt면 None
    snippet: str             # 해당 청크 일부 (앞부분)

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    time_taken: float
    sources: List[SourceInfo]  # 어떤 문서를 썼는지 리스트로 반환

# 4. RAG 체인 생성 함수
def get_vectorstore():
    """저장된 ChromaDB를 불러옵니다."""
    return Chroma(persist_directory=DB_DIR, embedding_function=embedding_model)

def get_rag_chain():
    """영어/러시아어 전용 가이드 페르소나 설정 (컨텍스트 밖 지식 사용 금지)."""
    
    template = """
    You are an AI assistant specialized in Tajikistan tourism.

    You MUST follow these rules strictly:
    - Use ONLY the information provided in [Context].
    - Do NOT add any facts, names, numbers, or locations that are not explicitly stated in [Context].
    - If the answer cannot be clearly found in [Context], you MUST say:
      * English: "I don't have information about that in my documents."
      * Russian: "У меня нет информации об этом в моих документах."
    - If the user asks in English, answer in English.
    - If the user asks in Russian (Cyrillic), answer in Russian.
    - Do NOT use Korean or Tajik.
    - Keep your answer concise and helpful.

    [Context]:
    {context}

    [User Question]:
    {question}

    [Answer]:
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    return prompt | llm | StrOutputParser()

# 5. API 엔드포인트

@app.post("/ingest", summary="Document Upload & Learning")
async def ingest_document(file: UploadFile = File(...)):    
    """PDF/TXT 파일을 업로드하여 AI에게 학습시킵니다 (ChromaDB 저장)."""
    try:
        # 1. 파일 저장
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. 확장자에 따라 다른 로더 사용
        filename_lower = file.filename.lower()

        if filename_lower.endswith(".pdf"):
            # PDF → PyMuPDFLoader
            loader = PyMuPDFLoader(file_path)

        elif filename_lower.endswith(".txt"):
            # TXT → TextLoader
            # 인코딩은 상황에 맞게 조정 (영어/러시아어면 utf-8이면 거의 안전)
            loader = TextLoader(file_path, encoding="utf-8")

        else:
            # 지원하지 않는 확장자
            raise HTTPException(
                status_code=400,
                detail="Only .pdf and .txt files are supported."
            )

        # 3. 문서 로드
        docs = loader.load()
        
        # 4. 텍스트 쪼개기 (500자 단위)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        splits = text_splitter.split_documents(docs)
        
        # 5. 벡터 DB 저장 (기존 DB에 추가하는 방식)
        Chroma.from_documents(
            documents=splits,
            embedding=embedding_model,
            persist_directory=DB_DIR
        )
        
        return {
            "message": f"Successfully ingested {len(splits)} chunks from {file.filename}."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse, summary="Ask to AI Guide")
async def chat(request: ChatRequest):
    """학습된 내용을 바탕으로 질문에 답변합니다 (영어/러시아어, 출처 정보 포함)."""
    start_time = time.time()
    
    try:
        # 1. 벡터스토어 & 리트리버 생성
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 10}
        )
        
        # 2. 관련 문서 조회
        docs = retriever.invoke(request.question)

        # 2-1. 아예 아무 문서도 못 찾은 경우 → 바로 "모른다" 응답
        if not docs:
            end_time = time.time()
            # 러시아어(키릴 문자) 여부 간단 판별
            is_russian = any("\u0400" <= ch <= "\u04FF" for ch in request.question)

            if is_russian:
                answer = "У меня нет информации об этом в моих документах."
            else:
                answer = "I don't have information about that in my documents."

            return ChatResponse(
                answer=answer,
                time_taken=end_time - start_time,
                sources=[]
            )

        # 3. 컨텍스트 문자열 구성 + 출처 리스트 만들기
        context_parts = []
        source_infos: List[SourceInfo] = []

        for doc in docs:
            source_path = doc.metadata.get("source", "unknown")
            filename = os.path.basename(source_path)
            page = doc.metadata.get("page")  # PyMuPDFLoader는 page 번호 넣어줌 (0-based or 1-based)
            
            # 스니펫: 앞부분 300자 정도만
            snippet = doc.page_content[:300]

            # 컨텍스트에 본문만 붙이기
            context_parts.append(doc.page_content)

            # 응답에 넣을 출처 정보
            source_infos.append(
                SourceInfo(
                    file=filename,
                    page=page,
                    snippet=snippet
                )
            )

        context_str = "\n\n---\n\n".join(context_parts)

        # 4. RAG 체인 실행 (오직 context_str만 넘겨줌)
        chain = get_rag_chain()
        answer = chain.invoke({
            "context": context_str,
            "question": request.question
        })

        end_time = time.time()

        return ChatResponse(
            answer=answer,
            time_taken=end_time - start_time,
            sources=source_infos
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # 서버 실행 (포트 8000)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
# ```

# ---

### 3단계: 실행 및 테스트 방법 (매우 쉬움)

# FastAPI는 **자동 테스트 페이지(Swagger UI)**를 제공합니다. 복잡하게 HTML 만들 필요 없이 바로 테스트 가능합니다.

# 1.  **서버 실행:**
#     터미널에서 아래 명령어를 입력합니다.
#     ```bash
#     python main.py
#     ```

# 2.  **테스트 페이지 접속:**
#     웹 브라우저를 켜고 **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)** 로 접속합니다.

# 3.  **테스트 순서:**
#     * **Step 1: 문서 학습시키기 (`/ingest`)**
#         1.  화면에서 `POST /ingest` 클릭 -> `Try it out` 클릭.
#         2.  `file` 항목에서 준비된 **타지키스탄 관련 PDF(영어 또는 러시아어)**를 선택.
#         3.  `Execute` 버튼 클릭.
#         4.  Response Body에 "Successfully ingested..."가 뜨면 성공!

#     * **Step 2: 질문하기 (`/chat`)**
#         1.  화면에서 `POST /chat` 클릭 -> `Try it out` 클릭.
#         2.  Request body에 질문 입력:
#             ```json
#             {
#               "question": "What are the famous foods in Tajikistan?"
#             }