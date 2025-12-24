"""
Ollama RAG Chat Backend Server
FastAPI 기반 백엔드 서버
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import ollama
import os
import uuid
import json
from typing import List, Optional

app = FastAPI(title="Ollama RAG Chat API")

# CORS 설정 (프론트엔드와 통신 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== 데이터 저장소 (간단한 메모리 저장) =====
documents_store = {}  # {doc_id: {"source": filename, "content": text}}
UPLOAD_DIR = "uploads"

# 업로드 폴더 생성
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ===== Pydantic 모델 =====
class QuestionRequest(BaseModel):
    question: str

class DocumentInfo(BaseModel):
    doc_id: str
    source: str

class AskResponse(BaseModel):
    answer: str
    sources: List[dict] = []

# ===== API 엔드포인트 =====

@app.get("/")
async def root():
    """메인 페이지 (index.html) 반환"""
    return FileResponse("index.html")


@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    """PDF/TXT 파일 업로드"""

    # 파일 확장자 확인
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ['.pdf', '.txt']:
        raise HTTPException(status_code=400, detail="PDF 또는 TXT 파일만 업로드 가능합니다.")

    # 파일 저장
    doc_id = str(uuid.uuid4())[:8]
    file_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{filename}")

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # 텍스트 추출
    text_content = ""
    if ext == '.txt':
        text_content = content.decode('utf-8', errors='ignore')
    elif ext == '.pdf':
        # PDF 처리 (PyPDF2 또는 pdfplumber 필요)
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text_content += page.extract_text() or ""
        except ImportError:
            # pdfplumber가 없으면 파일명만 저장
            text_content = f"[PDF 파일: {filename}]"
        except Exception as e:
            text_content = f"[PDF 읽기 오류: {str(e)}]"

    # 문서 저장
    documents_store[doc_id] = {
        "source": filename,
        "content": text_content,
        "path": file_path
    }

    return {
        "success": True,
        "doc_id": doc_id,
        "source": filename,
        "message": "파일이 성공적으로 업로드되었습니다."
    }


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: QuestionRequest):
    """질문에 대한 답변 생성 (RAG)"""

    question = request.question

    # 모든 문서 내용 수집 (간단한 RAG)
    context = ""
    sources = []

    for doc_id, doc in documents_store.items():
        context += f"\n\n[문서: {doc['source']}]\n{doc['content'][:2000]}"  # 최대 2000자
        sources.append({
            "source": doc['source'],
            "page": None,
            "content": doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
        })

    # Ollama로 답변 생성
    try:
        if context:
            prompt = f"""다음 문서 내용을 참고하여 질문에 답변해주세요.

[참고 문서]
{context}

[질문]
{question}

[답변]"""
        else:
            prompt = question

        response = ollama.chat(
            model='llama2',  # 또는 'mistral', 'gemma' 등 설치된 모델
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        answer = response['message']['content']

        return AskResponse(
            answer=answer,
            sources=sources if context else []
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 오류: {str(e)}")


@app.get("/list_documents")
async def list_documents():
    """업로드된 문서 목록 조회"""
    documents = [
        {"doc_id": doc_id, "source": doc["source"]}
        for doc_id, doc in documents_store.items()
    ]
    return {"documents": documents}


@app.delete("/delete_document")
async def delete_document(doc_id: str):
    """개별 문서 삭제"""
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")

    # 파일 삭제
    doc = documents_store[doc_id]
    if os.path.exists(doc.get("path", "")):
        os.remove(doc["path"])

    del documents_store[doc_id]
    return {"success": True, "message": "문서가 삭제되었습니다."}


@app.delete("/clear_all")
async def clear_all():
    """모든 문서 삭제"""
    for doc_id, doc in documents_store.items():
        if os.path.exists(doc.get("path", "")):
            os.remove(doc["path"])

    documents_store.clear()
    return {"success": True, "message": "모든 문서가 삭제되었습니다."}


# ===== 서버 실행 =====
if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("Ollama RAG Chat Server")
    print("=" * 50)
    print("서버 시작: http://localhost:8000")
    print("API 문서: http://localhost:8000/docs")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)
