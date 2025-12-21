from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
import shutil
import uuid

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

app = FastAPI(title="Ollama RAG Demo")
# uvicorn main:app --reload 해서 실험해볼수 있음

# 프로젝트 기본 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")

# Embedding 모델 -> 텍스트를 벡터로 바꿔줌.
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# LLM 모델 (Ollama의 gemma2-2b) -> 답변을 생성할때 사용
llm = Ollama(model="gemma2:2b")

# Chroma DB 불러오기
def get_vectorstore():
    vectordb = Chroma(
        persist_directory=PERSIST_DIR, # 디렉토리가 있으면 이어서 쓰고 없으면 생성
        embedding_function=embedding
    )
    return vectordb

# 질문 Body 형식
class Question(BaseModel): # Pydantic BaseModel은 JSON 데이터를 Python 객체로 변환하는 툴임.
    question: str

#{
#  "question": "UHPC?"   ->  Question(question="UHPC?")
#}

# 답변이 바로 시작하도록 하는 Prompt (불필요한 문장 금지)
prompt = PromptTemplate(
    input_variables=["context", "question"], # 나중에 chain이 context,question을 채움.
    template="""
Using the context below, answer the user's question with a clear and complete explanation.
Provide additional helpful details and background information when relevant.
Do not say phrases like "Based on the provided text" or "According to the document."
Start your answer naturally without introductory phrases.
Use only the information from the provided context.
If the answer is not explicitly stated in the context, say:
"The document does not contain information about this topic."
Do not add or invent any external facts.

Context:
{context}

Question:
{question}

Answer:
"""
)

# 파일 업로드 API
@app.post("/upload_file") # fastApi는 BaseModel을 보고 Body데이터를 자동으로 파싱해줌.
async def upload_file(file: UploadFile = File(...)): # form-data에서 업로드된 파일을 받겠다는 의미, ...->필수라는 의미

    filename = file.filename # 업로드 된 파일 이름 저장
    temp_path = "temp_" + filename # 서버에 저장하는 경로 설정

    # 업로드된 파일 저장
    # f 라는 파일을 열고 밑에 다 실행하고 close까지 해줌.
    with open(temp_path, "wb") as f: # pdf,이미지,업로드 관련은 바이너리로 다뤄야 하기 때문에 wb사용
        f.write(await file.read())

    # txt 혹은 pdf 로더 선택
    if filename.endswith(".txt"):
        loader = TextLoader(temp_path, encoding="utf-8")
    elif filename.endswith(".pdf"):
        loader = PyMuPDFLoader(temp_path)
    else:
        os.remove(temp_path)
        raise HTTPException(status_code=400, detail="txt, pdf만 업로드 가능합니다.")

    # 문서 로드
    docs = loader.load()

    # chunk 분리
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    split_docs = splitter.split_documents(docs)

    # 문서 고유 ID 생성
    doc_uuid = str(uuid.uuid4()) # uuid.uuid4() 랜덤한 고유 id 생성
    for d in split_docs:
        d.metadata["doc_id"] = doc_uuid
        d.metadata["source"] = filename

    vectordb = get_vectorstore()

    # 파일 중복 업로드 시 기존 chunk 삭제
    # existing = vectordb._collection.get(where={"source": filename})
    # if existing and existing["ids"]:
    #     vectordb._collection.delete(existing["ids"])
    vectordb._collection.delete(where={"source": filename})

    # 새로운 chunk 저장 , 내부적으로 임베딩 실행
    vectordb.add_documents(split_docs)

    # 임시 파일 삭제
    os.remove(temp_path)

    return {
        "status": "ok",
        "message": f"{filename} uploaded and indexed",
        "chunks_added": len(split_docs),
        "doc_id": doc_uuid
    }

# 업로드된 문서 목록 조회
@app.get("/list_documents")
async def list_documents():
    vectordb = get_vectorstore()
    data = vectordb._collection.get() # _collection.get() 전체 벡터데이터

    docs = {}
    #doc_id는 “문서 하나를 대표하는 ID” 이고 그 문서는 여러 chunk로 쪼개지기 때문에 하나의 doc_id는 여러 metadata에서 반복된다.
    for meta in data.get("metadatas", []): # 각 벡터에 대응되는 메타데이터 리스트 , doc_id,source 현재 두개만 있음.
        if meta is None:
            continue
        docs[meta["doc_id"]] = meta["source"] # doc_id가 여러번 나오면 마지막 값으로 덮어씌워짐

    return {"documents": docs}

# 특정 문서 삭제
@app.delete("/delete_document")
async def delete_document(doc_id: str):

    vectordb = get_vectorstore()
    data = vectordb._collection.get(where={"doc_id": doc_id})

    if not data["ids"]:
        raise HTTPException(status_code=404, detail="Document not found.")

    vectordb._collection.delete(data["ids"])

    return {
        "status": "ok",
        "deleted_chunks": len(data["ids"])
    }

# 전체 문서 삭제 (초기화)
@app.delete("/clear_all")
async def clear_all():
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
    return {"status": "ok", "message": "All data cleared"}

# 질문 API (RAG)
@app.post("/ask")
async def ask_question(item: Question):

    vectordb = get_vectorstore()
    retriever = vectordb.as_retriever(search_kwargs={"k": 3}) # 가장 유사한 3개의 chunk를 반환하는 retriever 객체 생성.

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    result = chain.invoke({"query": item.question}) # result dictionary임,query,result,source_documnets

    #Document(page_content="...", metadata={"source": "a.pdf", "page": 0}),
    #Document(page_content="...", metadata={"source": "a.pdf", "page": 1}),
    #Document(page_content="...", metadata={"source": "b.txt"})

    return {
        "question": item.question,
        "answer": result["result"],
        "sources": [
            {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", -1)
            }
            for doc in result["source_documents"]
        ]
    }

