import os
import re
import torch
import time  # Import time module
from flask_cors import CORS

from flask import Flask, request, jsonify, render_template
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from kiwipiepy import Kiwi
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema import Document
from langchain.document_loaders import PyMuPDFLoader  


#랭스미스 사용시 설정 
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGCHAIN_API_KEY"] = "langchain api key 발급 받아 작성 "
# os.environ["LANGCHAIN_PROJECT"] = "프로젝트 이름 작성"

app = Flask(__name__)
CORS(app, resources={r"/upload": {"origins": "*"}})  # 모든 출처 허용

kiwi = Kiwi()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
# Load and preprocess documents  - 기존에 전처리랑 로드 합쳐놓은 함수 
def load_and_preprocess_documents(filepath):
    loader = PyMuPDFLoader(filepath)
    docs = loader.load()
    for i in range(len(docs)):
        docs[i].page_content = docs[i].page_content.replace("ft", "처")
        docs[i].page_content = re.sub(r'\s+', ' ', docs[i].page_content)
    text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=100, separator=' ')
    texts = text_splitter.split_documents(docs)
    return texts
"""
# 3차 디벨롭 추가 내용) 문서 전처리 함수
def preprocess_documents(docs):
    for i in range(len(docs)):
        docs[i].page_content = docs[i].page_content.replace("ft", "처")
        docs[i].page_content = re.sub(r'\s+', ' ', docs[i].page_content)
    return docs

# 3차 디벨롭 추가 내용) 시맨틱 청크 생성 함수
def split_documents(docs, embeddings):
    semantic_splitter = SemanticChunker(embeddings=embeddings)
    semantic_chunks = semantic_splitter.split_documents(docs)
    return semantic_chunks

#  3차 디벨롭 추가 내용) 문서 로드 및 전처리 함수
def load_and_preprocess_documents(filepath, hf_embeddings):
    loader = PyMuPDFLoader(filepath)
    docs = loader.load()
    preprocessed_docs = preprocess_documents(docs)  # 문서 전처리
    texts = split_documents(preprocessed_docs, hf_embeddings)  # 청크 생성
    print(f"생성된 청크 수: {len(texts)}") #디버깅 용 
    return texts

def load_hf_embeddings(model_name, device):
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': device}, encode_kwargs={'normalize_embeddings': True})

#파시스 디비 사용 
def create_faiss_vector_store(docs, hf):
    return FAISS.from_documents(docs, hf)

def create_bm25_retriever(docs):
    return BM25Retriever.from_documents(docs, preprocess_func=lambda text: [token.form for token in kiwi.tokenize(text)])

#앙상블 사용했으나, 속도 느림 주의 
def create_ensemble_retriever(bm25_retriever, faiss_retriever): 
    return EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.7, 0.3], search_type='mmr')

def get_prompt_template(): #Prompt template - alpaca 사용 
    return '''
    <Developer's Instruction> 
    You are an AI assistant designed for Tajikistan tourism services.  
    Please answer questions in accordance with the following rules.
    The following information (context) is content retrieved from the document.  
    Rules for providing answers:  
    1. Your responses must be strictly based on the information provided in the context.  
    2. If you cannot find sufficient information to answer, respond with: "I'm sorry, I don't know."  
    3. All responses must be short, no more than five lines, and in Korean.  
    4. Please respond in complete sentences.
    
   ############ Similar information found in the document: {context} ############
    
   ############ Question about the document: {question} ###########
    
   ############ AI Assistant's Answer:
    '''

def load_llm_model(model_name):
    return ChatOllama(model=model_name, temperature=0)

def create_rag_chain(retriever, llm, prompt_template):
    prompt = ChatPromptTemplate.from_template(prompt_template)
    return {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()

# Flask route to render HTML page
@app.route('/')
def index():
    return render_template('index.html')

@app.before_request
def before_request():
    if request.method == 'POST' and request.path == '/upload':
        print("파일 업로드 요청에 대한 미들웨어 실행")
        
        # 파일이 요청에 포함되어 있는지 확인
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # 질문이 요청 폼에 있는지 확인
        if 'question' not in request.form or not request.form['question']:
            return jsonify({"error": "Question cannot be empty"}), 400

        print("요청 파라미터가 유효합니다.")



@app.route("/upload", methods=["POST"])
def upload():
    # Debugging - http 통신 확인 겸 
    print("upload 접근::upload함수 시작") 

    file = request.files['file']
    print("request file")

    question = request.form['question']
    print("request form question")
    
    # 챗봇 Debugging log
    if 'file' not in request.files:
        print("파일 없음")
        return jsonify({"error": "No file part"}), 400
        
    print(f"File part present: {'file' in request.files}")
    print(f"Question part present: {'question' in request.form}")

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400
    
    filepath = os.path.join("upload", file.filename)
    file.save(filepath)
    
    print(f"File saved at {filepath}")

    start_time = time.time()  # Start timer

    # 1. HuggingFace 임베딩 로드
    hf_embeddings = load_hf_embeddings("intfloat/multilingual-e5-large-instruct", device)
    print("HuggingFace embeddings loaded")  # 디버깅 코드

    docs = load_and_preprocess_documents(filepath,hf_embeddings)
    print("Documents loaded and preprocessed") #디버깅 코드 

    hf = load_hf_embeddings("intfloat/multilingual-e5-large-instruct", device)
    print("HuggingFace embeddings loaded") #디버깅 코드 
    
    faiss_vector = create_faiss_vector_store(docs, hf)
    print("FAISS vector store created") #디버깅 코드 
    
    bm25_retriever = create_bm25_retriever(docs)
    print("BM25 retriever created") #디버깅 코드 
    
    ensemble_retriever = create_ensemble_retriever(bm25_retriever, faiss_vector.as_retriever(search_kwargs={'k': 3}))
    print("Ensemble retriever created") #디버깅 코드 
    
    #  가져올 모델 입력 
    # llm = load_llm_model("bnksys/yanolja-eeve-korean-instruct-10.8b:latest")
    llm = load_llm_model("gemma3:1b") #gemma3:1b 사용 
    print("LLM model loaded") #디버깅 코드 
    
    prompt_template = get_prompt_template()
    print("prompt_template loaded") #디버깅 코드 
    
    rag_chain = create_rag_chain(ensemble_retriever, llm, prompt_template)
    
    question = request.form['question']
    result = rag_chain.invoke(question)
    print(f"RAG chain result: {result}")

    # 답변 생성 시간 계산 
    end_time = time.time()
    time_taken = end_time - start_time
    
    return jsonify({"answer": result, "time_taken": time_taken})  


