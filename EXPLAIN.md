# 📘 Tajikistan RAG API 사용법 가이드  
*(main.py + static/index.html 기준)*

---

## 1️⃣ 실행 환경 준비

### 1. Python 환경

```bash
python --version
# Python 3.9 이상 권장
```

가상환경 사용을 권장합니다.

```bash
python -m venv venv
source venv/bin/activate
```

---

### 2. 필수 패키지 설치

```bash
pip install fastapi uvicorn langchain langchain-community langchain-core
pip install langchain-chroma langchain-ollama chromadb
pip install pydantic pymupdf
```

---

### 3. Ollama 실행 (LLM + Embedding)

```bash
ollama pull gemma2:2b
ollama pull snowflake-arctic-embed2
ollama serve
```

⚠️ Ollama는 **백그라운드에서 실행 중이어야 합니다.**

---

## 2️⃣ 서버 실행

### 프로젝트 구조 예시

```
project/
 ├─ main.py
 ├─ data_storage/
 ├─ chroma_db/
 └─ static/
     └─ index.html
```

### 서버 실행

```bash
python main.py
```

정상 실행 시 출력 예시:

```
>> Server Started
INFO: Uvicorn running on http://0.0.0.0:8000
```

---

## 3️⃣ 웹 UI 접속

브라우저에서 아래 주소로 접속합니다.

http://localhost:8000

- `static/index.html`이 자동으로 로드됩니다.
- 문서 업로드 + 채팅 UI를 제공합니다.

---

## 4️⃣ 문서 업로드 (RAG 데이터 구축)

### 4.1 업로드 가능한 파일 형식

- ✅ `.pdf`
- ✅ `.txt`
- ❌ 그 외 형식은 지원하지 않습니다.

예시 문서:
- tajikistan_travel_guide.pdf
- embassy_list.txt
- hospital_info.txt

---

### 4.2 업로드 동작 과정 (자동 처리)

문서 업로드 시 내부적으로 다음 과정이 수행됩니다.

```
PDF/TXT 업로드
 → 텍스트 추출
 → 전처리
 → SemanticChunker로 의미 단위 분할
 → 임베딩 생성 (Snowflake Arctic Embed2)
 → Chroma DB 저장
 → BM25 인덱스 재생성
```

성공 응답 예시:

```json
{
  "message": "Ingested 128 chunks",
  "doc_id": "c3a2f9c1-..."
}
```

---

## 5️⃣ 질문하기 (Chat 사용법)

### 5.1 일반 관광 질문

두샨베에서 가볼 만한 곳은?

- RAG 기반 관광 정보 답변 제공  
- 관련 문서 출처 함께 반환  

---

### 5.2 맥락 질문

사용자: 두샨베 국립박물관 알려줘  
사용자: 거기 입장료는?

- 이전 대화(history)를 기반으로  
- “국립박물관”을 자동으로 참조하여 답변  

---

### 5.3 러시아어 질문

Я потерял паспорт.

- 러시아어 자동 감지  
- 러시아어로 답변 생성  

---

## 6️⃣ 대사관 기능 사용법 (여권 분실)

### 예시 1: 영어

I lost my passport.

챗봇 응답:

I’m sorry you’re dealing with this.  
What is your nationality?

Korea

→ 대한민국 대사관 정보 출력  
- 주소  
- 전화번호  
- 도시  

---

### 예시 2: 러시아어

Я потерял паспорт.  
Корея

→ 러시아어 대사관 정보 출력  
- 하드맵 기반 제공  
- 주소·전화번호 정확성 보장  

---

## 7️⃣ 병원 기능 사용법

### 7.1 도시를 포함한 질문

I feel sick. Is there a hospital in Dushanbe?

→ 두샨베 병원 목록 제공  

---

### 7.2 도시를 나중에 말하는 경우

I need a hospital.

챗봇:  
Which city are you in?

Dushanbe

→ 해당 도시 병원 정보 출력  

---

### 7.3 러시아어 병원 질문

Мне плохо. Где больница?  
Душанбе

→ 러시아어 병원 정보 제공  
→ 하드맵에 있는 병원만 출력 (정확성 우선)  

---

## 8️⃣ 출처(Source) 확인

모든 일반 RAG 답변에는 출처가 포함됩니다.

```json
{
  "answer": "두샨베 국립박물관 입장료는 30소모니...",
  "sources": [
    {
      "file": "tajikistan_travel_guide.pdf",
      "page": 15,
      "snippet": "National Museum entrance: 30 TJS..."
    }
  ]
}
```

- ✔ 사용자는 원본 문서 직접 검증 가능  
- ✔ 환각(Hallucination) 방지  

---

## 9️⃣ 문서 관리 API (선택 기능)

### 9.1 업로드된 문서 목록 조회

GET /documents

---

### 9.2 특정 문서 삭제

DELETE /documents/{doc_id}

---

### 9.3 전체 문서 초기화

DELETE /documents

---

## 🔁 전체 사용자 흐름 요약

```
서버 실행
 → 웹 접속
 → 문서 업로드
 → 질문 입력
 → RAG 검색 + LLM 응답
 → 출처 확인
```

---

## 📌 한 문장 요약

이 시스템은 문서를 업로드하면 즉시 반영되는  
RAG 기반 관광 챗봇으로,  
일반 관광 정보부터 대사관·병원 같은 응급 상황까지  
다국어로 정확하게 안내할 수 있습니다.
