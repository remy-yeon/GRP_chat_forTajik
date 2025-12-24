# Ollama RAG Chat - 실행 가이드

## 프로젝트 개요

PDF/TXT 문서를 기반으로 AI에게 질문할 수 있는 RAG(Retrieval-Augmented Generation) 챗봇입니다.

- **프론트엔드**: HTML/CSS/JavaScript (index.html)
- **백엔드**: FastAPI + LangChain (llm.py)
- **LLM**: Ollama (gemma2:2b)
- **벡터DB**: ChromaDB

### 구조

```
관리자: upload.py로 문서 등록 → ChromaDB에 저장
클라이언트: 웹 UI에서 질문 → AI가 문서 기반 답변
```

---

## 최초 설치 (한 번만)

### 1. WSL Ubuntu 진입

```bash
wsl
```

### 2. 프로젝트 폴더로 이동

```bash
cd "/mnt/c/Users/boxo0/OneDrive/바탕 화면/2025-2/캡스톤디자인1/grp 파일/챗봇 프론트"
```

### 3. Python 가상환경 생성

```bash
sudo apt update && sudo apt install -y python3.12-venv
python3 -m venv .venv
```

### 4. 패키지 설치

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 5. Ollama 설치 및 모델 다운로드

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma2:2b
```

---

## 관리자: 문서 업로드

클라이언트가 질문하기 전에 관리자가 문서를 먼저 등록해야 합니다.

### 방법 1: 폴더 기반 업로드 (권장)

가장 간편한 방법입니다. 파일을 폴더에 넣고 명령어 하나로 업로드합니다.

```
📁 챗봇 프론트/
├── 추가할 파일/     ← 여기에 PDF/TXT 파일을 넣으세요
├── 추가된 파일/     ← 업로드 완료된 파일이 자동으로 이동됩니다
└── upload.py
```

**사용 방법:**

1. `추가할 파일` 폴더에 업로드할 PDF/TXT 파일을 넣습니다
2. 아래 명령어를 실행합니다:

```bash
source .venv/bin/activate
python upload.py --process
```

3. 업로드가 완료되면 파일이 자동으로 `추가된 파일` 폴더로 이동됩니다

**출력 예시:**
```
=== 3개 파일 발견 ===

업로드 중: manual.pdf... 완료! (15개 청크)
  → 'manual.pdf'로 이동 완료
업로드 중: guide.txt... 완료! (8개 청크)
  → 'guide.txt'로 이동 완료
업로드 중: faq.pdf... 완료! (12개 청크)
  → 'faq.pdf'로 이동 완료

=== 처리 완료 ===
성공: 3개, 실패: 0개

완료된 파일들은 '추가된 파일' 폴더에 있습니다.
```

---

### 방법 2: 직접 파일 지정 업로드

특정 파일을 직접 지정하여 업로드할 수도 있습니다.

```bash
# 가상환경 활성화
source .venv/bin/activate

# 단일 파일 업로드
python upload.py document.pdf

# 여러 파일 업로드
python upload.py manual.pdf guide.txt faq.pdf

# 폴더 내 모든 PDF 업로드
python upload.py ./documents/*.pdf
```

---

### 문서 목록 확인

```bash
python upload.py --list
```

출력 예시:
```
=== 저장된 문서 목록 ===
  [a1b2c3d4] manual.pdf
  [e5f6g7h8] guide.txt

총 2개 문서
```

### 문서 삭제

```bash
# 특정 문서 삭제 (doc_id 사용)
python upload.py --delete a1b2c3d4

# 전체 문서 삭제
python upload.py --clear
```

### upload.py 전체 옵션

| 옵션 | 설명 |
|------|------|
| `--process`, `-p` | 폴더 기반 자동 업로드 |
| `--list`, `-l` | 저장된 문서 목록 조회 |
| `--delete <id>`, `-d <id>` | 특정 문서 삭제 |
| `--clear` | 모든 문서 삭제 |

---

## 서버 실행 방법

### 터미널 1: Ollama 서버

```bash
wsl
ollama serve
```

### 터미널 2: FastAPI 서버

```bash
wsl
cd "/mnt/c/Users/boxo0/OneDrive/바탕 화면/2025-2/캡스톤디자인1/grp 파일/챗봇 프론트"
source .venv/bin/activate
uvicorn llm:app --host 0.0.0.0 --port 8000
```

### 브라우저에서 접속

- **웹 UI**: http://localhost:8000
- **API 문서**: http://localhost:8000/docs

---

## 클라이언트: 사용 방법

1. 브라우저에서 http://localhost:8000 접속
2. 입력창에 질문 입력
3. Enter 또는 전송 버튼 클릭
4. AI가 등록된 문서를 참고하여 답변
5. 답변 아래에 참조한 문서 표시

---

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/` | 웹 UI (index.html) |
| POST | `/upload_file` | 파일 업로드 (API) |
| POST | `/ask` | 질문하기 |
| GET | `/list_documents` | 문서 목록 조회 |
| DELETE | `/delete_document?doc_id=xxx` | 문서 삭제 |
| DELETE | `/clear_all` | 전체 문서 삭제 |

---

## 문제 해결

### Ollama 연결 오류

```bash
curl http://localhost:11434/api/tags
# 응답이 없으면 ollama serve 실행
```

### 포트 충돌

```bash
uvicorn llm:app --host 0.0.0.0 --port 8001
```

### 임베딩 모델 다운로드 오래 걸림

첫 실행 시 `intfloat/multilingual-e5-large-instruct` 모델 (약 2GB)을 다운로드합니다.
5~10분 정도 소요될 수 있습니다.

---

## 파일 구조

```
챗봇 프론트/
├── index.html          # 클라이언트 웹 UI
├── llm.py              # FastAPI 백엔드 (RAG)
├── upload.py           # 관리자용 문서 업로드 도구
├── requirements.txt    # Python 패키지 목록
├── SETUP_GUIDE.md      # 이 파일
├── 추가할 파일/         # 업로드 대기 폴더 (여기에 파일 넣기)
├── 추가된 파일/         # 업로드 완료 폴더 (처리된 파일 보관)
├── .venv/              # Python 가상환경
├── chroma_db/          # 벡터 DB (문서 저장)
└── temp_uploads/       # 임시 업로드 폴더
```

---

## 서버 종료

- **FastAPI 서버**: `Ctrl + C`
- **Ollama 서버**: `Ctrl + C`
