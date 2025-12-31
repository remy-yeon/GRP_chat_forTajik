# Ollama RAG Chat - 설치 및 실행 가이드

## 프로젝트 개요

PDF/TXT 문서를 기반으로 AI에게 질문할 수 있는 RAG(Retrieval-Augmented Generation) 챗봇입니다.

- **프론트엔드**: HTML/CSS/JavaScript (index.html)
- **백엔드**: FastAPI + LangChain (llm.py)
- **임베딩 모델**: Snowflake Arctic Embed Large (1024차원)
- **LLM**: Ollama (gemma2:2b)
- **벡터DB**: ChromaDB

### 구조

```
관리자: upload.py로 문서 등록 → ChromaDB에 저장
클라이언트: 웹 UI에서 질문 → AI가 문서 기반 답변
```

---

## 환경별 설치 가이드

### macOS (맥북)

#### 1. Homebrew 설치 (없는 경우)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### 2. Python 설치

```bash
brew install python@3.12
```

#### 3. 프로젝트 폴더로 이동

```bash
cd ~/path/to/챗봇\ 프론트
```

#### 4. Python 가상환경 생성 및 활성화

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 5. 패키지 설치

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 6. Ollama 설치 및 모델 다운로드

```bash
# Ollama 설치
brew install ollama

# Ollama 서비스 시작 (백그라운드)
brew services start ollama

# 또는 수동으로 시작
ollama serve &

# LLM 모델 다운로드
ollama pull gemma2:2b
```

---

### Windows (WSL Ubuntu)

#### 1. WSL Ubuntu 진입

```bash
wsl
```

#### 2. 프로젝트 폴더로 이동

```bash
cd "/mnt/c/Users/YOUR_USERNAME/path/to/챗봇 프론트"
```

#### 3. Python 가상환경 생성

```bash
sudo apt update && sudo apt install -y python3.12-venv
python3 -m venv .venv
```

#### 4. 패키지 설치

```bash
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### 5. Ollama 설치 및 모델 다운로드

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma2:2b
```

---

## 중요 주의사항

### 서버와 upload.py 동시 실행 금지

ChromaDB는 동시에 여러 프로세스가 접근하면 충돌이 발생합니다.

```
⚠️  반드시 아래 순서를 지켜주세요:

1. 서버(llm.py) 중지
2. upload.py로 문서 업로드
3. 서버(llm.py) 시작
```

**잘못된 예:**
```
❌ 서버 실행 중 → upload.py 실행 → DB 충돌 오류
```

**올바른 예:**
```
✅ 서버 중지 → upload.py 실행 → 서버 시작
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

1. **서버가 실행 중이라면 먼저 중지합니다** (Ctrl+C)
2. `추가할 파일` 폴더에 업로드할 PDF/TXT 파일을 넣습니다
3. 아래 명령어를 실행합니다:

```bash
# macOS
source .venv/bin/activate
python upload.py --process

# Windows (WSL)
source .venv/bin/activate
python upload.py --process
```

4. 업로드가 완료되면 파일이 자동으로 `추가된 파일` 폴더로 이동됩니다
5. **업로드 완료 후 서버를 시작합니다**

**출력 예시:**
```
임베딩 디바이스: cpu  (또는 mps/cuda)
임베딩 모델 로딩 완료!

=== 3개 파일 발견 ===

업로드 중: manual.pdf...
  - chunks: 15 (batch 64)
완료! (15개 청크)
  → 'manual.pdf'로 이동 완료

=== 처리 완료 ===
성공: 3개, 실패: 0개
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

### 문서 관리 명령어

```bash
# 문서 목록 확인
python upload.py --list

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

### macOS

**터미널 1: Ollama 서버** (이미 brew services로 시작했다면 생략)

```bash
ollama serve
```

**터미널 2: FastAPI 서버**

```bash
cd ~/path/to/챗봇\ 프론트
source .venv/bin/activate
python llm.py
```

### Windows (WSL)

**터미널 1: Ollama 서버**

```bash
wsl
ollama serve
```

**터미널 2: FastAPI 서버**

```bash
wsl
cd "/mnt/c/Users/YOUR_USERNAME/path/to/챗봇 프론트"
source .venv/bin/activate
python llm.py
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
5. 답변 아래의 **"📚 참조 문서"**를 클릭하면 참조한 문서와 내용 미리보기 표시

---

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/` | 웹 UI (index.html) |
| POST | `/ingest` | 파일 업로드 (multipart/form-data) |
| POST | `/chat` | 질문하기 |
| GET | `/documents` | 문서 목록 조회 |
| DELETE | `/documents/{doc_id}` | 특정 문서 삭제 |
| DELETE | `/documents` | 전체 문서 삭제 |

---

## 문제 해결

### Ollama 연결 오류

```bash
curl http://localhost:11434/api/tags
# 응답이 없으면 ollama serve 실행
```

### 포트 충돌

llm.py의 마지막 줄에서 포트 변경:
```python
uvicorn.run("llm:app", host="0.0.0.0", port=8001, reload=False)
```

### ChromaDB 오류 (HNSW index 로딩 실패)

DB가 손상되었거나 임베딩 모델이 변경된 경우:

```bash
# chroma_db 폴더 삭제
rm -rf chroma_db

# 문서 다시 업로드
python upload.py --process
```

### 임베딩 모델 다운로드 오래 걸림

첫 실행 시 `Snowflake/snowflake-arctic-embed-l` 모델 (약 1GB)을 다운로드합니다.
네트워크 상태에 따라 5~15분 정도 소요될 수 있습니다.

### macOS MPS (Metal) 가속

M1/M2/M3 맥북에서는 자동으로 MPS 가속이 활성화됩니다:
```
>> Embedding device: mps
```

CPU로 표시되면 PyTorch 버전을 확인하세요:
```bash
pip install --upgrade torch
```

---

## 파일 구조

```
챗봇 프론트/
├── index.html          # 클라이언트 웹 UI
├── llm.py              # FastAPI 백엔드 (RAG 서버)
├── upload.py           # 관리자용 문서 업로드 도구
├── requirements.txt    # Python 패키지 목록
├── SETUP_GUIDE.md      # 이 파일
├── 추가할 파일/         # 업로드 대기 폴더 (여기에 파일 넣기)
├── 추가된 파일/         # 업로드 완료 폴더 (처리된 파일 보관)
├── .venv/              # Python 가상환경
├── chroma_db/          # 벡터 DB (문서 저장)
└── data_storage/       # 업로드된 원본 파일 저장
```

---

## 서버 종료

- **FastAPI 서버**: `Ctrl + C`
- **Ollama 서버**: `Ctrl + C`
- **macOS Ollama (brew services)**: `brew services stop ollama`

---

## 버전 정보

- **임베딩 모델**: Snowflake Arctic Embed Large (1024차원)
- **ChromaDB**: 1.4.0
- **Python**: 3.12+
