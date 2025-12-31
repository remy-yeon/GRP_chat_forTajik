#!/usr/bin/env python3
"""
관리자용 문서 업로드 스크립트

사용법:
    python upload.py 파일1.pdf 파일2.txt ...
    python upload.py ./documents/*.pdf
    python upload.py --process              # 폴더 기반 자동 업로드

예시:
    python upload.py manual.pdf guide.txt
    python upload.py /path/to/documents/*.pdf
    python upload.py --process              # "추가할 파일" → 업로드 → "추가된 파일"로 이동
"""

import os
import sys
import uuid
import shutil
import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

# -----------------------------
# LangChain / VectorStore
# -----------------------------
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# 선택적 import
SEMANTIC_CHUNK_AVAILABLE = True
try:
    from langchain_experimental.text_splitter import SemanticChunker
except Exception:
    SEMANTIC_CHUNK_AVAILABLE = False

SEMANTIC_MAX_DOC_CHARS = int(os.getenv("SEMANTIC_MAX_DOC_CHARS", "50000"))
SEMANTIC_MAX_TOTAL_CHARS = int(os.getenv("SEMANTIC_MAX_TOTAL_CHARS", "200000"))
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
UPLOAD_BATCH_SIZE = int(os.getenv("UPLOAD_BATCH_SIZE", "64"))


# =============================
# 경로 설정
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")

# 폴더 기반 업로드용 경로
PENDING_DIR = os.path.join(BASE_DIR, "추가할 파일")      # 업로드 대기 폴더
COMPLETED_DIR = os.path.join(BASE_DIR, "추가된 파일")    # 업로드 완료 폴더


# =============================
# 임베딩 모델 (llm.py와 동일)
# =============================
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"임베딩 디바이스: {DEVICE}")


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


print("임베딩 모델 로딩 중... (최초 실행 시 다운로드에 시간이 걸립니다)")
embedding = ArcticEmbedEmbeddings()
print("임베딩 모델 로딩 완료!")


# =============================
# 벡터 DB
# =============================
def get_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding,
        collection_metadata={"hnsw:space": "cosine"},
    )


# =============================
# 텍스트 전처리
# =============================
def preprocess_text(text: str) -> str:
    import re
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_documents(docs):
    for d in docs:
        d.page_content = preprocess_text(d.page_content)
    return docs


# =============================
# 문서 청킹
# =============================
def _should_use_semantic(docs, use_semantic: bool) -> bool:
    if not use_semantic or not SEMANTIC_CHUNK_AVAILABLE:
        return False

    total_chars = 0
    max_doc_chars = 0
    for doc in docs:
        if not doc.page_content:
            continue
        size = len(doc.page_content)
        total_chars += size
        if size > max_doc_chars:
            max_doc_chars = size
        if max_doc_chars > SEMANTIC_MAX_DOC_CHARS or total_chars > SEMANTIC_MAX_TOTAL_CHARS:
            return False

    return True


def split_documents(
    docs,
    use_semantic=True,
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
):
    if _should_use_semantic(docs, use_semantic):
        try:
            return SemanticChunker(
                embedding=embedding,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=90,
            ).split_documents(docs)
        except Exception:
            pass

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


def _add_documents_in_batches(vectordb, chunks, batch_size: int, show_progress: bool) -> None:
    total = len(chunks)
    if total == 0:
        return

    if batch_size <= 0 or batch_size >= total:
        vectordb.add_documents(chunks)
        return

    for start in range(0, total, batch_size):
        batch = chunks[start : start + batch_size]
        vectordb.add_documents(batch)
        if show_progress:
            done = min(start + batch_size, total)
            print(f"  - embedded {done}/{total} chunks", end="\r", flush=True)

    if show_progress:
        print(f"  - embedded {total}/{total} chunks", flush=True)


# =============================
# 파일 업로드 함수
# =============================
def upload_file(file_path: str, show_progress: bool = False) -> dict:
    """파일을 ChromaDB에 업로드"""

    file_path = Path(file_path)

    if not file_path.exists():
        return {"success": False, "error": f"파일을 찾을 수 없습니다: {file_path}"}

    ext = file_path.suffix.lower()

    if ext not in ['.pdf', '.txt']:
        return {"success": False, "error": f"지원하지 않는 파일 형식: {ext} (PDF, TXT만 가능)"}

    try:
        # 문서 로드
        if ext == '.pdf':
            loader = PyMuPDFLoader(str(file_path))
        else:
            loader = TextLoader(str(file_path), encoding='utf-8')

        docs = preprocess_documents(loader.load())
        chunks = split_documents(docs)

        if show_progress and chunks:
            print(f"  - chunks: {len(chunks)} (batch {UPLOAD_BATCH_SIZE})", flush=True)

        # 메타데이터 추가
        doc_id = str(uuid.uuid4())
        filename = file_path.name

        for i, chunk in enumerate(chunks):
            chunk.metadata["doc_id"] = doc_id
            chunk.metadata["source"] = filename
            chunk.metadata["chunk_index"] = i

        # 벡터 DB에 저장
        vectordb = get_vectorstore()
        _add_documents_in_batches(vectordb, chunks, UPLOAD_BATCH_SIZE, show_progress)

        return {
            "success": True,
            "filename": filename,
            "doc_id": doc_id,
            "chunks": len(chunks),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================
# 문서 목록 조회
# =============================
def list_documents():
    """저장된 문서 목록 출력"""
    vectordb = get_vectorstore()

    try:
        data = vectordb._collection.get(include=["metadatas"])

        docs = {}
        for m in data["metadatas"]:
            if m and "doc_id" in m and "source" in m:
                docs[m["doc_id"]] = m["source"]

        return docs
    except Exception:
        return {}


# =============================
# 문서 삭제
# =============================
def delete_document(doc_id: str):
    """특정 문서 삭제"""
    vectordb = get_vectorstore()

    try:
        data = vectordb._collection.get(where={"doc_id": doc_id})
        if data["ids"]:
            vectordb._collection.delete(data["ids"])
            return {"success": True, "deleted_chunks": len(data["ids"])}
        else:
            return {"success": False, "error": "문서를 찾을 수 없습니다"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def clear_all():
    """모든 문서 삭제"""
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
        return {"success": True}
    return {"success": False, "error": "DB가 없습니다"}


# =============================
# 폴더 기반 업로드
# =============================
def process_folder():
    """
    '추가할 파일' 폴더의 파일들을 업로드하고
    완료된 파일을 '추가된 파일' 폴더로 이동
    """
    # 폴더 생성 (없으면)
    os.makedirs(PENDING_DIR, exist_ok=True)
    os.makedirs(COMPLETED_DIR, exist_ok=True)

    # 지원 파일 확장자
    supported_extensions = {'.pdf', '.txt'}

    # 업로드 대기 파일 목록
    pending_files = []
    for file_name in os.listdir(PENDING_DIR):
        file_path = Path(PENDING_DIR) / file_name
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            pending_files.append(file_path)

    if not pending_files:
        print(f"\n'{PENDING_DIR}' 폴더에 업로드할 파일이 없습니다.")
        print(f"PDF 또는 TXT 파일을 해당 폴더에 넣어주세요.")
        return

    print(f"\n=== {len(pending_files)}개 파일 발견 ===\n")

    success_count = 0
    fail_count = 0

    for file_path in pending_files:
        print(f"업로드 중: {file_path.name}...")
        result = upload_file(str(file_path), show_progress=True)

        if result["success"]:
            print(f"완료! ({result['chunks']}개 청크)")

            # 완료된 파일을 '추가된 파일' 폴더로 이동
            dest_path = Path(COMPLETED_DIR) / file_path.name

            # 같은 이름의 파일이 있으면 번호 붙이기
            if dest_path.exists():
                base = file_path.stem
                ext = file_path.suffix
                counter = 1
                while dest_path.exists():
                    dest_path = Path(COMPLETED_DIR) / f"{base}_{counter}{ext}"
                    counter += 1

            shutil.move(str(file_path), str(dest_path))
            print(f"  → '{dest_path.name}'로 이동 완료")
            success_count += 1
        else:
            print(f"실패: {result['error']}")
            fail_count += 1

    print(f"\n=== 처리 완료 ===")
    print(f"성공: {success_count}개, 실패: {fail_count}개")

    if success_count > 0:
        print(f"\n완료된 파일들은 '{COMPLETED_DIR}' 폴더에 있습니다.")


# =============================
# 메인
# =============================
def main():
    parser = argparse.ArgumentParser(
        description="관리자용 문서 업로드 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python upload.py document.pdf           # 단일 파일 업로드
  python upload.py *.pdf *.txt            # 여러 파일 업로드
  python upload.py --process              # 폴더 기반 자동 업로드
  python upload.py --list                 # 문서 목록 조회
  python upload.py --delete <doc_id>      # 특정 문서 삭제
  python upload.py --clear                # 전체 삭제

폴더 기반 업로드:
  1. '추가할 파일' 폴더에 PDF/TXT 파일을 넣습니다
  2. python upload.py --process 실행
  3. 업로드 완료된 파일은 '추가된 파일' 폴더로 이동됩니다
        """
    )

    parser.add_argument("files", nargs="*", help="업로드할 파일들 (PDF, TXT)")
    parser.add_argument("--process", "-p", action="store_true", help="폴더 기반 자동 업로드 ('추가할 파일' → '추가된 파일')")
    parser.add_argument("--list", "-l", action="store_true", help="저장된 문서 목록 조회")
    parser.add_argument("--delete", "-d", metavar="DOC_ID", help="특정 문서 삭제")
    parser.add_argument("--clear", action="store_true", help="모든 문서 삭제")

    args = parser.parse_args()

    # 폴더 기반 자동 업로드
    if args.process:
        process_folder()
        return

    # 문서 목록 조회
    if args.list:
        docs = list_documents()
        if docs:
            print("\n=== 저장된 문서 목록 ===")
            for doc_id, source in docs.items():
                print(f"  [{doc_id[:8]}] {source}")
            print(f"\n총 {len(docs)}개 문서")
        else:
            print("저장된 문서가 없습니다.")
        return

    # 특정 문서 삭제
    if args.delete:
        result = delete_document(args.delete)
        if result["success"]:
            print(f"삭제 완료: {result['deleted_chunks']}개 청크 삭제됨")
        else:
            print(f"삭제 실패: {result['error']}")
        return

    # 전체 삭제
    if args.clear:
        confirm = input("정말 모든 문서를 삭제하시겠습니까? (y/N): ")
        if confirm.lower() == 'y':
            result = clear_all()
            if result["success"]:
                print("모든 문서가 삭제되었습니다.")
            else:
                print(f"삭제 실패: {result.get('error', 'Unknown error')}")
        else:
            print("취소되었습니다.")
        return

    # 파일 업로드
    if not args.files:
        parser.print_help()
        return

    print(f"\n=== {len(args.files)}개 파일 업로드 시작 ===\n")

    success_count = 0
    fail_count = 0

    for file_path in args.files:
        print(f"업로드 중: {file_path}...")
        result = upload_file(file_path, show_progress=True)

        if result["success"]:
            print(f"완료! ({result['chunks']}개 청크)")
            success_count += 1
        else:
            print(f"실패: {result['error']}")
            fail_count += 1

    print(f"\n=== 업로드 완료 ===")
    print(f"성공: {success_count}개, 실패: {fail_count}개")


if __name__ == "__main__":
    main()
