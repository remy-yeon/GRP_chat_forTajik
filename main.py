"""
main.py - Single-file RAG API (Ollama Embeddings + Chroma + Hybrid Retriever)

Stack:
- Chunking: SemanticChunker (fallback: RecursiveCharacterTextSplitter)
- Vector DB: Chroma (persist_directory)
- Embedding: OllamaEmbeddings("snowflake-arctic-embed2")
- LLM: ChatOllama("gemma2:2b")
- Hybrid Retrieval: Dense(MMR) + Sparse(BM25) merged by RRF
- Performance: VectorDB + BM25 cached in app.state, BM25 rebuilt only on ingest/delete/clear
"""

from __future__ import annotations

import os
import re
import shutil
import time
import uuid
import asyncio
import math
from typing import List, Optional, Dict, Tuple
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata


# =========================
# 1) Path 설정
# =========================
UPLOAD_DIR = "data_storage"
DB_DIR = "chroma_db"
STATIC_DIR = "static"


# =========================
# 2) Retrieval 튜닝 파라미터
# =========================
# Dense 검색(MMR)
K_DENSE = 20
FETCH_K = 40
LAMBDA_MULT = 0.35

# Sparse 검색(BM25)
K_SPARSE = 20

# Hybrid merge
K_FINAL = 8
RRF_K = 60

# Context 길이 제한(너무 길면 속도/품질 흔들림 방지)
MAX_CONTEXT_CHARS = 6500

# History 길이 제한(프롬프트 과부하 방지)
MAX_HISTORY_CHARS = 2500
MAX_HISTORY_TURNS = 12


# =========================
# 3) Prompt
# =========================
PROMPT_TEMPLATE = """
You are a helpful travel assistant for tourists interested in visiting Tajikistan.

Use ONLY the information provided in [Context] for factual claims.

You may use [History] only to understand conversational references
(e.g., "that place", "the previous one", "what you said earlier"),
but do NOT introduce new facts from [History] that are not supported by [Context].

Language rules (STRICT):
- If the question is in Russian, answer in Russian.
- Otherwise, answer in English.
- Do not use any other language.

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
7. Write in clear, natural sentences suitable for a travel guide or tourism app.

[History]:
{history}

[Context]:
{context}

[Question]:
{question}

[Answer]:
""".strip()

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)


# =========================
# 3.1) Embassy Prompt
# =========================
EMBASSY_PROMPT_TEMPLATE = """
You are ONLY responsible for writing natural sentences.

STRICT RULES:
- You MUST copy EmbassyName, City, Address, and Phone EXACTLY as given.
- You MUST NOT translate, rephrase, or modify any names, addresses, or phone numbers.
- You MUST NOT use Markdown, lists, labels, or formatting.
- Output plain text only.
- You MUST mention ONLY ONE embassy.
- You MUST NOT mention any other embassy or country.


LANGUAGE OVERRIDE (ABSOLUTE):
- The answer language is already decided.
- DO NOT decide the language yourself.
- Write ONLY in the language specified below.

Answer language: {language}

Your job:
- Write 2–3 short, clear sentences for a traveler who lost a passport.
- Use the provided values exactly as they appear.

[Context]:
{context}

[Answer]:
""".strip()

embassy_prompt = ChatPromptTemplate.from_template(EMBASSY_PROMPT_TEMPLATE)


# =========================
# 3.2) Hospital Prompt
# =========================
HOSPITAL_PROMPT_TEMPLATE = """
You are ONLY responsible for writing natural sentences.

STRICT RULES:
- You MUST copy hospital name, city, address, and phone EXACTLY as given.
- You MUST NOT translate, rephrase, or modify any names, addresses, or phone numbers.
- You MUST NOT add hospitals or remove hospitals.
- You MUST NOT merge information from different hospitals.
- You MUST NOT use Markdown, lists, bullets, or labels.
- Output plain text only.

FORMAT RULES (VERY IMPORTANT):
- Each hospital MUST be written as a separate paragraph.
- Insert a blank line between hospitals.
- Each paragraph MUST mention only ONE hospital.

LANGUAGE OVERRIDE:
- The answer language is already decided.
- Write ONLY in the language specified below.

Answer language: {language}

Your job:
- For EACH hospital block, write exactly 2 short sentences:
  1) where the hospital is located
  2) how to contact it (phone)

[Context]:
{context}

[Answer]:
""".strip()

hospital_prompt = ChatPromptTemplate.from_template(HOSPITAL_PROMPT_TEMPLATE)


# =========================
# 4) LLM / Embedding (Ollama)
# =========================
llm = ChatOllama(model="gemma2:2b", temperature=0.2)
embedding_model = OllamaEmbeddings(model="snowflake-arctic-embed2")

translator_llm = ChatOllama(model="gemma2:2b", temperature=0.0)


# =========================
# 5) Vector Store
# =========================
def get_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )


# =========================
# 6) API Models
# =========================
class SourceInfo(BaseModel):
    file: str
    page: Optional[int] = None
    snippet: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history: Optional[List[ChatMessage]] = None


class ChatResponse(BaseModel):
    answer: str
    time_taken: float
    sources: List[SourceInfo]


# =========================
# 7) Utils
# =========================
def preprocess_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def is_russian(text: str) -> bool:
    return any("\u0400" <= c <= "\u04FF" for c in (text or ""))


def is_russian_with_history(text: str, history: Optional[List[ChatMessage]]) -> bool:
    combined = text or ""
    if history:
        for m in history:
            combined += " " + (m.content or "")
    return any("\u0400" <= c <= "\u04FF" for c in combined)


def translate_answer_if_needed(answer: str, question: str) -> str:
    if not is_russian(question):
        return answer

    prompt_txt = f"""
Translate the following text into Russian.

Rules:
- Preserve ALL factual details exactly.
- Do NOT add new information.
- Do NOT remove information.
- Output ONLY Russian.

TEXT:
{answer}

RUSSIAN:
""".strip()

    try:
        return translator_llm.invoke(prompt_txt).content.strip()
    except Exception:
        return answer


def translate_context_if_needed(context: str, question: str) -> str:
    if not is_russian(question):
        return context

    translate_prompt = f"""
Translate the following text into Russian.

Rules:
- Preserve ALL factual details exactly (addresses, phone numbers, names).
- Do NOT summarize.
- Do NOT omit any information.
- Keep formatting readable.

TEXT:
{context}

RUSSIAN:
""".strip()

    try:
        return translator_llm.invoke(translate_prompt).content
    except Exception:
        return context


def sanitize_metadata(meta: dict) -> dict:
    clean = {}
    for k, v in (meta or {}).items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean


def split_semantic_then_fallback(docs: List[Document]) -> List[Document]:
    try:
        return SemanticChunker(
            embedding=embedding_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=90,
        ).split_documents(docs)
    except Exception:
        splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
        return splitter.split_documents(docs)


def build_context(docs: List[Document], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    parts = []
    total = 0
    for d in docs:
        text = (d.page_content or "").strip()
        if not text:
            continue
        if total + len(text) > max_chars:
            remain = max_chars - total
            if remain > 200:
                parts.append(text[:remain])
            break
        parts.append(text)
        total += len(text)
    return "\n\n---\n\n".join(parts)


def build_history(history: Optional[List[ChatMessage]]) -> str:
    if not history:
        return ""

    recent = history[-MAX_HISTORY_TURNS:]
    lines: List[str] = []
    for m in recent:
        role = (m.role or "").strip().lower()
        content = (m.content or "").strip()
        if not content:
            continue
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        else:
            lines.append(f"{role.capitalize() if role else 'Message'}: {content}")

    text = "\n".join(lines).strip()
    if len(text) <= MAX_HISTORY_CHARS:
        return text
    return text[-MAX_HISTORY_CHARS:]


def doc_key(d: Document) -> str:
    m = d.metadata or {}
    if "doc_id" in m and "chunk_index" in m and "source" in m:
        return f'{m["doc_id"]}:{m["chunk_index"]}:{m["source"]}'
    return (m.get("source", "unknown") + ":" + str(hash(d.page_content)))


def rrf_merge(
    dense_docs: List[Document],
    sparse_docs: List[Document],
    k_final: int = K_FINAL,
    rrf_k: int = RRF_K,
    dense_weight: float = 0.3,
    sparse_weight: float = 0.7,
) -> List[Document]:
    scores: Dict[str, float] = {}
    by_key: Dict[str, Document] = {}

    def add(docs: List[Document], weight: float):
        for rank, d in enumerate(docs, start=1):
            key = doc_key(d)
            by_key[key] = d
            scores[key] = scores.get(key, 0.0) + weight * (1.0 / (rrf_k + rank))

    add(dense_docs, weight=dense_weight)
    add(sparse_docs, weight=sparse_weight)

    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [by_key[k] for k, _ in merged[:k_final]]


async def rebuild_bm25(app: FastAPI) -> None:
    vectordb: Chroma = app.state.vectordb
    raw = vectordb._collection.get(include=["documents", "metadatas"])
    docs = [
        Document(page_content=t, metadata=(m or {}))
        for t, m in zip(raw.get("documents", []), raw.get("metadatas", []))
        if t and t.strip()
    ]
    if docs:
        bm25 = BM25Retriever.from_documents(docs)
        bm25.k = K_SPARSE
        app.state.bm25 = bm25
    else:
        app.state.bm25 = None


def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-12)


def rerank_by_embedding(query: str, docs: List[Document], top_k: int) -> List[Document]:
    q = embedding_model.embed_query(query)
    doc_vecs = embedding_model.embed_documents([d.page_content for d in docs])
    scored = [(cosine(q, v), d) for v, d in zip(doc_vecs, docs)]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_k]]


def is_fact_question(q: str) -> bool:
    ql = (q or "").lower()

    if any(ch.isdigit() for ch in (q or "")):
        return True

    patterns = [
        r"\bhow many\b",
        r"\bhow high\b",
        r"\bwhat(?:'s| is) the name\b",
        r"\bwhich\b",
        r"\bwhere\b",
        r"\blocated\b",
        r"\byear\b",
        r"\bcentury\b",
        r"\bmeters?\b",
        r"\bmetres?\b",
        r"\bkm\b",
        r"\baltitude\b",
        r"\bheight\b",
    ]
    return any(re.search(p, ql) for p in patterns)


# =========================
# Embassy / Hospital intent
# =========================
def is_passport_or_embassy_topic(q: str) -> bool:
    ql = (q or "").lower()
    keywords = ["passport", "visa", "embassy", "consulate", "lost passport", "stolen passport"]
    return any(k in ql for k in keywords)


def needs_embassy_help(q: str) -> bool:
    ql = (q or "").lower()
    keywords = [
        "lost passport",
        "lost my passport",
        "passport lost",
        "stolen passport",
        "passport stolen",
        "my passport was stolen",
        "i lost my passport",
        "i have lost my passport",
        "i lost passport",
        "embassy",
        "consulate",
        "emergency",
        "urgent",
        "robbed",
        "theft",
        "stolen",
        "visa problem",
        "need help",
        "lost documents",
        "lost my id",
    ]
    if any(k in ql for k in keywords):
        return True

    ru_keywords = [
        "потерял паспорт",
        "потеряла паспорт",
        "украли паспорт",
        "посольство",
        "консульство",
        "срочно",
        "экстренно",
    ]
    if any(k in ql for k in ru_keywords):
        return True

    return False


def needs_hospital_help(q: str) -> bool:
    ql = (q or "").lower()
    keywords = [
        "hospital",
        "clinic",
        "doctor",
        "medical",
        "i am sick",
        "i'm sick",
        "i feel sick",
        "i am ill",
        "i'm ill",
        "fever",
        "pain",
        "injury",
        "injured",
        "need a doctor",
        "need hospital",
        "where is a hospital",
        "where can i see a doctor",
        "emergency room",
        "er",
    ]
    if any(k in ql for k in keywords):
        return True

    ru_keywords = [
        "больница",
        "клиника",
        "врач",
        "мне плохо",
        "я болен",
        "температура",
        "боль",
        "травма",
        "скорая",
    ]
    if any(k in ql for k in ru_keywords):
        return True

    return False


# =========================
# Hospital city parsing / normalization
# =========================
CITY_ALIAS = {
    "dushanbe": "dushanbe",
    "душанбе": "dushanbe",
    "khujand": "khujand",
    "худжанд": "khujand",
    "bokhtar": "bokhtar",
    "бохтар": "bokhtar",
    "kulob": "kulob",
    "куляб": "kulob",
    "khorog": "khorog",
    "хорог": "khorog",
    "panjakent": "panjakent",
    "пенджикент": "panjakent",
}


def normalize_city(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"[^a-z\u0400-\u04FF ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def canonical_city(text: str) -> str:
    n = normalize_city(text)
    return CITY_ALIAS.get(n, n)


def extract_city(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.strip()

    patterns = [
        r"\bI am in\s+([A-Za-z][A-Za-z \-]{1,40})\b",
        r"\bI'm in\s+([A-Za-z][A-Za-z \-]{1,40})\b",
        r"\bI’m in\s+([A-Za-z][A-Za-z \-]{1,40})\b",
        r"\bin\s+([A-Za-z][A-Za-z \-]{1,40})\b",
        r"\bat\s+([A-Za-z][A-Za-z \-]{1,40})\b",
        r"\bcity:\s*([A-Za-z][A-Za-z \-]{1,40})\b",
    ]
    for p in patterns:
        m = re.search(p, t, flags=re.IGNORECASE)
        if m:
            cand = m.group(1).strip(" .,!?:;\"'")
            if cand:
                return cand
    return None


def was_city_requested(history: Optional[List[ChatMessage]]) -> bool:
    if not history:
        return False
    recent = history[-MAX_HISTORY_TURNS:]
    for m in reversed(recent):
        if (m.role or "").strip().lower() != "assistant":
            continue
        c = (m.content or "").lower()
        if "which city" in c or "what city" in c or "city are you in" in c or "currently in" in c:
            return True
        if "в каком городе" in c:
            return True
        break
    return False


def find_city_from_history(history: Optional[List[ChatMessage]]) -> Optional[str]:
    if not history:
        return None
    recent = history[-MAX_HISTORY_TURNS:]
    for m in reversed(recent):
        if (m.role or "").strip().lower() != "user":
            continue
        city = extract_city(m.content or "")
        if city:
            return city
    return None


def extract_city_fallback_if_awaiting(history: Optional[List[ChatMessage]], text: str) -> Optional[str]:
    if not text:
        return None
    if not was_city_requested(history):
        return None

    t = text.strip()
    if len(t) > 40:
        return None
    if any(ch.isdigit() for ch in t):
        return None
    if not re.fullmatch(r"[A-Za-z\u0400-\u04FF \-]+", t):
        return None
    return t.strip(" -")


def parse_hospital_record(text: str) -> Dict[str, str]:
    """
    One-line record expected:
    Type: Hospital City: ... Name: ... Address: ... Phone: ...
    """
    t = (text or "").strip()
    out: Dict[str, str] = {}

    def pick_between(start_key: str, end_key: Optional[str]) -> Optional[str]:
        if end_key:
            m = re.search(
                rf"{re.escape(start_key)}\s*:\s*(.*?)\s*(?={re.escape(end_key)}\s*:)",
                t,
                flags=re.IGNORECASE,
            )
        else:
            m = re.search(rf"{re.escape(start_key)}\s*:\s*(.*)", t, flags=re.IGNORECASE)
        return m.group(1).strip() if m else None

    typ = pick_between("Type", "City")
    city = pick_between("City", "Name")
    name = pick_between("Name", "Address")
    address = pick_between("Address", "Phone")
    phone = pick_between("Phone", None)

    if typ:
        out["Type"] = typ
    if city:
        out["City"] = city
    if name:
        out["Name"] = name
        out["HospitalName"] = name
    if address:
        out["Address"] = address
    if phone:
        out["Phone"] = phone

    return out


def normalize_hospital_key(name: str) -> str:
    if not name:
        return ""
    t = name.lower()
    t = re.sub(r"\(.*?\)", "", t)
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# =========================
# Embassy nationality parsing / matching
# =========================
def normalize_nationality(nat: str) -> str:
    n = (nat or "").strip().lower()

    mapping = {
        # US
        "usa": "united states",
        "us": "united states",
        "u.s.": "united states",
        "u.s.a.": "united states",
        "united states": "united states",
        "united states of america": "united states",
        "america": "united states",
        "american": "united states",
        "сша": "united states",
        "соединенные штаты": "united states",
        "соединённые штаты": "united states",

        # Korea
        "korea": "republic of korea",
        "south korea": "republic of korea",
        "republic of korea": "republic of korea",
        "rok": "republic of korea",
        "korean": "republic of korea",
        "южная корея": "republic of korea",
        "корея": "republic of korea",

        # Russia
        "russia": "russian federation",
        "russian": "russian federation",
        "russian federation": "russian federation",
        "rf": "russian federation",
        "россия": "russian federation",
        "российская федерация": "russian federation",

        # Germany
        "germany": "germany",
        "german": "germany",
        "deutschland": "germany",
        "германия": "germany",
    }

    return mapping.get(n, n)


def extract_nationality(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.strip()

    patterns = [
        r"\bI am\s+([A-Za-z][A-Za-z \-]{1,40})\b",
        r"\bI'm\s+([A-Za-z][A-Za-z \-]{1,40})\b",
        r"\bI’m\s+([A-Za-z][A-Za-z \-]{1,40})\b",
        r"\bmy nationality is\s+([A-Za-z][A-Za-z \-]{1,40})\b",
        r"\bmy country is\s+([A-Za-z][A-Za-z \-]{1,40})\b",
        r"\bnationality:\s*([A-Za-z][A-Za-z \-]{1,40})\b",
        r"\bcitizen of\s+([A-Za-z][A-Za-z \-]{1,40})\b",
    ]

    for p in patterns:
        m = re.search(p, t, flags=re.IGNORECASE)
        if m:
            cand = m.group(1).strip(" .,!?:;\"'")
            if cand:
                return cand
    return None


def was_nationality_requested(history: Optional[List[ChatMessage]]) -> bool:
    if not history:
        return False
    recent = history[-MAX_HISTORY_TURNS:]
    for m in reversed(recent):
        if (m.role or "").strip().lower() != "assistant":
            continue
        c = (m.content or "").lower()
        if "nationality" in c or "what is your nationality" in c or "tell me your nationality" in c:
            return True
        if "гражданство" in c or "какое у вас гражданство" in c:
            return True
        break
    return False


def find_nationality_from_history(history: Optional[List[ChatMessage]]) -> Optional[str]:
    if not history:
        return None
    recent = history[-MAX_HISTORY_TURNS:]
    for m in reversed(recent):
        if (m.role or "").strip().lower() != "user":
            continue
        nat = extract_nationality(m.content or "")
        if nat:
            return nat
    return None


def embassy_matches_country(doc: Document, nat_l: str) -> bool:
    txt = doc.page_content or ""

    m = re.search(r"Country\s*:\s*([^\n\r]+)", txt, flags=re.IGNORECASE)
    if m:
        country = m.group(1).strip().lower()
        if country == nat_l:
            return True

    m = re.search(r"CountryAlias\s*:\s*([^\n\r]+)", txt, flags=re.IGNORECASE)
    if m:
        aliases = [a.strip().lower() for a in m.group(1).split(",")]
        if nat_l in aliases:
            return True

    return False


def parse_embassy_record(text: str) -> Dict[str, str]:
    t = text or ""
    out: Dict[str, str] = {}

    def pick(key: str) -> Optional[str]:
        m = re.search(rf"{re.escape(key)}\s*:\s*([^\n\r]+)", t, flags=re.IGNORECASE)
        return m.group(1).strip() if m else None

    for k in ["EmbassyName", "City", "MapLocation", "Phone"]:
        v = pick(k)
        if v:
            out[k] = v
    return out


# =========================
# RU hard maps (optional, as in your code)
# =========================
EMBASSY_RU = {
    "united states": {
        "name": "Посольство Соединённых Штатов Америки в Таджикистане",
        "city": "Душанбе",
        "address": "Республика Таджикистан, г. Душанбе, проспект Исмоили Сомони, дом 109А",
        "phone": "+992 37 229 2000",
    },
    "russian federation": {
        "name": "Посольство Российской Федерации в Таджикистане",
        "city": "Душанбе",
        "address": "Республика Таджикистан, г. Душанбе, ул. Абу Али ибн Сино, дом 29/31",
        "phone": "+992 37 235 9827",
    },
    "republic of korea": {
        "name": "Посольство Республики Корея в Таджикистане",
        "city": "Душанбе",
        "address": "Республика Таджикистан, г. Душанбе, ул. Льва Толстого, дом 9",
        "phone": "+992 37 229 3001",
    },
}

HOSPITAL_RU_MAP = {
    "shifobakhsh national medical center": {
        "name": "Национальный медицинский центр «Шифобахш»",
        "city": "Душанбе",
        "address": "Республика Таджикистан, г. Душанбе, ул. Ибн Сино, 59",
    },
    "istiqlol medical complex": {
        "name": "Медицинский комплекс «Истиклол»",
        "city": "Душанбе",
        "address": "Республика Таджикистан, г. Душанбе, проспект Немата Карабаева, 61",
    },
    "republican clinical cardiology center": {
        "name": "Республиканский клинический кардиологический центр",
        "city": "Душанбе",
        "address": "Республика Таджикистан, г. Душанбе, проспект Ибн Сино, 59",
    },
    "children s hospital of infectious diseases": {
        "name": "Детская инфекционная больница",
        "city": "Душанбе",
        "address": "Республика Таджикистан, г. Душанбе, ул. Шероз, 20",
    },
    "dushanbe city clinical hospital no 1": {
        "name": "Городская клиническая больница №1 города Душанбе",
        "city": "Душанбе",
        "address": "Республика Таджикистан, г. Душанбе",
    },
    "dushanbe city clinical hospital no 3": {
        "name": "Городская клиническая больница №3 города Душанбе",
        "city": "Душанбе",
        "address": "Республика Таджикистан, г. Душанбе",
    },
    "sughd regional clinical hospital": {
        "name": "Согдийская областная клиническая больница",
        "city": "Худжанд",
        "address": "Республика Таджикистан, г. Худжанд, ул. Рахмона Набиева, 111",
    },
    "khujand city hospital": {
        "name": "Городская больница города Худжанд",
        "city": "Худжанд",
        "address": "Республика Таджикистан, г. Худжанд",
    },
    "aga khan medical centre khorog": {
        "name": "Медицинский центр Ага Хана в Хороге",
        "city": "Хорог",
        "address": "Республика Таджикистан, г. Хорог, ул. Шогуниева, 1",
    },
    "khorog central regional hospital": {
        "name": "Центральная региональная больница города Хорог",
        "city": "Хорог",
        "address": "Республика Таджикистан, г. Хорог",
    },
    "bokhtar city hospital": {
        "name": "Городская больница города Бохтар",
        "city": "Бохтар",
        "address": "Республика Таджикистан, г. Бохтар",
    },
    "kulob city hospital": {
        "name": "Городская больница города Куляб",
        "city": "Куляб",
        "address": "Республика Таджикистан, г. Куляб",
    },
    "panjakent city hospital": {
        "name": "Городская больница города Пенджикент",
        "city": "Пенджикент",
        "address": "Республика Таджикистан, г. Пенджикент",
    },
}


# =========================
# 8) FastAPI Lifespan (캐시/락 준비)
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(DB_DIR, exist_ok=True)
    os.makedirs(STATIC_DIR, exist_ok=True)

    app.state.vectordb = get_vectorstore()
    app.state.bm25 = None

    app.state.write_lock = asyncio.Lock()
    app.state.rebuild_lock = asyncio.Lock()

    async with app.state.rebuild_lock:
        await rebuild_bm25(app)

    print(">> Server Started")
    yield
    print(">> Server Shutdown")


app = FastAPI(
    title="Tajikistan RAG API (Ollama Embeddings + Gemma2)",
    lifespan=lifespan,
)


# =========================
# Frontend (Static)
# =========================
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# =========================
# 9) Ingest API
# =========================
@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    doc_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{file.filename}")

    async with app.state.write_lock:
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
            d.metadata = d.metadata or {}
            d.metadata.update(
                {
                    "doc_id": doc_id,
                    "source": file.filename,
                    "chunk_index": i,
                }
            )

            page = d.metadata.get("page")
            if page is None:
                d.metadata.pop("page", None)
            else:
                try:
                    d.metadata["page"] = int(page)
                except Exception:
                    d.metadata.pop("page", None)

        chunks = filter_complex_metadata(chunks)
        for d in chunks:
            d.metadata = sanitize_metadata(d.metadata)

        vectordb: Chroma = app.state.vectordb
        vectordb.add_documents(chunks)

    async with app.state.rebuild_lock:
        await rebuild_bm25(app)

    return {"message": f"Ingested {len(chunks)} chunks", "doc_id": doc_id}


# =========================
# 10) Chat API (Hybrid RAG + Hospital + Embassy + Travel)
# =========================
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    start = time.time()
    vectordb: Chroma = app.state.vectordb
    bm25: Optional[BM25Retriever] = app.state.bm25

    history_text = build_history(req.history)

    # =========================
    # Hospital flow
    # =========================
    hospital_intent = (
        (not is_passport_or_embassy_topic(req.question))
        and (needs_hospital_help(req.question) or was_city_requested(req.history))
    )

    if hospital_intent:
        city = (
            extract_city(req.question)
            or find_city_from_history(req.history)
            or extract_city_fallback_if_awaiting(req.history, req.question)
        )

        if city is None:
            answer = (
                "Понимаю. Чтобы подсказать больницу, скажите, пожалуйста, в каком городе вы находитесь "
                "(например: Душанбе, Худжанд, Бохтар)."
                if is_russian_with_history(req.question, req.history)
                else
                "I understand. To suggest a hospital, which city are you currently in "
                "(e.g., Dushanbe, Khujand, Bokhtar)?"
            )
            return ChatResponse(answer=answer, time_taken=time.time() - start, sources=[])

        city_norm = canonical_city(city)

        search_query = f"Type: Hospital City: {city} phone address"

        dense_docs = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": K_DENSE},
        ).invoke(search_query)

        sparse_docs = bm25.invoke(search_query) if bm25 else []

        merged_docs = rrf_merge(
            dense_docs,
            sparse_docs,
            k_final=50,
            rrf_k=RRF_K,
            dense_weight=0.3,
            sparse_weight=0.7,
        )

        def extract_hospital_records_from_chunk(text: str) -> List[str]:
            if not text:
                return []
            pattern = r"Type:\s*Hospital\s+City:\s*.*?(?=(?:Type:\s*Hospital\s+City:)|$)"
            matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            return [re.sub(r"\s+", " ", m).strip() for m in matches if m.strip()]

        unique_hospitals: Dict[str, Dict[str, str]] = {}
        used_docs: List[Document] = []

        for d in merged_docs:
            records = extract_hospital_records_from_chunk(d.page_content)
            for rec in records:
                fields = parse_hospital_record(rec)

                doc_city_norm = canonical_city(fields.get("City", ""))
                if doc_city_norm != city_norm:
                    continue

                name = (fields.get("HospitalName") or fields.get("Name") or "").strip()
                if not name:
                    continue

                unique_hospitals[name.lower()] = fields
                used_docs.append(d)

        if not unique_hospitals:
            answer = (
                f"У меня нет информации о больницах в городе {city}."
                if is_russian_with_history(req.question, req.history)
                else f"I don’t have hospital information for {city}."
            )
            return ChatResponse(answer=answer, time_taken=time.time() - start, sources=[])

        hospital_blocks: List[str] = []
        is_ru = is_russian_with_history(req.question, req.history)

        for key in sorted(unique_hospitals.keys()):
            f = unique_hospitals[key]
            name_en = f.get("HospitalName", "N/A")
            address_en = f.get("Address", "N/A")
            phone = f.get("Phone", "N/A")

            if is_ru:
                key_norm = normalize_hospital_key(name_en)
                ru = HOSPITAL_RU_MAP.get(key_norm)

                if not ru:
                    for k_norm, v in HOSPITAL_RU_MAP.items():
                        if k_norm and (k_norm in key_norm or key_norm in k_norm):
                            ru = v
                            break

                if not ru:
                    continue

                name = ru["name"]
                address = ru["address"]
                city_out = ru["city"]
            else:
                name = name_en
                address = address_en
                city_out = city

            hospital_blocks.append(
                f"HospitalName: {name}\nCity: {city_out}\nAddress: {address}\nPhone: {phone}".strip()
            )

        if is_ru and not hospital_blocks:
            answer = f"У меня нет подтверждённых данных о больницах в городе {city} на русском языке."
            return ChatResponse(answer=answer, time_taken=time.time() - start, sources=[])

        context_for_llm = "\n\n".join(hospital_blocks)

        chain = hospital_prompt | llm | StrOutputParser()
        language = "Russian" if is_ru else "English"

        answer_raw = chain.invoke({"context": context_for_llm, "language": language})
        final_answer = (answer_raw or "").strip()

        unique_docs: Dict[str, Document] = {}
        for d in used_docs:
            unique_docs[doc_key(d)] = d

        sources = [
            SourceInfo(
                file=(d.metadata or {}).get("source", "unknown"),
                page=(d.metadata or {}).get("page"),
                snippet=d.page_content,
            )
            for d in unique_docs.values()
        ]

        return ChatResponse(answer=final_answer, time_taken=time.time() - start, sources=sources)

    # =========================
    # Embassy flow  (핵심: doc 1개만 뽑고, Context도 1개만)
    # =========================
    embassy_intent = needs_embassy_help(req.question) or was_nationality_requested(req.history)

    if is_russian_with_history(req.question, req.history):
        if "паспорт" in (req.question or "").lower():
            embassy_intent = True

    if embassy_intent:
        nat_in_question = extract_nationality(req.question)
        nat_in_history = find_nationality_from_history(req.history)
        nationality = nat_in_question or nat_in_history

        if nationality is None and was_nationality_requested(req.history):
            t = (req.question or "").strip()
            if 1 <= len(t) <= 40:
                nationality = t

        if nationality is None:
            answer = (
                "Понимаю — это срочная ситуация.\n"
                "Чтобы подсказать правильное посольство в Таджикистане, "
                "скажите, пожалуйста, ваше гражданство "
                "(например: США, Южная Корея, Германия)."
                if is_russian_with_history(req.question, req.history)
                else
                "I’m sorry you’re dealing with this.\n"
                "To point you to the correct embassy in Tajikistan, "
                "what is your nationality (e.g., USA, Korea, Germany)?"
            )
            return ChatResponse(answer=answer, time_taken=time.time() - start, sources=[])

        nat_l = normalize_nationality(nationality)

        search_query = f"embassy {nat_l} Tajikistan Dushanbe phone address"

        dense_docs = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": K_DENSE},
        ).invoke(search_query)

        sparse_docs = []
        if bm25 is not None:
            bm25.k = K_SPARSE
            sparse_docs = bm25.invoke(search_query)

        final_candidates = rrf_merge(
            dense_docs,
            sparse_docs,
            k_final=10,
            rrf_k=RRF_K,
            dense_weight=0.3,
            sparse_weight=0.7,
        )

        embassy_only: List[Document] = []
        for d in final_candidates:
            src = ((d.metadata or {}).get("source") or "").lower()
            txt = (d.page_content or "").lower()
            if "type: embassy" in txt and "embassy" in src:
                embassy_only.append(d)
        if embassy_only:
            final_candidates = embassy_only

        embassy_candidates = [
            d for d in final_candidates
            if "type: embassy" in (d.page_content or "").lower()
        ]

        matched = [
            d for d in embassy_candidates
            if embassy_matches_country(d, nat_l)
        ]

        # 🔒 여기서 강제 1개 확정
        matched = matched[:1]

        if not matched:
            answer = (
                "У меня нет информации о посольстве этой страны в моих документах."
                if is_russian_with_history(req.question, req.history)
                else "I don’t have embassy information for that country in my documents."
            )
            return ChatResponse(answer=answer, time_taken=time.time() - start, sources=[])

        # ✅ 여기서 doc 1개 확정
        doc = matched[0]

        fields = parse_embassy_record(doc.page_content)

        embassy_name_en = fields.get("EmbassyName", "N/A")
        city_en = fields.get("City", "N/A")
        address_en = fields.get("MapLocation", "N/A")
        phone = fields.get("Phone", "N/A")

        if is_russian_with_history(req.question, req.history):
            ru = EMBASSY_RU.get(nat_l)
            if not ru:
                return ChatResponse(
                    answer="У меня нет подтверждённых данных об этом посольстве на русском языке.",
                    time_taken=time.time() - start,
                    sources=[]
                )
            embassy_name_final = ru["name"]
            city_final = ru["city"]
            address_final = ru["address"]
            phone = ru["phone"]
        else:
            embassy_name_final = embassy_name_en
            city_final = city_en
            address_final = address_en

        # ✅ Context도 딱 1개 레코드만
        context_for_llm = f"""
EmbassyName: {embassy_name_final}
City: {city_final}
Address: {address_final}
Phone: {phone}
""".strip()

        chain = embassy_prompt | llm | StrOutputParser()
        language = "Russian" if is_russian_with_history(req.question, req.history) else "English"

        answer_raw = chain.invoke({"context": context_for_llm, "language": language})
        final_answer = (answer_raw or "").strip()

        sources = [
            SourceInfo(
                file=(doc.metadata or {}).get("source", "unknown"),
                page=(doc.metadata or {}).get("page"),
                snippet=doc.page_content,
            )
        ]

        return ChatResponse(answer=final_answer, time_taken=time.time() - start, sources=sources)

    # =========================
    # Travel RAG flow (기존)
    # =========================
    fact = is_fact_question(req.question)

    if fact:
        dense_search_type = "similarity"
        dense_kwargs = {"k": K_DENSE}
        dense_weight, sparse_weight = 0.3, 0.7
        rerank = True
    else:
        dense_search_type = "mmr"
        dense_kwargs = {"k": K_DENSE, "fetch_k": FETCH_K, "lambda_mult": 0.7}
        dense_weight, sparse_weight = 0.6, 0.4
        rerank = True

    dense_docs = vectordb.as_retriever(
        search_type=dense_search_type,
        search_kwargs=dense_kwargs,
    ).invoke(req.question)

    sparse_docs: List[Document] = []
    if bm25 is not None:
        bm25.k = K_SPARSE
        sparse_docs = bm25.invoke(req.question)

    final_candidates = rrf_merge(
        dense_docs,
        sparse_docs,
        k_final=20,
        rrf_k=RRF_K,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
    )

    if rerank:
        final_docs = rerank_by_embedding(req.question, final_candidates, top_k=K_FINAL)
    else:
        final_docs = final_candidates[:K_FINAL]

    if not final_docs:
        answer = (
            "У меня нет информации об этом в моих документах."
            if is_russian_with_history(req.question, req.history)
            else "I don't have information about that in my documents."
        )
        return ChatResponse(answer=answer, time_taken=time.time() - start, sources=[])

    context = build_context(final_docs, max_chars=MAX_CONTEXT_CHARS)
    context = translate_context_if_needed(context, req.question)

    sources = [
        SourceInfo(
            file=(d.metadata or {}).get("source", "unknown"),
            page=(d.metadata or {}).get("page"),
            snippet=d.page_content,
        )
        for d in final_docs
    ]

    chain = prompt | llm | StrOutputParser()
    lang_hint = "Russian" if is_russian_with_history(req.question, req.history) else "English"

    answer = chain.invoke(
        {"context": context, "question": f"[Language: {lang_hint}]\n{req.question}", "history": history_text}
    )
    answer = translate_answer_if_needed(answer, req.question)
    return ChatResponse(answer=answer, time_taken=time.time() - start, sources=sources)


# =========================
# 11) List Documents API
# =========================
@app.get("/documents")
async def list_documents():
    vectordb: Chroma = app.state.vectordb
    data = vectordb._collection.get(include=["metadatas"])

    documents: Dict[str, str] = {}
    for meta in data.get("metadatas", []):
        if meta and "doc_id" in meta and "source" in meta:
            documents[meta["doc_id"]] = meta["source"]

    return {"documents": documents}


# =========================
# 12) Delete Document API
# =========================
@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    async with app.state.write_lock:
        vectordb: Chroma = app.state.vectordb
        data = vectordb._collection.get(where={"doc_id": doc_id})
        ids = data.get("ids", [])
        if not ids:
            raise HTTPException(status_code=404, detail="Document not found")

        vectordb._collection.delete(ids=ids)

    async with app.state.rebuild_lock:
        await rebuild_bm25(app)

    return {"deleted_chunks": len(ids), "doc_id": doc_id}


# =========================
# 13) Clear All Documents API
# =========================
@app.delete("/documents")
async def clear_all_documents():
    try:
        async with app.state.write_lock:
            if os.path.exists(DB_DIR):
                shutil.rmtree(DB_DIR)
            os.makedirs(DB_DIR, exist_ok=True)

            if os.path.exists(UPLOAD_DIR):
                shutil.rmtree(UPLOAD_DIR)
            os.makedirs(UPLOAD_DIR, exist_ok=True)

            app.state.vectordb = get_vectorstore()
            app.state.bm25 = None

        return {"message": "All documents cleared (DB + uploads)."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# 14) Run
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
