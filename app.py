# -*- coding: utf-8 -*-
import os
from pathlib import Path
import sys
from collections import OrderedDict
import hashlib
import re
import shutil
import sqlite3

from dotenv import load_dotenv
import pytesseract
from PIL import Image, ImageEnhance, ImageOps

from langchain_community.document_loaders import (
    DirectoryLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:1b")
EMBEDDING_MODEL = "nomic-embed-text"
DATA_DIR = Path("data")
CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "ollama_rag_demo"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
DOCUMENT_EXTENSIONS = {".txt", ".pdf", ".docx"} | IMAGE_EXTENSIONS
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "3"))
RETRIEVER_FETCH_K = int(os.getenv("RETRIEVER_FETCH_K", str(max(RETRIEVER_K * 4, 8))))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.18"))
KEYWORD_RESULT_LIMIT = int(os.getenv("KEYWORD_RESULT_LIMIT", "5"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "2500"))
NUM_PREDICT = int(os.getenv("NUM_PREDICT", "512"))
NUM_CTX = int(os.getenv("NUM_CTX", "2048"))
ANSWER_CACHE_SIZE = int(os.getenv("ANSWER_CACHE_SIZE", "32"))
REBUILD_VECTORSTORE = os.getenv("REBUILD_VECTORSTORE", "").lower() in {"1", "true", "yes"}
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
OUT_OF_CONTEXT_ANSWER = "I don't have enough information in the indexed sources to answer that question."
FOLLOW_UP_REFERENCES = {
    "he",
    "her",
    "him",
    "his",
    "it",
    "one",
    "she",
    "that",
    "their",
    "them",
    "they",
    "this",
    "those",
}
QUESTION_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "can",
    "did",
    "does",
    "for",
    "from",
    "give",
    "how",
    "into",
    "is",
    "list",
    "many",
    "me",
    "of",
    "on",
    "or",
    "show",
    "tell",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "won",
    "year",
    "number",
}

answer_cache = OrderedDict()

tesseract_cmd = os.getenv("TESSERACT_CMD")
if tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
else:
    default_tesseract_path = Path("C:/Program Files/Tesseract-OCR/tesseract.exe")
    if default_tesseract_path.exists():
        pytesseract.pytesseract.tesseract_cmd = str(default_tesseract_path)


def format_docs(docs, max_chars=MAX_CONTEXT_CHARS):
    """Keep retrieved context small so Ollama can start generating sooner."""
    parts = []
    used_chars = 0

    for index, doc in enumerate(docs):
        text = " ".join(doc.page_content.split())
        if not text:
            continue

        if index > 0 and looks_like_direct_answer(parts[0]):
            break

        remaining = max_chars - used_chars
        if remaining <= 0:
            break

        parts.append(text[:remaining])
        used_chars += len(parts[-1])

    return "\n\n".join(parts)


def looks_like_direct_answer(text):
    lower_text = text.lower()
    return (
        "10 examples" in lower_text
        or "examples of artificial intelligence" in lower_text
        or sum(keyword in lower_text for keyword in ("virtual assistants", "healthcare", "retail", "robotics")) >= 2
    )


def extract_image_text(image):
    """Use multiple OCR passes because infographic layouts often have columns."""
    width, height = image.size
    crops = [
        image,
        image.crop((0, height // 5, width // 2, height)),
        image.crop((width // 2, height // 5, width, height)),
    ]
    text_parts = []

    for crop in crops:
        gray = ImageOps.grayscale(crop)
        enhanced = ImageEnhance.Contrast(gray).enhance(1.5)
        enlarged = enhanced.resize((enhanced.width * 3, enhanced.height * 3))

        for config in ("", "--psm 3"):
            text = pytesseract.image_to_string(enlarged, config=config).strip()
            if text:
                text_parts.append(text)

    return "\n".join(dict.fromkeys(text_parts))


def load_ocr_documents(data_dir):
    """Extract text from image files and return LangChain documents."""
    ocr_docs = []

    for image_path in data_dir.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        if image_path.with_suffix(".txt").exists():
            continue

        try:
            with Image.open(image_path) as image:
                text = extract_image_text(image)
                # Ensure text is properly encoded as UTF-8
                text = text.encode('utf-8', errors='replace').decode('utf-8')
        except pytesseract.TesseractNotFoundError as exc:
            raise RuntimeError(
                "Tesseract OCR is not installed or is not on PATH. "
                "Install Tesseract, then set TESSERACT_CMD in .env if needed. "
                "Example: TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
            ) from exc
        except Exception as exc:
            try:
                print(f"Skipping OCR for {image_path}: {exc}", file=sys.stderr)
            except:
                pass
            continue

        if text:
            ocr_docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(image_path),
                        "loader": "pytesseract",
                    },
                )
            )
        else:
            try:
                print(f"No OCR text found in {image_path}", file=sys.stderr)
            except:
                pass

    return ocr_docs


def load_source_documents():
    docs = []

    text_loader = DirectoryLoader(
        str(DATA_DIR),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs.extend(text_loader.load())

    pdf_loader = DirectoryLoader(
        str(DATA_DIR),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
    )
    docs.extend(pdf_loader.load())

    docx_loader = DirectoryLoader(
        str(DATA_DIR),
        glob="**/*.docx",
        loader_cls=Docx2txtLoader,
    )
    docs.extend(docx_loader.load())

    docs.extend(load_ocr_documents(DATA_DIR))
    return docs


def file_signature(source):
    source_path = Path(source)
    if not source_path.exists():
        return ""

    stat = source_path.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def local_source_signatures(data_dir):   
    signatures = {}

    for source_path in data_dir.rglob("*"):
        if not source_path.is_file() or source_path.suffix.lower() not in DOCUMENT_EXTENSIONS:
            continue
        if source_path.suffix.lower() in IMAGE_EXTENSIONS and source_path.with_suffix(".txt").exists():
            continue

        signatures[str(source_path)] = file_signature(source_path)

    return signatures


def split_documents_with_ids(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    split_docs = text_splitter.split_documents(docs)
    chunk_counts = {}

    for doc in split_docs:
        source = doc.metadata.get("source", "unknown-source")
        page = doc.metadata.get("page", "")
        chunk_key = (source, page)
        chunk_index = chunk_counts.get(chunk_key, 0)
        chunk_counts[chunk_key] = chunk_index + 1

        content_hash = hashlib.sha1(doc.page_content.encode("utf-8", errors="replace")).hexdigest()[:16]
        raw_id = f"{source}|{page}|{chunk_index}|{content_hash}"
        chunk_id = hashlib.sha1(raw_id.encode("utf-8", errors="replace")).hexdigest()
        doc.metadata["chunk_id"] = chunk_id
        doc.metadata["source_signature"] = file_signature(source)

    return split_docs


def get_existing_vectorstore_state(vectorstore):
    collection = vectorstore._collection
    try:
        result = collection.get(include=["metadatas"])
    except Exception:
        return set(), set(), {}

    existing_ids = set(result.get("ids", []))
    legacy_sources = set()
    source_signatures = {}

    for metadata in result.get("metadatas", []):
        if not metadata:
            continue
        source = metadata.get("source")
        if source and not metadata.get("chunk_id"):
            legacy_sources.add(source)
        if source and metadata.get("source_signature"):
            source_signatures[source] = metadata.get("source_signature")

    return existing_ids, legacy_sources, source_signatures


def sync_vectorstore(vectorstore):
    print("Checking data folder for new documents...")
    existing_ids, legacy_sources, source_signatures = get_existing_vectorstore_state(vectorstore)
    local_signatures = local_source_signatures(DATA_DIR)
    changed_sources = [
        source
        for source, signature in local_signatures.items()
        if source not in legacy_sources and source_signatures.get(source) != signature
    ]

    if not changed_sources:
        print("No new or changed local files found. Vector store is up to date.")
        return vectorstore

    docs = load_source_documents()

    if not docs:
        print(f"No documents found in {DATA_DIR.resolve()}.")
        return vectorstore

    print(f"Loaded {len(docs)} documents. Splitting into chunks...")
    split_docs = split_documents_with_ids(docs)

    new_docs = []
    new_ids = []
    for doc in split_docs:
        source = doc.metadata.get("source")
        chunk_id = doc.metadata["chunk_id"]
        if chunk_id in existing_ids:
            continue
        if source in legacy_sources:
            continue
        if source_signatures.get(source) == doc.metadata.get("source_signature"):
            continue

        new_docs.append(doc)
        new_ids.append(chunk_id)

    if not new_docs:
        print("No new chunks found. Vector store is up to date.")
        return vectorstore

    print(f"Adding {len(new_docs)} new chunks to the existing vector store...")
    vectorstore.add_documents(new_docs, ids=new_ids)
    vectorstore.persist()
    print("Vector store sync complete.")
    return vectorstore


def chroma_db_is_usable():
    sqlite_path = CHROMA_DIR / "chroma.sqlite3"
    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        return False
    if not sqlite_path.exists() or sqlite_path.stat().st_size == 0:
        return False

    try:
        with sqlite3.connect(str(sqlite_path)) as connection:
            rows = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='collections'"
            ).fetchall()
    except sqlite3.DatabaseError:
        return False

    return bool(rows)


def rebuild_vectorstore(reason):
    if CHROMA_DIR.exists():
        print(f"{reason} Rebuilding Chroma DB from data/...")
        shutil.rmtree(CHROMA_DIR)
    return build_vectorstore()


embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL,
)


def build_vectorstore():
    print("Loading source documents...")
    docs = load_source_documents()

    if not docs:
        raise ValueError(f"No documents found in {DATA_DIR.resolve()}")

    print(f"Loaded {len(docs)} documents. Splitting into chunks...")
    split_docs = split_documents_with_ids(docs)
    print(f"Split into {len(split_docs)} chunks. Building vector store...")

    # Build vectorstore with batch processing and progress
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        ids=[doc.metadata["chunk_id"] for doc in split_docs],
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    print("Persisting vectorstore...")
    vectorstore.persist()
    print("Done!")
    return vectorstore


def load_vectorstore():
    if REBUILD_VECTORSTORE and CHROMA_DIR.exists():
        return rebuild_vectorstore("REBUILD_VECTORSTORE is enabled.")

    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        if not chroma_db_is_usable():
            return rebuild_vectorstore("Existing Chroma DB is missing required tables or is empty.")

        print("Using existing Chroma DB.")
        try:
            vectorstore = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=str(CHROMA_DIR),
            )
            return sync_vectorstore(vectorstore)
        except Exception as exc:
            if "no such table: collections" in str(exc).lower():
                return rebuild_vectorstore("Existing Chroma DB is corrupted.")
            raise

    print("No existing Chroma DB found. Building it from files in data/...")
    return build_vectorstore()


vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

llm = OllamaLLM(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.2,
    num_predict=NUM_PREDICT,
    num_ctx=NUM_CTX,
    top_k=10,
    top_p=0.7,
    keep_alive=600,
)

prompt = ChatPromptTemplate.from_template(
    """Answer only using the context. If the context does not directly answer the question, say exactly: "I don't have enough information in the indexed sources to answer that question."
Do not explain what unrelated context is about.
Be clear and complete, with enough detail to directly answer the question. If the answer is a number, include the number and what it counts. If the user asks for a numbered list, include all visible items from the context.

Context:
{context}

Question: {question}
Answer:"""
)


def get_cached_answer(question):
    cached_answer = answer_cache.get(question)
    if cached_answer is None:
        return None

    answer_cache.move_to_end(question)
    return cached_answer


def cache_answer(question, answer):
    if ANSWER_CACHE_SIZE <= 0:
        return

    answer_cache[question] = answer
    answer_cache.move_to_end(question)

    while len(answer_cache) > ANSWER_CACHE_SIZE:
        answer_cache.popitem(last=False)


def build_prompt(question):
    docs = get_relevant_documents(question)
    context = format_docs(docs)
    return prompt.format(context=context, question=question)


def get_relevant_documents(question):
    docs = find_matching_text_documents(question)
    if docs:
        return docs

    candidates = []
    candidates.extend(get_vector_documents_with_scores(question))
    candidates.extend(get_keyword_documents(question))
    ranked_docs = merge_and_rank_documents(candidates)
    return ranked_docs[:RETRIEVER_K]


def get_vector_documents_with_scores(question):
    try:
        scored_docs = vectorstore.similarity_search_with_relevance_scores(question, k=RETRIEVER_FETCH_K)
    except Exception:
        return [
            add_retrieval_metadata(doc, retrieval_type="vector", score=0.0)
            for doc in retriever.invoke(question)
        ]

    docs = []
    for doc, score in scored_docs:
        if score is not None and score < SIMILARITY_THRESHOLD:
            continue
        docs.append(add_retrieval_metadata(doc, retrieval_type="vector", score=score or 0.0))

    return docs


def add_retrieval_metadata(doc, retrieval_type, score):
    metadata = dict(doc.metadata or {})
    metadata["retrieval_type"] = retrieval_type
    metadata["retrieval_score"] = round(float(score), 4)
    return Document(page_content=doc.page_content, metadata=metadata)


def get_keyword_documents(question):
    query_terms = question_keywords(question)
    if not query_terms:
        return []

    try:
        result = vectorstore._collection.get(include=["documents", "metadatas"])
    except Exception:
        return []

    matches = []
    documents = result.get("documents", [])
    metadatas = result.get("metadatas", [])

    for page_content, metadata in zip(documents, metadatas):
        if not page_content:
            continue

        score = keyword_score(question, query_terms, page_content, metadata or {})
        if score <= 0:
            continue

        scored_metadata = dict(metadata or {})
        scored_metadata["retrieval_type"] = "keyword"
        scored_metadata["retrieval_score"] = round(score, 4)
        matches.append(Document(page_content=page_content, metadata=scored_metadata))

    matches.sort(key=lambda doc: doc.metadata.get("retrieval_score", 0), reverse=True)
    return matches[:KEYWORD_RESULT_LIMIT]


def keyword_score(question, query_terms, page_content, metadata):
    haystack = " ".join(
        [
            page_content,
            str(metadata.get("source", "")),
            str(metadata.get("title", "")),
        ]
    ).lower()
    text_terms = set(re.findall(r"[a-zA-Z0-9]+", haystack))
    overlap = len(query_terms & text_terms)
    phrase_bonus = 0

    for phrase in important_phrases(question):
        if phrase in haystack:
            phrase_bonus += 2

    return overlap + phrase_bonus


def important_phrases(question):
    terms = meaningful_words(question)
    phrases = []
    lowered_question = question.lower()

    for first in range(len(terms)):
        for second in range(first + 1, min(first + 4, len(terms))):
            phrase = " ".join(terms[first : second + 1])
            if phrase in lowered_question:
                phrases.append(phrase)

    return phrases


def meaningful_words(text):
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [word for word in words if len(word) > 2 and word not in QUESTION_STOPWORDS]


def document_key(doc):
    metadata = doc.metadata or {}
    if metadata.get("chunk_id"):
        return metadata["chunk_id"]

    source = metadata.get("source", "")
    content_hash = hashlib.sha1(doc.page_content.encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"{source}:{content_hash}"


def merge_and_rank_documents(docs):
    merged = {}

    for doc in docs:
        key = document_key(doc)
        existing = merged.get(key)
        if existing is None:
            merged[key] = doc
            continue

        existing_score = existing.metadata.get("retrieval_score", 0)
        new_score = doc.metadata.get("retrieval_score", 0)
        if new_score > existing_score:
            merged[key] = doc

        if existing.metadata.get("retrieval_type") != doc.metadata.get("retrieval_type"):
            merged[key].metadata["retrieval_type"] = "hybrid"

    return sorted(
        merged.values(),
        key=lambda doc: (
            1 if doc.metadata.get("retrieval_type") == "hybrid" else 0,
            doc.metadata.get("retrieval_score", 0),
        ),
        reverse=True,
    )


def normalize_history(history):
    if not history:
        return []

    normalized = []
    for message in history[-6:]:
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            normalized.append({"role": role, "content": content[:600]})

    return normalized


def is_follow_up(question):
    words = re.findall(r"[a-zA-Z0-9]+", question.lower())

    if not words:
        return False

    word_set = set(words)
    if word_set & FOLLOW_UP_REFERENCES:
        return True

    short_follow_ups = {
        ("how", "many"),
        ("how", "much"),
        ("which", "one"),
        ("what", "about"),
        ("tell", "me", "more"),
    }
    return any(tuple(words[: len(pattern)]) == pattern for pattern in short_follow_ups)


def contextualize_question(question, history=None):
    normalized_history = normalize_history(history)
    if not normalized_history or not is_follow_up(question):
        return question

    previous_user = next(
        (message["content"] for message in reversed(normalized_history) if message["role"] == "user"),
        "",
    )
    previous_answer = next(
        (message["content"] for message in reversed(normalized_history) if message["role"] == "assistant"),
        "",
    )
    count_hint = ""
    if question.lower().strip().startswith(("how many", "how much")):
        count_hint = "\nNeed: total count, number, titles, wins, championships, or amount."

    return (
        f"Previous question: {previous_user}\n"
        f"Previous answer: {previous_answer}\n"
        f"Current follow-up question: {question}"
        f"{count_hint}"
    )


def question_keywords(question):
    return set(meaningful_words(question))


def context_matches_question(question, docs):
    keywords = question_keywords(question)
    if not keywords:
        return True

    context_text = " ".join(doc.page_content.lower() for doc in docs)
    matched_keywords = {keyword for keyword in keywords if keyword in context_text}
    if len(keywords) == 1:
        return bool(matched_keywords)

    return len(matched_keywords) >= 2 or any(phrase in context_text for phrase in important_phrases(question))


def question_asks_count(question):
    lower_question = question.lower()
    return "how many" in lower_question or "how much" in lower_question


def answer_supported_by_context(answer, docs, question=""):
    if OUT_OF_CONTEXT_ANSWER.lower() in answer.lower():
        return True

    context_text = " ".join(doc.page_content.lower() for doc in docs)
    answer_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", answer)
    if answer_numbers and not all(number in context_text for number in answer_numbers):
        return False
    if question_asks_count(question):
        if answer_numbers and not count_numbers_supported(answer_numbers, answer, context_text, question):
            return False
        year_like_numbers = [number for number in answer_numbers if number.isdigit() and 1900 <= int(number) <= 2100]
        if answer_numbers and len(year_like_numbers) == len(answer_numbers):
            return False

    answer_terms = {
        term
        for term in meaningful_words(answer)
        if term not in {"answer", "context", "indexed", "information", "question", "source", "sources"}
    }
    if not answer_terms:
        return True

    return any(term in context_text for term in answer_terms)


def count_numbers_supported(answer_numbers, answer, context_text, question):
    subject_terms = question_keywords(question) | question_keywords(answer)
    subject_terms -= {"count", "number", "total", "amount", "many", "much"}
    unit_terms = {
        "championship",
        "championships",
        "constructor",
        "constructors",
        "title",
        "titles",
        "win",
        "wins",
        "won",
        "race",
        "races",
    }

    for number in answer_numbers:
        supported = False
        for match in re.finditer(rf"\b{re.escape(number)}\b", context_text):
            window = context_text[max(match.start() - 130, 0) : match.end() + 130]
            has_subject = not subject_terms or any(term in window for term in subject_terms)
            has_unit = any(term in window for term in unit_terms)
            if has_subject and has_unit:
                supported = True
                break

        if not supported:
            return False

    return True


def format_sources(docs, question=None):
    sources = []
    seen = set()

    for doc in docs:
        metadata = doc.metadata or {}
        source = metadata.get("source", "Unknown source")
        if source in seen:
            continue

        seen.add(source)
        source_text = str(source)
        pdf_url = metadata.get("pdf_url", "")
        source_url = source_text if source_text.startswith("http") else pdf_url
        sources.append(
            {
                "source": source_text,
                "title": metadata.get("title") or Path(source_text).name,
                "loader": metadata.get("loader", "document"),
                "published": metadata.get("published", ""),
                "url": source_url,
                "snippet": source_snippet(doc, question),
                "retrieval_type": metadata.get("retrieval_type", "document"),
                "score": metadata.get("retrieval_score", ""),
            }
        )

    return sources


def source_snippet(doc, question=None, max_chars=260):
    text = " ".join(doc.page_content.split())
    if not text:
        return ""

    keywords = question_keywords(question or "")
    lower_text = text.lower()
    start = 0

    for keyword in keywords:
        index = lower_text.find(keyword)
        if index >= 0:
            start = max(index - 70, 0)
            break

    snippet = text[start : start + max_chars].strip()
    if start > 0:
        snippet = "..." + snippet
    if start + max_chars < len(text):
        snippet += "..."
    return snippet


def answer_question(question, history=None):
    search_question = contextualize_question(question, history)
    used_history = search_question != question
    docs = get_relevant_documents(search_question)
    cached_answer = get_cached_answer(search_question)

    if cached_answer is not None:
        return {
            "answer": cached_answer,
            "sources": format_sources(docs, search_question) if context_matches_question(search_question, docs) else [],
            "cached": True,
            "used_history": used_history,
        }

    if not context_matches_question(search_question, docs):
        cache_answer(search_question, OUT_OF_CONTEXT_ANSWER)
        return {
            "answer": OUT_OF_CONTEXT_ANSWER,
            "sources": [],
            "cached": False,
            "used_history": used_history,
        }

    context = format_docs(docs)
    formatted_prompt = prompt.format(context=context, question=search_question)
    answer = llm.invoke(formatted_prompt)

    if not answer_supported_by_context(answer, docs, search_question):
        answer = OUT_OF_CONTEXT_ANSWER

    cache_answer(search_question, answer)

    return {
        "answer": answer,
        "sources": format_sources(docs, search_question) if answer != OUT_OF_CONTEXT_ANSWER else [],
        "cached": False,
        "used_history": used_history,
    }


def find_matching_text_documents(question):
    question_terms = {term for term in question.lower().split() if len(term) > 2}
    matches = []

    for text_path in DATA_DIR.rglob("*.txt"):
        searchable_name = text_path.stem.lower().replace("-", " ").replace("_", " ")
        name_terms = set(searchable_name.split())
        if len(question_terms & name_terms) < 3:
            continue

        matches.append(
            Document(
                page_content=text_path.read_text(encoding="utf-8"),
                metadata={"source": str(text_path), "loader": "text-title-match"},
            )
        )

    return matches


def ask_question(question):
    """Stream responses for faster perceived speed."""
    docs = get_relevant_documents(question)
    cached_answer = get_cached_answer(question)
    if cached_answer is not None:
        print("\nAnswer:\n" + cached_answer)
        return

    if not context_matches_question(question, docs):
        cache_answer(question, OUT_OF_CONTEXT_ANSWER)
        print("\nAnswer:\n" + OUT_OF_CONTEXT_ANSWER)
        return

    final_answer = []
    context = format_docs(docs)
    formatted_prompt = prompt.format(context=context, question=question)

    print("\nAnswer:\n", end="", flush=True)
    for chunk in llm.stream(formatted_prompt):
        final_answer.append(chunk)
        print(chunk, end="", flush=True)
    print()
    answer = "".join(final_answer)
    if not answer_supported_by_context(answer, docs, question):
        answer = OUT_OF_CONTEXT_ANSWER
        print("\nCorrected answer:\n" + answer)
    cache_answer(question, answer)


if __name__ == "__main__":
    print("\nRAG assistant is ready.")
    print("Ask a question, or type 'exit' to quit.")

    while True:
        question = input("\nQuestion: ").strip()

        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        if not question:
            continue

        ask_question(question)
