# -*- coding: utf-8 -*-
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
import pytesseract
from PIL import Image

from langchain_community.document_loaders import (
    DirectoryLoader,
    Docx2txtLoader,
    PyPDFLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral")
EMBEDDING_MODEL = "nomic-embed-text"
DATA_DIR = Path("data")
CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "ollama_rag_demo"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

tesseract_cmd = os.getenv("TESSERACT_CMD")
if tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
else:
    default_tesseract_path = Path("C:/Program Files/Tesseract-OCR/tesseract.exe")
    if default_tesseract_path.exists():
        pytesseract.pytesseract.tesseract_cmd = str(default_tesseract_path)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def load_ocr_documents(data_dir):
    """Extract text from image files and return LangChain documents."""
    ocr_docs = []

    for image_path in data_dir.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        try:
            with Image.open(image_path) as image:
                text = pytesseract.image_to_string(image).strip()
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
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Larger chunks = fewer embeddings = faster
        chunk_overlap=150,
    )
    split_docs = text_splitter.split_documents(docs)
    print(f"Split into {len(split_docs)} chunks. Building vector store...")

    # Build vectorstore with batch processing and progress
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    print("Persisting vectorstore...")
    vectorstore.persist()
    print("Done!")
    return vectorstore


def load_vectorstore():
    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        print("Using existing Chroma DB. Skipping document loading and OCR.")
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )

    print("No existing Chroma DB found. Building it from files in data/...")
    return build_vectorstore()


vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  

llm = OllamaLLM(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.2,
    num_predict=128,  # Reduced from 256 - shorter responses = faster
    top_k=30,  # Reduced from 40
    top_p=0.85,  # Reduced from 0.92 - narrows token choices
    keep_alive=600,
)

prompt = ChatPromptTemplate.from_template(
    """
You are an AI assistant using retrieved knowledge.

Rules:
- Use provided context first
- If context is insufficient, say "I don't have enough information"
- Be concise and accurate

Context:
{context}

Question:
{question}
"""
)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)


def ask_question(question):
    """Stream responses for faster perceived speed."""
    print("\nAnswer:\n", end="", flush=True)
    for chunk in rag_chain.stream(question):
        print(chunk, end="", flush=True)
    print()


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
