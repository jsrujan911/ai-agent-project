import os

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings, OllamaLLM

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
LLM_MODEL = "mistral"
EMBEDDING_MODEL = "nomic-embed-text"


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


documents = [
    Document(
        page_content=(
            "Claude Cowork is an agentic AI system developed by Anthropic designed for knowledge work "
            "It allows users to give Claude a goal and let it autonomously handle tasks across local files and applications, producing polished documents, reports, and other outputs without manual intervention. "
            "Key features include: Autonomous Task Execution: Claude can read, edit, and create files on your computer, executing multi-step tasks from start to finish."
            " File Management: It can organize downloads, compile research, and manage files efficiently."
            " User-Friendly Interface: Designed for non-technical users, it simplifies complex tasks and enhances productivity."
            " Integration with Other Tools: Claude Cowork can interact with various applications like Chrome, Word, and Excel, making it versatile for different workflows."
        ),
        metadata={"source": "claude_cowork_notes"},
    ),
    Document(
        page_content=(
            "Ollama can run both language models and embedding models locally. "
            "Use an embedding model such as nomic-embed-text for Chroma, and use "
            "a text generation model such as mistral for final answers."
        ),
        metadata={"source": "ollama_notes"},
    ),
    Document(
        page_content=(
            "Chroma does not connect directly to the LLM. Chroma connects to an "
            "embedding function. In LangChain, the RAG chain connects Chroma's "
            "retriever output to the LLM prompt after similar documents have "
            "been retrieved."
        ),
        metadata={"source": "chroma_notes"},
    ),
    Document(
        page_content=("OpenAI CEO Sam Altman, in an interview with Commonwealth Bank of Australia CEO Matt Comyn on Tuesday, said he was “pretty wrong” about AI’s economic impact—a reversal from his June 2025 warnings that entry-level roles were at serious risk."
                      " Anthropic CEO Dario Amodei, who once claimed AI could eliminate 50% of white-collar jobs, now says automation may actually expand the work people do. "
                      "Solomon, meanwhile, has argued consistently since at least late 2025 that the panic was overblown—and is now pointing to a century of American economic history to say he was right."
                      ),
         metadata={"source": "ai_economic_impact_notes"},)
]

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL,
)

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="ollama_rag_demo",
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

llm = OllamaLLM(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.2,
)

prompt = ChatPromptTemplate.from_template(
    """Answer the question using only the context below.
Be direct and mention the two separate Ollama models: one for embeddings and one for generation.

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

question = "What happened to claude mythos?"
response = rag_chain.invoke(question)

print("Question:\n", question)
print("\nAnswer:\n", response)
