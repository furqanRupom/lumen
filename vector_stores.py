import os
import tempfile

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()
SAMPLE_DOCS = [
    Document(
        page_content="Python is a high-level, interpreted programming language known for its clean syntax and massive ecosystem.",
        metadata={
            "source": "python_overview.md",
            "author": "Python Core Team",
            "category": "language",
        },
    ),
    Document(
        page_content="ChromaDB is an open-source vector database designed to store, index, and query vector embeddings efficiently.",
        metadata={
            "source": "chroma_docs_v2.pdf",
            "author": "Chroma Documentation",
            "category": "database",
        },
    ),
    Document(
        page_content="LangChain is a framework designed to simplify the creation of applications using large language models through composable chains.",
        metadata={
            "source": "langchain_handbook.txt",
            "author": "LangChain Community",
            "category": "framework",
        },
    ),
    Document(
        page_content="FastAPI is a modern, high-performance web framework for building APIs with Python, leveraging standard Python type hints.",
        metadata={
            "source": "fastapi_guide.rst",
            "author": "Tiangolo",
            "category": "backend",
        },
    ),
    Document(
        page_content="Retrieval-Augmented Generation (RAG) combines external document search with LLM text generation to provide accurate, context-aware answers.",
        metadata={
            "source": "rag_architecture_whitepaper.pdf",
            "author": "AI Research Group",
            "category": "architecture",
        },
    ),
    Document(
        page_content="Vector embeddings are numerical representations of text that capture semantic meaning, allowing math to measure similarity.",
        metadata={
            "source": "embeddings_deep_dive.md",
            "author": "ML Engineering Blog",
            "category": "theory",
        },
    ),
]
API_KEY = os.getenv("GENAI_API_KEY")


def chroma_basics():

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001", api_key=API_KEY
    )

    with tempfile.TemporaryDirectory() as tempdir:
        # Create vector store from documents
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS,
            embedding=embeddings,
            persist_directory=tempdir,
        )

        # Perform similarity search
        results = vectorstore.similarity_search("What is Python?")

        for result in results:
            print(result.page_content)


def similarity_search_with_scores():

    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001", api_key=API_KEY
    )
    with tempfile.TemporaryDirectory() as tempdir:
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS, embedding=embeddings_model, persist_directory=tempdir
        )
        query = "What is FastAPI?"
        result_with_scores = vectorstore.similarity_search_with_score(query, k=3)
        print(f"Top 3 results with score : {query}")
        for i, (doc, score) in enumerate(result_with_scores):
            print(
                f"Result {i + 1}: {doc.page_content} score : {score:.4f} Source : {doc.metadata['source']}"
            )


def metadata_filtering():
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001", api_key=API_KEY
    )
    with tempfile.TemporaryDirectory() as tempdir:
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS, embedding=embeddings_model, persist_directory=tempdir
        )
        query = "What is FastAPI?"
        filter_criteria = {"category": "backend"}
        filter_results = vectorstore.similarity_search(
            query, filter=filter_criteria, k=3
        )
        print(f"\nResults with metadata filtering for query : {query}")
        for i, doc in enumerate(filter_results):
            print(
                f"Result {i + 1}: {doc.page_content} Source : {doc.metadata['source']}"
            )


if __name__ == "__main__":
    # chroma_basics()
    similarity_search_with_scores()
    metadata_filtering()
