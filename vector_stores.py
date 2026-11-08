import os
import tempfile

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


def chroma_basics():
    documents = [
        Document(page_content="Python is a programming language."),
        Document(page_content="ChromaDB is a vector database."),
        Document(page_content="LangChain helps build LLM applications."),
    ]

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY")
    )

    with tempfile.TemporaryDirectory() as tempdir:
        # Create vector store from documents
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=tempdir,
        )

        # Perform similarity search
        results = vectorstore.similarity_search("What is Python?")

        for result in results:
            print(result.page_content)


def similarity_search_with_scores():
    documents = [
        Document(page_content="Python is a programming language."),
        Document(page_content="JavaScript is used for web development."),
        Document(page_content="ChromaDB stores vector embeddings."),
    ]

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY")
    )

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
    )

    results = vectorstore.similarity_search_with_score(
        "What is Python?",
        k=2,
    )

    for document, score in results:
        print(f"Content: {document.page_content}")
        print(f"Score: {score}")


if __name__ == "__main__":
    chroma_basics()
    similarity_search_with_scores()
