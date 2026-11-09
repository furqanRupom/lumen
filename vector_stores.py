import os
import tempfile

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()
SAMPLE_DOCS = [
    Document(page_content="Python is a programming language."),
    Document(page_content="ChromaDB is a vector database."),
    Document(page_content="LangChain helps build LLM applications."),
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
        query = "What is Python?"
        result_with_scores = vectorstore.similarity_search_with_score(query, k=3)
        print(f"Top 3 results with score : {query}")
        for i, (doc, score) in enumerate(result_with_scores):
            print(f"Result {i + 1}: {doc.page_content} score : {score:.4f}")


if __name__ == "__main__":
    chroma_basics()
    similarity_search_with_scores()
