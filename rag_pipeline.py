import os
import tempfile
from typing import List

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

load_dotenv()

API_KEY = os.getenv("GENAI_API_KEY")
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001", api_key=API_KEY
)

KNOWLEDGE_BASE = """
- Python is a high-level, interpreted programming language known for its clean syntax.
- ChromaDB is an open-source vector database designed to store and query vector embeddings efficiently.
- LangChain is a framework designed to simplify the creation of applications using large language models.
- FastAPI is a modern, high-performance web framework for building APIs with Python.
"""


def create_kb():
    """Create a vectore from kb"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    doc = Document(
        page_content=KNOWLEDGE_BASE, metadata={"source": "langchain_knowledge_base.md"}
    )
    chunks = splitter.split_documents([doc])

    # create a vector store from chunks
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=tempfile.mkdtemp(),
    )
    return vector_store


def demo_basic_rag():
    vector_store = create_kb()
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 2}
    )
    llm = init_chat_model(
        model="gemini-3.5-flash-lite",
        model_provider="google-genai",
        api_key=API_KEY,
    )

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant. Answer the question using ONLY the provided context below. If you don't know the answer, say "I don't know."

   Context:
   {context}

   Question: {question}

   Answer:"""
    )

    # format retrieved documents
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    # Rag Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    # Test the Rag Chain
    questions = ["What is Python?", "What is Chromadb?", "What is Langchain?"]
    print("Basic Rag Demo : \n")
    for q in questions:
        answer = rag_chain.invoke(q)
        print(f"Q : {q}")
        print(f"A : {answer}\n")


if __name__ == "__main__":
    demo_basic_rag()
