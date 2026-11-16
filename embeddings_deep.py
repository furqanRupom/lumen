import os

import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

API_KEY = os.getenv("GENAI_API_KEY")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", api_key=API_KEY)


def basic_embedding():
    text = "What is LangChain?"
    single_embedding = embeddings.embed_query(text)
    print(f"Vector dimensions : {len(single_embedding)}")
    print(f"First 5 values : {single_embedding[:5]}")
    print(f"Vector norm : {np.linalg.norm(single_embedding):.4f}")


def batch_embeddings():
    text = ["What is LangChain?", "What is Rag", "How does a neural network work?"]

    batch_embedding = embeddings.embed_documents(text)

    for i, emb in enumerate(batch_embedding):
        print(f"Text {i + 1} Vector dimensions : {len(emb)}")
        print(f"Text {i + 1} First 5 values : {emb[:5]}")
        print(f"Text {i + 1} Vector norm : {np.linalg.norm(emb):.4f}")


if __name__ == "__main__":
    basic_embedding()
    batch_embeddings()
