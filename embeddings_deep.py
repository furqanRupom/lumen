"""
Embedding Fundamentals with Google Gemini and LangChain
=======================================================

This script demonstrates different embedding techniques using
Google's Gemini embedding model through LangChain.

Topics covered:
    1. Basic query embeddings
    2. Batch document embeddings
    3. Cosine similarity
    4. Euclidean distance
    5. Dot product

Purpose:
    Understand how text is converted into numerical vectors
    and how those vectors can be compared in a RAG system.
"""

import os

import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

API_KEY = os.getenv("GENAI_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", api_key=API_KEY)


def basic_embedding():
    """
    Generate an embedding for a single text query.

    embed_query() converts a text string into a numerical vector.

    We inspect:
        - Vector dimensions
        - First five vector values
        - Vector norm (magnitude)
    """
    text = "What is LangChain?"

    vector = embeddings.embed_query(text)

    print("\n--- Basic Embedding ---")
    print(f"Dimensions: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")
    print(f"Vector norm: {np.linalg.norm(vector):.4f}")


def batch_embeddings():
    """
    Generate embeddings for multiple documents at once.

    embed_documents() accepts a list of text documents
    and returns a corresponding list of numerical vectors.

    Each document gets its own embedding.
    """
    texts = ["What is LangChain?", "What is RAG?", "How does a neural network work?"]

    vectors = embeddings.embed_documents(texts)

    print("\n--- Batch Embeddings ---")

    for i, vector in enumerate(vectors):
        print(f"\nText: {texts[i]}")
        print(f"Dimensions: {len(vector)}")
        print(f"First 5 values: {vector[:5]}")
        print(f"Vector norm: {np.linalg.norm(vector):.4f}")


def similarity_search():
    """
    Perform semantic similarity search using cosine similarity.

    Steps:
        1. Convert the query into an embedding.
        2. Convert documents into embeddings.
        3. Calculate cosine similarity between query and documents.
        4. Rank documents by similarity score.
        5. Display the most relevant documents first.
    """

    query = "What is LangChain?"

    docs = [
        "LangChain is a framework for building applications powered by LLMs.",
        "Python is a popular programming language.",
        "RAG stands for Retrieval-Augmented Generation.",
        "LangChain helps developers build AI applications.",
        "Neural networks are inspired by the human brain.",
    ]

    # generate embeddings
    query_vector = np.array(embeddings.embed_query(query))
    doc_vectors = embeddings.embed_documents(docs)

    # compute cosine similarities
    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    similarities = [
        cosine_similarity(query_vector, np.array(doc_vec)) for doc_vec in doc_vectors
    ]

    # rank documents by similarity
    ranked_docs = sorted(zip(docs, similarities), key=lambda x: x[1], reverse=True)

    # display ranked documents
    print("\n--- Similarity Search ---")
    print(f"Query: {query}\n")

    for doc, score in ranked_docs:
        print(f"Document: {doc}")
        print(f"Similarity score: {score:.4f}")
        print()


def euclidean_distance():
    """
    Calculate Euclidean distance between two embeddings.

    Euclidean distance measures the straight-line distance
    between two points in a vector space.

    A smaller distance means the vectors are closer together.
    """
    text1 = "What is LangChain?"
    text2 = "How does a neural network work?"

    vector1 = np.array(embeddings.embed_query(text1))
    vector2 = np.array(embeddings.embed_query(text2))

    distance = np.linalg.norm(vector1 - vector2)

    print("\n--- Euclidean Distance ---")
    print(f"Distance: {distance:.4f}")


def dot_product():
    """
    Calculate the dot product between two embeddings.

    The dot product measures the alignment and magnitude
    of two vectors.

    For normalized vectors, the dot product is equivalent
    to cosine similarity.
    """
    text1 = "What is LangChain?"
    text2 = "Explain LangChain framework."

    vector1 = np.array(embeddings.embed_query(text1))
    vector2 = np.array(embeddings.embed_query(text2))

    result = np.dot(vector1, vector2)

    print("\n--- Dot Product ---")
    print(f"Result: {result:.4f}")


if __name__ == "__main__":
    """
    Main execution point.

    Run each embedding experiment sequentially.
    """
    basic_embedding()
    batch_embeddings()
    similarity_search()
    euclidean_distance()
    dot_product()
