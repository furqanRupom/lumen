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


def cosine_similarity():
    """
    Calculate cosine similarity between two text embeddings.

    Cosine similarity measures the angle between two vectors.

    A value closer to 1 means greater directional similarity.
    A value around 0 means little directional similarity.
    A value closer to -1 means opposite directions.

    This is commonly used in RAG for semantic similarity search.
    """
    text1 = "What is LangChain?"
    text2 = "Explain LangChain framework."

    vector1 = np.array(embeddings.embed_query(text1))
    vector2 = np.array(embeddings.embed_query(text2))

    similarity = np.dot(vector1, vector2) / (
        np.linalg.norm(vector1) * np.linalg.norm(vector2)
    )

    print("\n--- Cosine Similarity ---")
    print(f"Text 1: {text1}")
    print(f"Text 2: {text2}")
    print(f"Similarity: {similarity:.4f}")


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
    cosine_similarity()
    euclidean_distance()
    dot_product()
