"""
Text Splitters & Chunking Strategies
Optimizing document chunks for RAG
"""

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


SAMPLE_TEXT = """
Retrieval-Augmented Generation (RAG) combines information retrieval
with language generation. It allows LLMs to retrieve relevant information
from external knowledge sources before generating an answer.

Chunking is an important part of RAG. Documents are divided into smaller
pieces called chunks. These chunks are converted into embeddings and stored
in a vector database.

When a user asks a question, the system retrieves relevant chunks based on
semantic similarity. The retrieved information is then passed to an LLM
to generate a contextual answer.

Good chunking improves retrieval accuracy, reduces noise, and helps the
language model understand the context of retrieved information.
"""


def recursive_splitter():
    # Initialize the recursive text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=30,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    # Split the text into smaller chunks
    chunks = text_splitter.split_text(SAMPLE_TEXT)

    # Display the generated chunks
    print(f"Total Chunks: {len(chunks)}")

    for index, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {index} ---")
        print(chunk)
        print(f"Chunk Length: {len(chunk)}")


if __name__ == "__main__":
    recursive_splitter()
