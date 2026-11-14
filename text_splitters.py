"""
Text Splitters & Chunking Strategies
Optimizing document chunks for RAG
"""

from dotenv import load_dotenv
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

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


SAMPLE_MARKDOWN = """
# Introduction to RAG

Retrieval-Augmented Generation combines retrieval with generation.

## What is RAG?

RAG allows LLMs to retrieve relevant information from external documents.

## How RAG Works

### Document Loading

Load documents from different sources.

### Text Chunking

Split large documents into smaller chunks.

### Embeddings

Convert chunks into vector representations.

# Conclusion

RAG helps LLMs generate answers using external knowledge.
"""


def recursive_splitter():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = text_splitter.split_text(SAMPLE_TEXT)

    print(f"Total Chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {i} ---")
        print(chunk)
        print(f"Chunk Length: {len(chunk)}")


def markdown_splitter():
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )

    chunks = markdown_splitter.split_text(SAMPLE_MARKDOWN)

    print(f"\nTotal Markdown Chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {i} ---")
        print(chunk.page_content)
        print(f"Metadata: {chunk.metadata}")


def overlap_importance():
    text = "The quick brown fox jumps over the lazy dogs." * 10

    # no overlap
    no_overlap = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)

    # with overlap
    with_overlap = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=20)

    chunk_no_overlap = no_overlap.split_text(text)
    chunk_with_overlap = with_overlap.split_text(text)

    print("Without overlap")
    print(f"Chunk 1 end : ...{chunk_no_overlap[0][-20:]}")
    print(f"Chunk 2 start : {chunk_no_overlap[1][:20]}...")

    print("With overlap")
    print(f"Chunk 1 end : ...{chunk_with_overlap[0][-20:]}")
    print(f"Chunk 2 start : {chunk_with_overlap[1][:20]}...")


if __name__ == "__main__":
    print("\n========== RECURSIVE SPLITTER ==========")
    recursive_splitter()

    print("\n========== MARKDOWN SPLITTER ==========")
    markdown_splitter()

    print("\n========== OVERLAP IMPORTANCE ==========")
    overlap_importance()
