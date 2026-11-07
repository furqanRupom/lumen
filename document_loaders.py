import os
import tempfile
from pathlib import Path

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)

load_dotenv()


def load_text_file():
    # create a temporary file for demostrations
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(
            b"Hello this is sample text file. \n This file used to testing is text file is loading or not. \n It will be great if its loaded. \n thanks you for your attention"
        )
        temp_file_path = temp_file.name

    try:
        # load the text file using TextLoader
        loader = TextLoader(temp_file_path)
        documents = loader.load()

        # print the loaded documents
        for doc in documents:
            print(doc.page_content)

    finally:
        # cleanup the temporary file
        os.remove(temp_file_path)


def load_pdf_file(file_path: str):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"Loaded document(s) {len(documents)} from PDF")

    for i, doc in enumerate(documents):
        print(f"Document {i + 1} Content Preview : {doc.page_content[:1]}")
        print(f"Document metadata : {doc.metadata}")


if __name__ == "__main__":
    # load_text_file()
    load_pdf_file("./docs/document.pdf")
