# Chromadb widely used for vector db
import chromadb

chroma_client = chromadb.Client()

collection_name = "test_collection"

collection = chroma_client.get_or_create_collection(collection_name)

# dumb documents

documents = [
    {"id": "doc-001", "text": "Im excited to learning RAG"},
    {"id": "doc-002", "text": "I love to write code"},
    {"id": "doc-003", "text": "I wants to become AI Engineer"},
]


for doc in documents:
    collection.upsert(ids=doc["id"], documents=[doc["text"]])

# default query text
query_text = "I wants to do something"


results = collection.query(query_texts=[query_text], n_results=3)


print(results)
