
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from src.retrieval.hyde import hyde_search
from src.llm.groq_client import generate_response


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "deployment_assets" / "all-MiniLM-L6-v2"
CHROMA_PATH = PROJECT_ROOT / "vectorstore" / "chroma_db"
COLLECTION_NAME = "langchain"

_embedding_model = None
_collection = None


def load_retriever():
    """Load and cache the existing embedding model and ChromaDB."""

    global _embedding_model, _collection

    if _embedding_model is not None and _collection is not None:
        return _embedding_model, _collection

    _embedding_model = SentenceTransformer(
        str(MODEL_PATH),
        local_files_only=True,
    )

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    _collection = client.get_collection(name=COLLECTION_NAME)

    return _embedding_model, _collection


def retrieve(query: str, top_k: int = 5):
    """Retrieve using the original vector-search method."""

    embedding_model, collection = load_retriever()

    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )


def _hyde_llm_adapter(prompt: str) -> str:
    """Use the existing Groq client to generate HyDE text."""

    response = generate_response(prompt=prompt)

    if isinstance(response, str):
        return response

    if hasattr(response, "content"):
        return response.content

    return str(response)


def retrieve_with_hyde(query: str, top_k: int = 5):
    """Retrieve real textbook chunks using HyDE."""

    embedding_model, collection = load_retriever()

    return hyde_search(
        query=query,
        llm=_hyde_llm_adapter,
        embedding_model=embedding_model,
        collection=collection,
        top_k=top_k,
    )
