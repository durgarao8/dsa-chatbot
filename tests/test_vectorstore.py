
from src.vectorstore.chroma_store import list_collections


def test_chromadb_collections():
    collections = list_collections()

    assert collections is not None
    assert len(collections) > 0, "No ChromaDB collections found"

    collection_names = [
        collection if isinstance(collection, str)
        else getattr(collection, "name", None)
        for collection in collections
    ]

    assert "langchain" in collection_names, (
        f"Expected collection 'langchain'. Found: {collection_names}"
    )
