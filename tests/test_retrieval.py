
from src.retrieval.retriever import retrieve


def test_retrieve_dsa_documents():
    queries = [
        "What is a stack?",
        "What is binary search?",
        "What is a binary tree?",
    ]

    for query in queries:
        results = retrieve(query, top_k=5)

        assert "documents" in results, f"Missing documents for: {query}"
        assert results["documents"], f"No document results for: {query}"

        documents = results["documents"][0]

        assert len(documents) > 0, f"No documents retrieved for: {query}"
        assert len(documents) <= 5
        assert all(isinstance(doc, str) for doc in documents)
