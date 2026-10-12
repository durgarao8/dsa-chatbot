
# src/retrieval/hyde.py

from typing import Any


def generate_hypothetical_document(llm: Any, query: str) -> str:
    """Generate a hypothetical textbook passage for HyDE retrieval."""

    prompt = f"""
You are a Data Structures and Algorithms textbook author.

Write a short, technically accurate textbook-style passage
that would be relevant to answering the question.

Include relevant algorithm terminology, definitions, and
complexity where appropriate.

Do not answer conversationally.
Do not mention that this passage is hypothetical.
Do not invent specific textbook page numbers.

Question:
{query}

Hypothetical textbook passage:
"""

    response = llm(prompt)

    # Support LangChain chat models and plain string responses.
    if isinstance(response, str):
        passage = response.strip()
    else:
        passage = response.content.strip()

    if not passage:
        raise ValueError("The LLM generated an empty hypothetical passage.")

    return passage



def hyde_search(query, llm, embedding_model, collection, top_k=5):
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    # 1. Generate a hypothetical document
    hypothetical_document = generate_hypothetical_document(
        llm=llm,
        query=query,
    )

    # 2. Embed the hypothetical document
    hypothetical_embedding = embedding_model.encode(
        hypothetical_document,
        normalize_embeddings=True,
    ).tolist()

    # 3. Retrieve real textbook chunks from ChromaDB
    results = collection.query(
        query_embeddings=[hypothetical_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Important: return the ChromaDB dictionary unchanged
    return results
