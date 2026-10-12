
from src.retrieval.retriever import retrieve, retrieve_with_hyde
from src.rag.context import build_context
from src.llm.prompt import build_prompt
from src.llm.groq_client import generate_response


def answer_question(
    question: str,
    top_k: int = 5,
    retrieval_mode: str = "hyde",
):
    # 1. Retrieve textbook chunks
    if retrieval_mode == "hyde":
        results = retrieve_with_hyde(
            query=question,
            top_k=top_k,
        )
    elif retrieval_mode == "hybrid":
        results = retrieve(
            query=question,
            top_k=top_k,
        )
    else:
        raise ValueError(
            "retrieval_mode must be 'hyde' or 'hybrid'"
        )

    # 2. Convert retrieved chunks into context
    context = build_context(results)

    # 3. Build the question-answering prompt
    prompt = build_prompt(
        question=question,
        context=context,
    )

    # 4. Generate the answer using Groq
    answer = generate_response(prompt=prompt)

    # 5. Collect source metadata
    metadatas = results.get("metadatas", [[]])[0]
    sources = []

    for metadata in metadatas:
        metadata = metadata or {}
        sources.append({
            "chapter": str(metadata.get("chapter", "Unknown")),
            "page": str(metadata.get("page", "Unknown")),
            "source": str(metadata.get("source", "Unknown")),
        })

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
    }
