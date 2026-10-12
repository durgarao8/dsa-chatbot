
from src.rag.pipeline import answer_question


if __name__ == "__main__":
    result = answer_question(
        question="How does Kruskal's algorithm detect cycles?",
        top_k=5,
        retrieval_mode="hyde",
    )

    print("\nQUESTION:")
    print(result["question"])

    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES:")
    for source in result["sources"]:
        print(source)
