
from src.rag.pipeline import answer_question


def test_complete_rag_pipeline():
    question = "What is a stack?"

    result = answer_question(
        question=question,
        top_k=5
    )

    assert isinstance(result, dict)
    assert result.get("question") == question

    # Verify that the LLM generated an answer.
    assert isinstance(result.get("answer"), str)
    assert len(result["answer"].strip()) > 0

    # Verify that source citations were returned.
    assert isinstance(result.get("sources"), list)
    assert len(result["sources"]) > 0

    # Verify the source metadata.
    for source in result["sources"]:
        assert isinstance(source, dict)
        assert "source" in source
