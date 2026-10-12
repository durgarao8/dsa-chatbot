
from src.llm.prompt import build_prompt


def test_build_prompt():
    question = "What is a stack?"

    context = """
    A stack is a linear data structure that follows
    the LIFO (Last In First Out) principle.
    The main operations are push and pop.
    """

    prompt = build_prompt(
        question=question,
        context=context
    )

    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert question in prompt
    assert "LIFO" in prompt
