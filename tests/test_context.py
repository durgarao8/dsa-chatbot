
from src.rag.context import build_context


def test_build_context():
    fake_results = {
        "documents": [[
            "A stack follows the LIFO principle.",
            "Push adds an element to the stack.",
            "Pop removes the top element."
        ]],
        "metadatas": [[
            {"chapter": "Stacks", "page": 10, "source": "dsa_pdf.pdf"},
            {"chapter": "Stacks", "page": 11, "source": "dsa_pdf.pdf"},
            {"chapter": "Stacks", "page": 11, "source": "dsa_pdf.pdf"}
        ]]
    }

    context = build_context(fake_results)

    assert isinstance(context, str)
    assert "LIFO" in context
    assert "Push" in context
    assert "Pop" in context
    assert len(context) > 0
