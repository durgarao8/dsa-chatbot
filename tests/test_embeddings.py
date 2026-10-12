
from src.embeddings.embedder import load_embedding_model


def test_embedding_model():
    model = load_embedding_model()

    test_text = "What is a binary tree?"

    embedding = model.encode(
        test_text,
        convert_to_numpy=True
    )

    assert embedding is not None
    assert embedding.shape == (384,)
    assert embedding.dtype.kind == "f"
