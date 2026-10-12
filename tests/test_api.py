from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_root():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["message"] == "DSA RAG Chatbot API is running"


def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


def test_invalid_question():

    response = client.post(
        "/ask",
        json={
            "question": "",
            "top_k": 5
        }
    )

    assert response.status_code == 422