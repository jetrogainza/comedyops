from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_rewrite_premise_returns_shape():
    # We mock Ollama so the test doesn't rely on your local model running.
    with patch("app.main._ollama_generate", return_value="A tighter rewritten premise."):
        r = client.post("/rewrite_premise", json={"premise": "I moved to Australia and everyone calls me mate."})

    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"rewritten", "model"}
    assert body["rewritten"] == "A tighter rewritten premise."
    assert isinstance(body["model"], str) and body["model"]