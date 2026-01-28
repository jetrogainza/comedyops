from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_rewrite_premise_returns_shape():
    # Simulate a well-behaved agent:
    # 1) proposes a rewrite
    # 2) then finalises
    fake_outputs = [
        '{"action": "rewrite", "text": "A tighter rewritten premise."}',
        '{"action": "final", "text": "A tighter rewritten premise."}',
    ]

    with patch("app.main.llm.generate", side_effect=fake_outputs):
        r = client.post(
            "/rewrite_premise",
            json={"premise": "I moved to Australia and everyone calls me mate."},
        )

    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"rewritten", "model"}
    assert body["rewritten"] == "A tighter rewritten premise."