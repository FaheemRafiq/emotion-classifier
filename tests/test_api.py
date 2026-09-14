import pytest

from src import config

pytestmark = pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="trained model missing; run `make evaluate`")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_label_and_probabilities(client):
    response = client.post("/predict", json={"text": "I wasted the whole day and regret it."})
    assert response.status_code == 200
    body = response.json()
    assert body["emotion"] in config.MENTOR_CLASSES
    if body["probabilities"] is not None:
        assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
        assert body["confidence"] == pytest.approx(max(body["probabilities"].values()))


@pytest.mark.parametrize("text", ["", "   ", "x" * (config.MAX_TEXT_CHARS + 1)])
def test_predict_rejects_invalid_text(client, text):
    assert client.post("/predict", json={"text": text}).status_code == 422
