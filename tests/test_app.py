import importlib
from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class FakeConn:
    async def fetchval(self, query: str):
        assert query == "SELECT 1"
        return 1


class FakeAcquire:
    async def __aenter__(self):
        return FakeConn()

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakePool:
    closed = False

    def acquire(self):
        return FakeAcquire()


class FakePinecone:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def Index(self, name: str):
        return {"index_name": name}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone-key")

    sys.modules.pop("app", None)
    import app as app_module

    importlib.reload(app_module)

    fake_pool = FakePool()

    async def fake_connect_db():
        return fake_pool

    async def fake_init_db(pool):
        return None

    async def fake_disconnect_db(pool):
        return None

    def fake_initialize_retrieval_components():
        return object(), object()

    monkeypatch.setattr(app_module, "connect_db", fake_connect_db)
    monkeypatch.setattr(app_module, "init_db", fake_init_db)
    monkeypatch.setattr(app_module, "disconnect_db", fake_disconnect_db)
    monkeypatch.setattr(app_module, "initialize_retrieval_components", fake_initialize_retrieval_components)
    monkeypatch.setattr(app_module, "Pinecone", FakePinecone)

    with TestClient(app_module.app) as test_client:
        yield test_client


def test_root_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"status": "working"}


def test_api_root_endpoint(client):
    response = client.get("/api")

    assert response.status_code == 200
    assert response.json() == {"message": "API is working"}


def test_health_endpoint_reports_models(client):
    response = client.get("/health")
    payload = response.json()

    assert response.status_code == 200
    assert payload["status"] == "healthy"
    assert payload["components"]["database"] == "connected"
    assert payload["components"]["openai_models"] == {
        "response": "gpt-5.4",
        "query_rewriter": "gpt-5.4-mini",
        "reranker": "gpt-5.4-nano",
    }


def test_telegram_endpoint_removed(client):
    response = client.post(
        "/telegram-chat",
        json={"question": "hello", "language": "en", "previous_chats": []},
    )

    assert response.status_code == 404


def test_websocket_invalid_payload_returns_validation_error(client):
    with client.websocket_connect("/chat") as websocket:
        connected = websocket.receive_json()
        assert connected["status"] == "connected"

        websocket.send_json({"invalid": "payload"})

        received = websocket.receive_json()
        assert received["status"] == "received"

        validation_error = websocket.receive_json()
        assert validation_error["error"] == "validation_error"
        assert validation_error["sources"] == []
