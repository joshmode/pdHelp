import sys
from unittest.mock import MagicMock, patch

# --- step 0: ensure app modules are not already loaded ---
for module_name in list(sys.modules.keys()):
    if module_name.startswith("app.") or module_name == "app":
        del sys.modules[module_name]

# --- step 1: mock heavy libraries before importing app.main or app.rag ---
mocks = {
    "langchain_community.document_loaders": MagicMock(),
    "langchain.text_splitter": MagicMock(),
    "langchain_text_splitters": MagicMock(),
    "langchain_community.vectorstores": MagicMock(),
    "langchain_community.embeddings": MagicMock(),
    "langchain_community.llms": MagicMock(),
    "langchain.chains": MagicMock(),
    "chromadb": MagicMock(),
    "chromadb.config": MagicMock(),
    "langchain_chroma": MagicMock(),
    "ctransformers": MagicMock(),
    "sentence-transformers": MagicMock(),
}
sys.modules.update(mocks)

# --- step 3: import the app ---
from fastapi.testclient import TestClient
from app.main import app

def test_serve_frontend():
    # we need to ensure rag_engine is properly mocked for lifespan
    with patch("app.main.rag_engine") as mock_engine:
        mock_engine.initialize.return_value = None
        mock_engine.vector_store = MagicMock()
        mock_engine.llm = MagicMock()

        with TestClient(app) as client:
            response = client.get("/")
            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]
            assert "<!DOCTYPE html>" in response.text

def test_health_check_moved():
    with patch("app.main.rag_engine") as mock_engine:
        mock_engine.initialize.return_value = None
        mock_engine.vector_store = MagicMock()
        mock_engine.llm = MagicMock()

        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "ok"
