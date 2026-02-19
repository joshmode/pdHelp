import sys
import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_dependencies():
    # --- step 0: ensure app modules are not already loaded ---
    for module_name in list(sys.modules.keys()):
        if module_name.startswith("app.") or module_name == "app":
            del sys.modules[module_name]

    # --- step 1: mock heavy libraries ---
    sys.modules["langchain_community.document_loaders"] = MagicMock()
    sys.modules["langchain.text_splitter"] = MagicMock()
    sys.modules["langchain_text_splitters"] = MagicMock()
    sys.modules["langchain_community.vectorstores"] = MagicMock()
    sys.modules["langchain_community.embeddings"] = MagicMock()
    sys.modules["langchain_community.llms"] = MagicMock()
    sys.modules["langchain.chains"] = MagicMock()
    sys.modules["chromadb"] = MagicMock()
    sys.modules["langchain_chroma"] = MagicMock()
    sys.modules["ctransformers"] = MagicMock()
    sys.modules["sentence-transformers"] = MagicMock()
    sys.modules["fastapi"] = MagicMock()
    sys.modules["fastapi.responses"] = MagicMock()
    sys.modules["pydantic"] = MagicMock()

    class FalsyWrapper:
        def __init__(self):
            self.mock = MagicMock()
        def __bool__(self):
            return False
        def __getattr__(self, name):
            return getattr(self.mock, name)

    falsy_instance = FalsyWrapper()

    mock_chroma_class = MagicMock(return_value=falsy_instance)
    sys.modules["langchain_chroma"].Chroma = mock_chroma_class

    mock_llm_class = MagicMock(return_value=MagicMock())
    sys.modules["langchain_community.llms"].CTransformers = mock_llm_class

    sys.modules["langchain_community.embeddings"].HuggingFaceEmbeddings = MagicMock()

    yield

    # cleanup
    for module_name in list(sys.modules.keys()):
        if module_name.startswith("app.") or module_name == "app":
            del sys.modules[module_name]

def test_startup_success_with_falsy_objects(mock_dependencies, capsys):
    import app.rag
    from app.rag import rag_engine

    rag_engine._download_model_if_needed = MagicMock()

    rag_engine.initialize()

    captured = capsys.readouterr()
    print(captured.out)

    assert "rag engine initialized successfully" in captured.out
    assert "initialized but components are missing" not in captured.out

    assert rag_engine.vector_store is not None
    assert bool(rag_engine.vector_store) is False

    if rag_engine.vector_store is None or rag_engine.llm is None:
        raise RuntimeError("failed to wake up properly")
