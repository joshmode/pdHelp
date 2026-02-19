import sys
from unittest.mock import MagicMock, patch
import pytest

@pytest.fixture(autouse=True)
def reset_mocks():
    # --- step 0: ensure app modules are not already loaded ---
    for module_name in list(sys.modules.keys()):
        if module_name.startswith("app.") or module_name == "app":
            del sys.modules[module_name]

    # mock heavy libraries before importing app.rag
    # we create new mocks for each test run to avoid state leakage
    mocks = {
        "langchain_community.document_loaders": MagicMock(),
        "langchain.text_splitter": MagicMock(),
        "langchain_text_splitters": MagicMock(),
        "langchain_community.vectorstores": MagicMock(),
        "langchain_community.embeddings": MagicMock(),
        "langchain_community.llms": MagicMock(),
        "langchain.chains": MagicMock(),
        "chromadb": MagicMock(),
        "langchain_chroma": MagicMock(),
        "ctransformers": MagicMock(),
        "sentence-transformers": MagicMock(),
    }
    sys.modules.update(mocks)

    yield

def test_initialize_success(capsys):
    from app.rag import RagEngine
    engine = RagEngine()
    engine._download_model_if_needed = MagicMock()

    mock_embeddings = MagicMock()
    sys.modules["langchain_community.embeddings"].HuggingFaceEmbeddings.return_value = mock_embeddings

    mock_chroma = MagicMock()
    sys.modules["langchain_chroma"].Chroma.return_value = mock_chroma

    mock_llm = MagicMock()
    sys.modules["langchain_community.llms"].CTransformers.return_value = mock_llm

    engine.initialize()

    assert engine.vector_store == mock_chroma
    assert engine.llm == mock_llm

    captured = capsys.readouterr()
    assert "rag engine initialized successfully" in captured.out

def test_initialize_embeddings_failure(capsys):
    from app.rag import RagEngine
    engine = RagEngine()
    engine._download_model_if_needed = MagicMock()

    sys.modules["langchain_community.embeddings"].HuggingFaceEmbeddings.side_effect = Exception("embeddings failed")

    with pytest.raises(Exception) as excinfo:
        engine.initialize()

    assert "embeddings failed" in str(excinfo.value)

    captured = capsys.readouterr()
    assert "error loading embeddings: embeddings failed" in captured.out
    assert "rag engine failed to initialize" in captured.out

def test_initialize_vector_store_failure(capsys):
    from app.rag import RagEngine
    engine = RagEngine()
    engine._download_model_if_needed = MagicMock()

    # ensure embeddings succeeds
    sys.modules["langchain_community.embeddings"].HuggingFaceEmbeddings.side_effect = None
    sys.modules["langchain_community.embeddings"].HuggingFaceEmbeddings.return_value = MagicMock()

    # mock failure in chroma
    sys.modules["langchain_chroma"].Chroma.side_effect = Exception("chroma failed")

    with pytest.raises(Exception) as excinfo:
        engine.initialize()

    assert "chroma failed" in str(excinfo.value)
    assert engine.vector_store is None

    captured = capsys.readouterr()
    assert "error connecting to vector store: chroma failed" in captured.out

def test_initialize_llm_failure(capsys):
    from app.rag import RagEngine
    engine = RagEngine()
    engine._download_model_if_needed = MagicMock()

    # ensure others succeed
    sys.modules["langchain_community.embeddings"].HuggingFaceEmbeddings.side_effect = None
    sys.modules["langchain_chroma"].Chroma.side_effect = None

    # mock failure in ctransformers
    sys.modules["langchain_community.llms"].CTransformers.side_effect = Exception("brain failed")

    with pytest.raises(Exception) as excinfo:
        engine.initialize()

    assert "brain failed" in str(excinfo.value)
    assert engine.llm is None

    captured = capsys.readouterr()
    assert "error loading llm: brain failed" in captured.out

@pytest.mark.asyncio
async def test_lifespan_success():
    # we need to import lifespan here to ensure it uses the mocked app.rag
    from app.main import lifespan

    with patch("app.main.rag_engine") as mock_engine:
        mock_engine.initialize.return_value = None
        mock_engine.vector_store = MagicMock()
        mock_engine.llm = MagicMock()

        async with lifespan(MagicMock()):
            pass

        mock_engine.initialize.assert_called_once()

@pytest.mark.asyncio
async def test_lifespan_failure_initialize():
    from app.main import lifespan

    with patch("app.main.rag_engine") as mock_engine:
        mock_engine.initialize.side_effect = Exception("init failed")

        with pytest.raises(Exception) as excinfo:
            async with lifespan(MagicMock()):
                pass

        assert "init failed" in str(excinfo.value)

@pytest.mark.asyncio
async def test_lifespan_failure_vector_store_none():
    from app.main import lifespan

    with patch("app.main.rag_engine") as mock_engine:
        mock_engine.initialize.return_value = None
        mock_engine.vector_store = None
        mock_engine.llm = MagicMock()

        with pytest.raises(RuntimeError) as excinfo:
            async with lifespan(MagicMock()):
                pass

        assert "initialization failed" in str(excinfo.value)
