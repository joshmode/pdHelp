import sys
from unittest.mock import MagicMock, patch
import pytest

# Fixture to provide a clean environment and mock heavy dependencies
@pytest.fixture(autouse=True)
def mock_dependencies():
    # We must preserve the original modules so we can restore them
    original_modules = sys.modules.copy()

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
        "requests": MagicMock(),
        "pypdf": MagicMock(),
    }

    # Apply the mocks to sys.modules
    sys.modules.update(mocks)

    # We need to clear app.rag if it was already imported,
    # so it gets re-imported using our mocked sys.modules
    if "app.rag" in sys.modules:
        del sys.modules["app.rag"]

    yield

    # Restore the original modules state
    sys.modules.clear()
    sys.modules.update(original_modules)
    if "app.rag" in sys.modules:
        del sys.modules["app.rag"]

def test_extract_text_with_pypdf_success():
    from app.rag import RagEngine
    engine = RagEngine()

    mock_page1 = MagicMock()
    mock_page1.extract_text.return_value = "Page 1 text"

    mock_page2 = MagicMock()
    mock_page2.extract_text.return_value = "Page 2 text"

    mock_reader_instance = MagicMock()
    mock_reader_instance.is_encrypted = False
    mock_reader_instance.pages = [mock_page1, mock_page2]

    mock_pdf_reader_class = MagicMock(return_value=mock_reader_instance)
    sys.modules["pypdf"].PdfReader = mock_pdf_reader_class

    result = engine._extract_text_with_pypdf("dummy.pdf")

    assert result == "Page 1 text\nPage 2 text"
    mock_pdf_reader_class.assert_called_once_with("dummy.pdf")

def test_extract_text_with_pypdf_encrypted():
    from app.rag import RagEngine
    engine = RagEngine()

    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Decrypted text"

    mock_reader_instance = MagicMock()
    mock_reader_instance.is_encrypted = True
    mock_reader_instance.pages = [mock_page]

    mock_pdf_reader_class = MagicMock(return_value=mock_reader_instance)
    sys.modules["pypdf"].PdfReader = mock_pdf_reader_class

    result = engine._extract_text_with_pypdf("encrypted.pdf")

    assert result == "Decrypted text"
    mock_reader_instance.decrypt.assert_called_once_with("")
    mock_pdf_reader_class.assert_called_once_with("encrypted.pdf")

def test_extract_text_with_pypdf_none_text():
    from app.rag import RagEngine
    engine = RagEngine()

    mock_page = MagicMock()
    mock_page.extract_text.return_value = None

    mock_reader_instance = MagicMock()
    mock_reader_instance.is_encrypted = False
    mock_reader_instance.pages = [mock_page]

    mock_pdf_reader_class = MagicMock(return_value=mock_reader_instance)
    sys.modules["pypdf"].PdfReader = mock_pdf_reader_class

    result = engine._extract_text_with_pypdf("none.pdf")

    assert result == ""
    mock_pdf_reader_class.assert_called_once_with("none.pdf")

def test_extract_text_with_pypdf_exception():
    from app.rag import RagEngine
    engine = RagEngine()

    mock_pdf_reader_class = MagicMock(side_effect=Exception("File not found"))
    sys.modules["pypdf"].PdfReader = mock_pdf_reader_class

    result = engine._extract_text_with_pypdf("error.pdf")

    assert result == ""
    mock_pdf_reader_class.assert_called_once_with("error.pdf")
