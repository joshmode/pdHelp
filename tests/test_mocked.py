import sys
from unittest.mock import MagicMock

# --- step 0: ensure app modules are not already loaded ---
for module_name in list(sys.modules.keys()):
    if module_name.startswith("app.") or module_name == "app":
        del sys.modules[module_name]

# --- step 1: mock heavy libraries before importing app.main or app.rag ---
sys.modules["langchain_community.document_loaders"] = MagicMock()
sys.modules["langchain.text_splitter"] = MagicMock()
sys.modules["langchain_text_splitters"] = MagicMock()
sys.modules["langchain_community.vectorstores"] = MagicMock()
sys.modules["langchain_community.embeddings"] = MagicMock()
sys.modules["langchain_community.llms"] = MagicMock()
sys.modules["langchain.chains"] = MagicMock()
sys.modules["chromadb"] = MagicMock()
sys.modules["chromadb.config"] = MagicMock()
sys.modules["langchain_chroma"] = MagicMock()
sys.modules["ctransformers"] = MagicMock()
sys.modules["sentence-transformers"] = MagicMock()

# --- step 2: configure the mocks ---
mock_doc = MagicMock()
mock_doc.page_content = "doc1 content"
mock_loader = MagicMock()
mock_loader.load.return_value = [mock_doc]
sys.modules["langchain_community.document_loaders"].PyPDFLoader.return_value = mock_loader

mock_splitter = MagicMock()
mock_splitter.split_documents.return_value = ["chunk1", "chunk2"]
sys.modules["langchain.text_splitter"].RecursiveCharacterTextSplitter.return_value = mock_splitter
sys.modules["langchain_text_splitters"].RecursiveCharacterTextSplitter.return_value = mock_splitter

mock_chroma = MagicMock()
# mock the retrieval chain result
mock_qa_chain = MagicMock()
mock_qa_chain.invoke.return_value = {"result": "the answer is 42."}

# we need to mock retrievalqa.from_chain_type to return our mock_qa_chain
sys.modules["langchain.chains"].RetrievalQA.from_chain_type.return_value = mock_qa_chain

# --- step 3: import the app ---
# now we can safely import the app code. it will use the mocked modules.
from fastapi.testclient import TestClient
from app.main import app
from app.rag import rag_engine
from app import rag # import the module to access its globals (which are our mocks)

# --- step 4: write the tests ---
client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "application is running"}

def test_upload_pdf(tmp_path):
    file_content = b"%PDF-1.4 dummy content"
    file_name = "test.pdf"

    original_process = rag_engine.process_document
    original_add = rag_engine.add_documents

    rag_engine.process_document = MagicMock(return_value=["chunk1"])
    rag_engine.add_documents = MagicMock()

    try:
        response = client.post(
            "/upload",
            files={"file": (file_name, file_content, "application/pdf")}
        )

        assert response.status_code == 200
        assert response.json() == {"message": "document processed successfully", "filename": file_name}

        rag_engine.process_document.assert_called_once()
        rag_engine.add_documents.assert_called_once_with(["chunk1"])

    finally:
        rag_engine.process_document = original_process
        rag_engine.add_documents = original_add

def test_upload_empty_content_pdf():
    file_content = b"%PDF-1.4 dummy content"
    file_name = "empty_content.pdf"

    original_process = rag_engine.process_document
    original_add = rag_engine.add_documents

    # mock process to return empty list
    rag_engine.process_document = MagicMock(return_value=[])
    rag_engine.add_documents = MagicMock()

    try:
        response = client.post(
            "/upload",
            files={"file": (file_name, file_content, "application/pdf")}
        )
        assert response.status_code == 400
        assert response.json() == {"detail": "document appears to be empty or unreadable."}

        rag_engine.process_document.assert_called_once()
        rag_engine.add_documents.assert_not_called()
    finally:
        rag_engine.process_document = original_process
        rag_engine.add_documents = original_add

def test_upload_invalid_file():
    response = client.post(
        "/upload",
        files={"file": ("test.txt", b"text content", "text/plain")}
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "only pdf files are supported."}

def test_query():
    original_query = rag_engine.query
    rag_engine.query = MagicMock(return_value="The capital of France is Paris.")

    try:
        response = client.post(
            "/query",
            json={"text": "What is the capital of France?"}
        )

        assert response.status_code == 200
        assert response.json() == {"reply": "The capital of France is Paris."}

        rag_engine.query.assert_called_once_with("What is the capital of France?")

    finally:
        rag_engine.query = original_query

def test_rag_logic_mocked():
    """
    test the actual logic inside ragengine using the mocked libraries.
    """
    # 1. test process_document
    file_path = "dummy/path.pdf"
    chunks = rag_engine.process_document(file_path)

    # check if pypdfloader was instantiated with file_path
    rag.PyPDFLoader.assert_called_with(file_path)
    # check if loader.load() was called
    rag.PyPDFLoader.return_value.load.assert_called()
    # check if splitter was used
    rag.RecursiveCharacterTextSplitter.return_value.split_documents.assert_called()

    # 2. test add_documents (requires initialization first, or mocking vector_store)
    rag_engine.vector_store = MagicMock()
    rag_engine.add_documents(["chunk1"])
    rag_engine.vector_store.add_documents.assert_called_with(["chunk1"])

    # 3. test query
    rag_engine.llm = MagicMock()
    answer = rag_engine.query("Question")
    assert answer == "the answer is 42." # from our mock_qa_chain setup above

def test_upload_pdf_case_insensitive():
    file_content = b"%PDF-1.4 dummy content"
    file_name = "test.PDF"

    original_process = rag_engine.process_document
    original_add = rag_engine.add_documents
    rag_engine.process_document = MagicMock(return_value=["chunk1"])
    rag_engine.add_documents = MagicMock()

    try:
        response = client.post(
            "/upload",
            files={"file": (file_name, file_content, "application/pdf")}
        )
        assert response.status_code == 200
    finally:
        rag_engine.process_document = original_process
        rag_engine.add_documents = original_add

def test_upload_empty_pdf():
    file_content = b""
    file_name = "empty.pdf"

    response = client.post(
        "/upload",
        files={"file": (file_name, file_content, "application/pdf")}
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "uploaded file is empty."}

def test_upload_processing_failure():
    file_content = b"%PDF-1.4 dummy content"
    file_name = "test.pdf"

    original_process = rag_engine.process_document
    rag_engine.process_document = MagicMock(side_effect=Exception("corrupted pdf"))

    try:
        response = client.post(
            "/upload",
            files={"file": (file_name, file_content, "application/pdf")}
        )
        assert response.status_code == 500
        assert "error processing file" in response.json()["detail"]
    finally:
        rag_engine.process_document = original_process

def test_query_engine_not_initialized():
    original_llm = rag_engine.llm
    original_vs = rag_engine.vector_store

    rag_engine.llm = None
    rag_engine.vector_store = None

    try:
        response = client.post(
            "/query",
            json={"text": "Hello?"}
        )
        assert response.status_code == 500
        assert "rag engine not initialized" in response.json()["detail"]
    finally:
        rag_engine.llm = original_llm
        rag_engine.vector_store = original_vs

def test_query_internal_error_handled():
    original_llm = rag_engine.llm
    original_vs = rag_engine.vector_store

    rag_engine.llm = MagicMock()
    rag_engine.vector_store = MagicMock()

    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = Exception("inference failed")

    # access retrievalqa from rag module to ensure we modify the correct mock
    rag.RetrievalQA.from_chain_type.return_value = mock_chain

    try:
        response = client.post(
            "/query",
            json={"text": "Hello?"}
        )

        assert response.status_code == 200
        assert response.json() == {"reply": "error processing request"}

    finally:
        rag_engine.llm = original_llm
        rag_engine.vector_store = original_vs
        rag.RetrievalQA.from_chain_type.return_value = mock_qa_chain
