import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# --- step 0: ensure app modules are not already loaded ---
for module_name in list(sys.modules.keys()):
    if module_name.startswith("app.") or module_name == "app":
        del sys.modules[module_name]

# --- mock heavy libraries before importing app.rag ---
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

# now import the module to test
from app.rag import rag_engine, MODEL_PATH

def test_download_model_success():
    # we patch requests.get because app.rag uses requests.get directly
    with patch("requests.get") as mock_get, \
         patch("builtins.open", new_callable=MagicMock) as mock_open, \
         patch("os.path.exists") as mock_exists, \
         patch("os.makedirs") as mock_makedirs, \
         patch("os.rename") as mock_rename, \
         patch("os.remove") as mock_remove:

        # scenario: model does not exist
        def exists_side_effect(path):
            if path == MODEL_PATH:
                return False
            return True # models dir exists

        mock_exists.side_effect = exists_side_effect

        # mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
        mock_get.return_value = mock_response

        # mock file writing
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        rag_engine._download_model_if_needed()

        # check if downloaded
        mock_get.assert_called()

        # check if wrote to temp file
        temp_path = MODEL_PATH + ".part"
        mock_open.assert_called_with(temp_path, "wb")

        mock_file.write.assert_any_call(b"chunk1")
        mock_file.write.assert_any_call(b"chunk2")

        # check if renamed
        mock_rename.assert_called_with(temp_path, MODEL_PATH)

def test_download_model_failure_cleanup():
    """
    test that if download fails, the temporary file is removed.
    """
    with patch("requests.get") as mock_get, \
         patch("builtins.open", new_callable=MagicMock) as mock_open, \
         patch("os.path.exists") as mock_exists, \
         patch("os.makedirs"), \
         patch("os.rename") as mock_rename, \
         patch("os.remove") as mock_remove:

        # model does not exist
        def exists_side_effect(path):
            if path == MODEL_PATH:
                return False
            # during cleanup, it checks if temp path exists
            if path == MODEL_PATH + ".part":
                return True
            return True

        mock_exists.side_effect = exists_side_effect

        # mock response that fails during iteration
        mock_response = MagicMock()
        mock_response.status_code = 200

        def iter_fail(*args, **kwargs):
            yield b"chunk1"
            raise ConnectionError("connection lost")

        mock_response.iter_content.side_effect = iter_fail
        mock_get.return_value = mock_response

        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        try:
            rag_engine._download_model_if_needed()
        except ConnectionError:
            pass

        # verify if cleanup happened
        temp_path = MODEL_PATH + ".part"
        mock_remove.assert_called_with(temp_path)

        # verify rename was not called
        mock_rename.assert_not_called()

def test_download_model_non_200_status():
    """
    test that if response status code is not 200, an exception is raised
    and the temporary file is removed.
    """
    with patch("requests.get") as mock_get, \
         patch("builtins.open", new_callable=MagicMock) as mock_open, \
         patch("os.path.exists") as mock_exists, \
         patch("os.makedirs"), \
         patch("os.rename") as mock_rename, \
         patch("os.remove") as mock_remove:

        # model does not exist
        def exists_side_effect(path):
            if path == MODEL_PATH:
                return False
            # during cleanup, it checks if temp path exists
            if path == MODEL_PATH + ".part":
                return True
            return True

        mock_exists.side_effect = exists_side_effect

        # mock response with 404
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        # check that exception is raised
        with pytest.raises(Exception, match="failed to download model. status: 404"):
            rag_engine._download_model_if_needed()

        # verify if cleanup happened
        temp_path = MODEL_PATH + ".part"
        mock_remove.assert_called_with(temp_path)

        # verify file operations were not called
        mock_open.assert_not_called()
        mock_rename.assert_not_called()
