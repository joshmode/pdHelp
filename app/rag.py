import os
from typing import List

import requests
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

MODEL_URL = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_PATH = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
EMBEDDING_NAME = "all-MiniLM-L6-v2"
MEMORY_PATH = "data/chroma_db"


class RagEngine:
    def __init__(self):
        self.vector_store = None
        self.llm = None
        self._embeddings_tool = None

    def initialize(self):
        try:
            self._download_model_if_needed()

            print(f"loading embedding tool: {EMBEDDING_NAME}")
            try:
                self._embeddings_tool = HuggingFaceEmbeddings(model_name=EMBEDDING_NAME)
            except Exception as e:
                print(f"error loading embeddings: {e}")
                raise

            print(f"connecting to vector store at {MEMORY_PATH}")
            try:
                self.vector_store = Chroma(
                    persist_directory=MEMORY_PATH,
                    embedding_function=self._embeddings_tool,
                )
            except Exception as e:
                print(f"error connecting to vector store: {e}")
                self.vector_store = None
                raise

            print(f"loading llm from {MODEL_PATH}")
            try:
                self.llm = CTransformers(
                    model=MODEL_PATH,
                    model_type="llama",
                    config={"max_new_tokens": 256, "temperature": 0.5, "context_length": 2048},
                )
            except Exception as e:
                print(f"error loading llm: {e}")
                self.llm = None
                raise

            if self.vector_store is not None and self.llm is not None:
                print("rag engine initialized successfully")
            else:
                print("rag engine initialized but components are missing")

        except Exception as e:
            print(f"rag engine failed to initialize: {e}")
            raise

    def _download_model_if_needed(self):
        if not os.path.exists("models"):
            os.makedirs("models", exist_ok=True)

        if os.path.exists(MODEL_PATH):
            if os.path.getsize(MODEL_PATH) < 100 * 1024 * 1024:
                print(f"model file found but too small ({os.path.getsize(MODEL_PATH)} bytes). redownloading...")
                os.remove(MODEL_PATH)
            else:
                print("model found locally")
                return

        print(f"model not found locally. downloading from {MODEL_URL}...")
        temp_model_path = MODEL_PATH + ".part"
        try:
            response = requests.get(MODEL_URL, stream=True, timeout=60)
            if response.status_code == 200:
                with open(temp_model_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                os.rename(temp_model_path, MODEL_PATH)
                print("model downloaded successfully")
            else:
                raise Exception(f"failed to download model. status: {response.status_code}")
        except Exception as e:
            print(f"error downloading model: {e}")
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)
            raise

    def process_document(self, file_path: str) -> List:
        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=80)

        raw_pages = PyPDFLoader(file_path).load()
        text_pages = [page for page in raw_pages if getattr(page, "page_content", "").strip()]
        if text_pages:
            chunks = splitter.split_documents(text_pages)
            chunks = [chunk for chunk in chunks if getattr(chunk, "page_content", "").strip()]
            if chunks:
                return chunks

        fallback_text = self._extract_text_with_pypdf(file_path)
        if not fallback_text.strip():
            return []

        return splitter.create_documents([fallback_text])

    def _extract_text_with_pypdf(self, file_path: str) -> str:
        try:
            reader = PdfReader(file_path)
            if reader.is_encrypted:
                reader.decrypt("")

            page_text = []
            for page in reader.pages:
                page_text.append(page.extract_text() or "")
            return "\n".join(page_text)
        except Exception:
            return ""

    def add_documents(self, documents: List):
        if self.vector_store is None:
            raise RuntimeError("rag engine not initialized")
        self.vector_store.add_documents(documents)

    def query(self, question: str) -> str:
        if self.vector_store is None or self.llm is None:
            raise RuntimeError("rag engine not initialized")

        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
        )

        try:
            result = qa_chain.invoke(question)
            return result.get("result", "no answer found")
        except Exception as e:
            print(f"error during qa: {e}")
            return "error processing request"


rag_engine = RagEngine()
